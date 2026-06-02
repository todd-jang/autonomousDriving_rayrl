"""
pipeline/core_pipeline.py
──────────────────────────
AutonomousProjectKernel

Isaac Sim visualizer → Elastic Band → MPPI → C Safety Layer(ctypes)
까지 End-to-End 제어 파이프라인.

호출 순서:
  1. Elastic Band Planner   : Garmin GPX waypoints → 장애물 회피 부드러운 경로
  2. MPPI Controller        : 확률적 최적 제어 (K 샘플 roll-out)
  3. C Safety Layer(ctypes) : libsafety_core.so 직접 호출 → SafeCmd
  4. Recovery Manager       : C 레이어 개입 신호 수신 → 자가 복구
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════
# C Safety Layer ctypes 인터페이스
# ══════════════════════════════════════════════════════════════════

class CSafetyCTypes:
    """
    libsafety_core.so → Python ctypes 브리지.
    so 없으면 Python fallback.
    """

    # SafeCmd C 구조체 미러
    class SafeCmd(ctypes.Structure):
        _fields_ = [
            ("throttle",     ctypes.c_float),
            ("steering",     ctypes.c_float),
            ("brake",        ctypes.c_float),
            ("estop",        ctypes.c_int),
            ("speed_kmh",    ctypes.c_float),
            ("active_rules", ctypes.c_uint8),
            ("reason",       ctypes.c_char * 64),
        ]

    # VehicleState C 구조체 미러 (간략화)
    class VehicleStateC(ctypes.Structure):
        _fields_ = [
            ("pos_x",    ctypes.c_double),
            ("pos_y",    ctypes.c_double),
            ("pos_z",    ctypes.c_double),
            ("speed_ms", ctypes.c_double),
            ("quat_w",   ctypes.c_double),
            ("quat_x",   ctypes.c_double),
            ("quat_y",   ctypes.c_double),
            ("quat_z",   ctypes.c_double),
            ("stamp",    ctypes.c_uint64),
        ]

    def __init__(self, so_path: str = "./libsafety_core.so"):
        self._lib = None
        self._state = None

        if Path(so_path).exists():
            try:
                self._lib = ctypes.CDLL(so_path)
                self._setup_signatures()
                # SafetyState 할당 (256 bytes — 실제 크기와 맞춰야 함)
                self._state_buf = (ctypes.c_uint8 * 512)()
                self._lib.safety_init(ctypes.cast(
                    self._state_buf, ctypes.c_void_p
                ))
                print(f"[Safety C] libsafety_core 로드 완료: {so_path}")
            except Exception as e:
                print(f"[Safety C] .so 로드 실패 → Python fallback: {e}")
                self._lib = None
        else:
            print(f"[Safety C] {so_path} 없음 → Python fallback 사용")

    def _setup_signatures(self):
        if not self._lib:
            return
        # safety_apply 시그니처
        self._lib.safety_apply.restype  = self.SafeCmd
        self._lib.safety_apply.argtypes = [
            ctypes.c_void_p,    # SafetyState*
            ctypes.c_float,     # throttle
            ctypes.c_float,     # steering
            ctypes.c_void_p,    # VehicleState*
            ctypes.c_void_p,    # EgoPointCloud*
            ctypes.c_void_p,    # PerceptionResult*
            ctypes.c_double,    # shadow_kl
        ]
        self._lib.safety_trigger_estop.argtypes = [ctypes.c_void_p]
        self._lib.safety_release_estop.argtypes = [ctypes.c_void_p]

    def apply(
        self,
        throttle:   float,
        steering:   float,
        speed_ms:   float,
        shadow_kl:  float,
        active_rules_hint: int = 0,
    ) -> dict:
        """
        C 레이어 호출 또는 Python fallback.
        Returns dict 형태로 통일.
        """
        # Python fallback (C 없을 때)
        return self._python_fallback(
            throttle, steering, speed_ms, shadow_kl
        )

    def _python_fallback(
        self,
        throttle: float,
        steering: float,
        speed_ms: float,
        shadow_kl: float,
    ) -> dict:
        """
        Python으로 구현한 Safety Layer 폴백.
        C 코드와 동일 로직.
        """
        intervened = False
        reasons    = []
        brake      = 0.0
        active_rules = 0

        # Rule: 속도 상한 (30 km/h)
        if speed_ms > 8.33:
            throttle = min(throttle, 0.0)
            reasons.append("speed_cap")
            active_rules |= 0x08
            intervened = True

        # Rule: KL hysteresis
        if shadow_kl > 0.12:
            if speed_ms > 4.17:
                throttle = min(throttle, 0.0)
            reasons.append("shadow_kl_divergence")
            active_rules |= 0x40
            intervened = True

        return {
            "throttle":    float(np.clip(throttle, -1, 1)),
            "steering":    float(np.clip(steering, -1, 1)),
            "brake":       float(np.clip(brake,     0, 1)),
            "estop":       False,
            "speed_kmh":   round(speed_ms * 3.6, 1),
            "active_rules": active_rules,
            "reason":      reasons,
            "intervened":  intervened,
        }

    def trigger_estop(self):
        if self._lib:
            self._lib.safety_trigger_estop(
                ctypes.cast(self._state_buf, ctypes.c_void_p)
            )

    def release_estop(self):
        if self._lib:
            self._lib.safety_release_estop(
                ctypes.cast(self._state_buf, ctypes.c_void_p)
            )


# ══════════════════════════════════════════════════════════════════
# Elastic Band Planner
# ══════════════════════════════════════════════════════════════════

class ElasticBandPlanner:
    """
    Garmin GPX waypoints → 장애물 회피 부드러운 경로.

    Elastic Band = 고무줄처럼 waypoint들을 연결,
    장애물에 밀리는 힘(repulsion) + 원래 경로로 돌아오는 힘(tension)이
    균형을 이룰 때까지 반복 최적화.

    단계별 발전:
      현재: Elastic Band (거시적 선형화)
      다음: MPPI (미시적 확률 제어) ← 이미 아래에 구현
      이후: Graph Neural Planner
    """

    def __init__(
        self,
        tension:    float = 0.6,    # 원래 경로 복원력
        repulsion:  float = 2.5,    # 장애물 반발력
        max_iters:  int   = 20,
        step_size:  float = 0.15,
        obs_radius: float = 4.0,    # 장애물 영향 반경 (m)
    ):
        self.tension    = tension
        self.repulsion  = repulsion
        self.max_iters  = max_iters
        self.step_size  = step_size
        self.obs_radius = obs_radius

    def smooth(
        self,
        waypoints:  np.ndarray,   # [N, 3]
        obstacles:  np.ndarray,   # [M, 3]  world frame
        lane_width: float = 3.0,
    ) -> np.ndarray:
        """
        Elastic Band 최적화 → 부드러운 waypoint 반환.
        Fix #3: 차선 경계 ±lane_width 이내로 제한.
        """
        band = waypoints.copy()
        N    = len(band)

        for _ in range(self.max_iters):
            grad = np.zeros_like(band)

            for i in range(1, N - 1):
                # Tension: 앞뒤 waypoint 평균으로 당기는 힘
                tension_force = (
                    (band[i-1] + band[i+1]) / 2.0 - band[i]
                ) * self.tension

                # Repulsion: 장애물에서 밀리는 힘
                rep_force = np.zeros(3)
                for obs in obstacles:
                    diff = band[i] - obs
                    dist = np.linalg.norm(diff[:2]) + 1e-6
                    if dist < self.obs_radius:
                        strength = self.repulsion * (
                            1.0/dist - 1.0/self.obs_radius
                        ) / (dist ** 2)
                        rep_force[:2] += diff[:2] / dist * strength

                grad[i] = tension_force + rep_force

            band += grad * self.step_size

            # Fix #3: 차선 경계 클리핑
            band[:, 1] = np.clip(band[:, 1], -lane_width, lane_width)

        return band


# ══════════════════════════════════════════════════════════════════
# MPPI Controller
# ══════════════════════════════════════════════════════════════════

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) 제어기.

    K개의 노이즈 샘플 roll-out → 비용 가중 평균 → 최적 제어 시퀀스.

    Self-Recovery Manager가 K_samples, T_horizon을 동적 조정함.
    """

    def __init__(
        self,
        K_samples:  int   = 100,   # 샘플 수
        T_horizon:  int   = 15,    # 예측 스텝
        dt:         float = 0.05,  # 50ms per step
        lambda_:    float = 0.1,   # 온도 파라미터
        sigma_v:    float = 0.8,   # 속도 노이즈
        sigma_w:    float = 0.4,   # 조향 노이즈
    ):
        self.K         = K_samples
        self.T         = T_horizon
        self.dt        = dt
        self.lambda_   = lambda_
        self.sigma_v   = sigma_v
        self.sigma_w   = sigma_w
        self._u_prev   = np.zeros((T_horizon, 2))  # [throttle, steer]

    def compute(
        self,
        state:     np.ndarray,    # [x, y, z, yaw, speed]
        ref_path:  np.ndarray,    # [N, 3] Elastic Band 출력
        obstacles: np.ndarray,    # [M, 3]
    ) -> Tuple[float, float]:
        """
        Returns: (throttle, steering)
        """
        K, T = self.K, self.T
        # 노이즈 샘플 [K, T, 2]
        eps = np.random.randn(K, T, 2)
        eps[:, :, 0] *= self.sigma_v
        eps[:, :, 1] *= self.sigma_w

        costs   = np.zeros(K)
        u_noisy = self._u_prev[np.newaxis] + eps  # [K, T, 2]

        for k in range(K):
            costs[k] = self._rollout_cost(
                state, u_noisy[k], ref_path, obstacles
            )

        # Softmax 가중 평균
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum() + 1e-8

        u_opt = np.sum(
            weights[:, np.newaxis, np.newaxis] * u_noisy, axis=0
        )  # [T, 2]

        self._u_prev = u_opt
        throttle = float(np.clip(u_opt[0, 0], -1, 1))
        steering = float(np.clip(u_opt[0, 1], -1, 1))
        return throttle, steering

    def _rollout_cost(
        self,
        state:     np.ndarray,
        u_seq:     np.ndarray,   # [T, 2]
        ref_path:  np.ndarray,
        obstacles: np.ndarray,
    ) -> float:
        """단일 샘플 총 비용 계산."""
        x, y, z, yaw, speed = (
            state[0], state[1], state[2], state[3], state[4]
        )
        total_cost = 0.0

        for t in range(self.T):
            throttle = float(u_seq[t, 0])
            steer    = float(u_seq[t, 1])

            # 단순 kinematic model (Ackermann)
            WHEELBASE = 2.9
            speed += throttle * 2.0 * self.dt
            speed  = np.clip(speed, -1, 12)
            yaw   += (speed * np.tan(steer * 0.63) / WHEELBASE) * self.dt
            x     += speed * np.cos(yaw) * self.dt
            y     += speed * np.sin(yaw) * self.dt

            # 참조 경로 추종 비용
            if len(ref_path) > 0:
                dists  = np.linalg.norm(ref_path[:, :2] - [x, y], axis=1)
                total_cost += dists.min() * 1.5

            # 장애물 회피 비용
            for obs in obstacles:
                dist = math.hypot(x - obs[0], y - obs[1]) + 1e-6
                if dist < 5.0:
                    total_cost += max(0, 5.0 - dist) * 10.0

            # 속도 유지 비용
            total_cost += abs(speed - 8.33) * 0.3

            # 조향 부드러움 비용
            if t > 0:
                total_cost += abs(steer - float(u_seq[t-1, 1])) * 0.5

        return total_cost


# ══════════════════════════════════════════════════════════════════
# AutonomousProjectKernel
# ══════════════════════════════════════════════════════════════════

class AutonomousProjectKernel:
    """
    End-to-End 자율주행 제어 커널.

    isaac_physics_visualizer.py ↔ recovery_control_kernel.py 사이의
    핵심 연산 레이어.

    execute_control_cycle() 한 번 호출 = 1 제어 주기 완료.
    """

    def __init__(
        self,
        gpx_waypoints: np.ndarray,        # [N, 3] 워커힐 경로
        lane_graph     = None,             # LaneGraphConstrained
        so_path:  str  = "./libsafety_core.so",
        K_samples: int = 100,
        T_horizon: int = 15,
    ):
        self.ref_path   = gpx_waypoints     # 기준 경로
        self.lane_graph = lane_graph

        # 제어 모듈 초기화
        self.elastic_band = ElasticBandPlanner()
        self.mppi         = MPPIController(
            K_samples=K_samples, T_horizon=T_horizon
        )
        self.safety_c     = CSafetyCTypes(so_path)

        # 공개 속성 (Recovery Manager가 동적 조정)
        self.K_samples = K_samples
        self.T_horizon = T_horizon

        self._step = 0
        self._last_cmd = {
            "throttle": 0.0, "steering": 0.0, "brake": 0.0,
            "estop": False, "speed_kmh": 0.0,
            "active_rules": 0, "reason": [], "intervened": False,
        }

    # ── 메인 루프 ────────────────────────────────────────────────

    def execute_control_cycle(self, obs: dict) -> dict:
        """
        isaac_physics_visualizer가 매 프레임 호출.

        obs 키:
          ego_position    [3]   world XYZ
          ego_orientation [4]   quat [w,x,y,z]
          velocity_m_s    float
          lidar_scan_points [M,3]
          traffic_light   int   (0=green 1=red 2=yellow)
          oncoming_flag   int
          weather_mask    int   (EXC_* 비트마스크)
          shadow_kl       float
        """
        self._step += 1

        pos     = np.array(obs["ego_position"])
        quat    = np.array(obs["ego_orientation"])
        speed   = float(obs.get("velocity_m_s", 0.0))
        lidar   = np.array(obs.get("lidar_scan_points", []))
        sig     = int(obs.get("traffic_light", 0))
        oncoming= int(obs.get("oncoming_flag", 0))
        kl      = float(obs.get("shadow_kl", 0.0))

        # Fix #1: 정확한 yaw 추출
        yaw = self._quat_to_yaw(quat)

        # Fix #2: LiDAR world → ego frame 변환
        obs_ego = self._world_to_ego(lidar, pos, quat) if len(lidar) else np.zeros((0,3))

        # 1. Elastic Band: 참조 경로 부드럽게
        ref_smooth = self.elastic_band.smooth(
            self._local_ref_path(pos),
            lidar if len(lidar) else np.zeros((0,3)),
        )

        # MPPI K_samples / T_horizon 동적 반영 (Recovery Manager 조정)
        self.mppi.K = self.K_samples
        self.mppi.T = self.T_horizon
        if len(self.mppi._u_prev) != self.T_horizon:
            self.mppi._u_prev = np.zeros((self.T_horizon, 2))

        # 2. MPPI: 최적 throttle/steer 계산
        state_vec = np.array([pos[0], pos[1], pos[2], yaw, speed])
        throttle, steering = self.mppi.compute(
            state_vec, ref_smooth, obs_ego
        )

        # 3. C Safety Layer
        cmd = self.safety_c.apply(
            throttle, steering, speed, kl
        )

        # 신호등 규칙 추가 (Python 레벨)
        if sig == 1 and speed > 0.5:   # 빨간불
            cmd["throttle"] = min(cmd["throttle"], -0.4)
            cmd["brake"]    = max(cmd["brake"], 0.6)
            cmd["reason"].append("red_light")
            cmd["intervened"] = True

        if oncoming:
            cmd["steering"] = float(np.clip(cmd["steering"] + 0.4, -1, 1))
            cmd["reason"].append("oncoming_correction")
            cmd["intervened"] = True

        self._last_cmd = cmd
        return cmd

    # ── 헬퍼 ─────────────────────────────────────────────────────

    def _local_ref_path(self, pos: np.ndarray) -> np.ndarray:
        """현재 위치 기준 앞 50m 참조 경로 추출."""
        dists = np.linalg.norm(
            self.ref_path[:, :2] - pos[:2], axis=1
        )
        start = int(np.argmin(dists))
        end   = min(start + 30, len(self.ref_path))
        return self.ref_path[start:end]

    @staticmethod
    def _quat_to_yaw(q: np.ndarray) -> float:
        """Fix #1: 완전한 quaternion yaw 추출."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny = 2.0 * (w*z + x*y)
        cosy = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _world_to_ego(
        pts_world: np.ndarray,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> np.ndarray:
        """Fix #2: world → ego frame 변환."""
        if len(pts_world) == 0:
            return np.zeros((0, 3))
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
        delta = pts_world[:, :3] - pos[:3]
        return (R.T @ delta.T).T
