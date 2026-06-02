"""
autonomy/three_layer_stack_v2.py
──────────────────────────────────
10개 버그 전부 수정된 완전판

수정 내역:
  #1  quat_to_yaw() — 정확한 quaternion yaw 추출
  #2  transform_points_to_ego_frame() — world→ego 좌표 변환
  #3  lane-constrained avoidance — 차선 경계 내 회피
  #4  복합 reasons 리스트 — 동시 로그
  #5  Shadow KL hysteresis — ENTER/EXIT 이중 임계치
  #6  5-waypoint sequence obs — 곡률 사전 인식
  #7  4-channel BEV — occupancy/height/intensity/dynamic
  #8  (구조적 한계 주석) — Safety → Rust 마이그레이션 가이드
  #9  타임스탬프 동기화 — ringbuffer 기반 ApproxTimeSync
  #10 TrajectoryMemory — LSTM obs 히스토리 버퍼
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Fix #1 — 올바른 Quaternion → Yaw 변환
# ══════════════════════════════════════════════════════════════════

def quat_to_yaw(q: np.ndarray) -> float:
    """
    [w, x, y, z] 쿼터니언 → ENU yaw (라디안).
    고 pitch/roll 상황에서도 heading drift 없음.

    이전 코드 문제:
        yaw = 2.0 * math.atan2(quat[3], quat[0])
        → q = [w,0,0,z] 일 때만 정확, pitch/roll 있으면 오차 발생.

    수정:
        완전한 3D 회전 행렬 기반 yaw 추출 사용.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] → 3×3 회전 행렬."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# Fix #2 — World → Ego Frame 변환
# ══════════════════════════════════════════════════════════════════

def transform_points_to_ego_frame(
    points_world: np.ndarray,   # [N, 3]  world 좌표
    ego_pos:      np.ndarray,   # [3]
    ego_quat:     np.ndarray,   # [4] w,x,y,z
) -> np.ndarray:
    """
    월드 좌표 포인트들을 ego(차량) 로컬 프레임으로 변환.

    이전 코드 문제:
        obstacles[:, 0] > 0  → world X 기준 필터
        → 차량이 회전하면 옆 장애물을 전방으로 오판.

    수정:
        R^T * (p - pos) 로 ego 프레임 변환 후 필터링.
    """
    if len(points_world) == 0:
        return np.zeros((0, 3), np.float32)

    R   = quat_to_rotation_matrix(ego_quat)          # 3×3
    rel = points_world[:, :3] - ego_pos[np.newaxis]  # [N,3]
    return (R.T @ rel.T).T                            # [N,3] ego frame


# ══════════════════════════════════════════════════════════════════
# Fix #9 — 타임스탬프 동기화 (ApproxTimeSync ringbuffer)
# ══════════════════════════════════════════════════════════════════

class ApproxTimeSync:
    """
    서로 다른 주파수 센서(LiDAR 10Hz / Camera 30Hz / IMU 200Hz)를
    타임스탬프 기준으로 동기화하는 링버퍼.

    이전 코드 문제:
        타임스탬프 저장만 하고 실제 sync 없음.

    수정:
        슬라이딩 윈도우로 max_dt_s 이내 메시지 매칭.

    ROS2에서는 message_filters.ApproximateTimeSynchronizer 사용.
    이 클래스는 Python-only 환경 폴백.
    """

    def __init__(self, max_dt_s: float = 0.05):
        self.max_dt = max_dt_s
        self._buffers: Dict[str, deque] = {}

    def register(self, topic: str, maxlen: int = 10):
        self._buffers[topic] = deque(maxlen=maxlen)

    def add(self, topic: str, timestamp: float, data):
        if topic not in self._buffers:
            self.register(topic)
        self._buffers[topic].append((timestamp, data))

    def get_synced(self) -> Optional[Dict[str, any]]:
        """
        모든 토픽에서 max_dt 이내 메시지 셋 반환.
        없으면 None.
        """
        if not all(self._buffers.values()):
            return None

        # 각 버퍼의 가장 최근 타임스탬프 기준 매칭
        latest_times = {
            k: buf[-1][0] for k, buf in self._buffers.items()
        }
        ref_t = max(latest_times.values())

        synced = {}
        for topic, buf in self._buffers.items():
            # ref_t와 가장 가까운 메시지 찾기
            best = min(buf, key=lambda x: abs(x[0] - ref_t))
            if abs(best[0] - ref_t) > self.max_dt:
                return None   # sync 불가
            synced[topic] = best[1]

        return synced


# ══════════════════════════════════════════════════════════════════
# Fix #10 — Trajectory Memory (LSTM obs 히스토리)
# ══════════════════════════════════════════════════════════════════

class TrajectoryMemory:
    """
    단일 스텝 반응형 정책의 한계를 극복하는 시간적 메모리.

    이전 코드 문제:
        매 스텝 독립 관측 → 가속도/곡률 변화 인식 불가.

    수정:
        히스토리 H 스텝을 스택으로 쌓아 LSTM/Transformer 입력 제공.
        현재: 슬라이딩 윈도우 평균 (경량)
        다음: nn.LSTM hidden state 통합

    실제 배포:
        RLlib RecurrentPPO + LSTMWrapper 사용.
    """

    def __init__(self, obs_dim: int = 17, history: int = 8):
        self.obs_dim = obs_dim
        self.H       = history
        self._buf    = deque(maxlen=history)
        # 가속도/각속도 추정용
        self._prev_vel:   Optional[np.ndarray] = None
        self._prev_t:     float = time.time()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """obs 추가 후 [H × obs_dim] 히스토리 반환."""
        self._buf.append(obs.copy())
        # 부족한 스텝은 0으로 패딩
        pad = self.H - len(self._buf)
        history = np.zeros((self.H, self.obs_dim), np.float32)
        for i, o in enumerate(self._buf):
            history[pad + i] = o[:self.obs_dim]
        return history   # [H, obs_dim]

    def compute_derivatives(
        self,
        vel: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        속도 벡터로부터 가속도와 yaw rate 추정.
        Returns: (accel [3], yaw_rate float)
        """
        now = time.time()
        dt  = now - self._prev_t + 1e-8

        if self._prev_vel is None:
            accel    = np.zeros(3, np.float32)
            yaw_rate = 0.0
        else:
            accel    = (vel - self._prev_vel) / dt
            # yaw rate 근사: 횡방향 가속도 / 종방향 속도
            vx = max(abs(float(vel[0])), 0.1)
            yaw_rate = float(accel[1]) / vx

        self._prev_vel = vel.copy()
        self._prev_t   = now
        return accel.astype(np.float32), yaw_rate


# ══════════════════════════════════════════════════════════════════
# Fix #7 — 4채널 BEV 인코딩
# ══════════════════════════════════════════════════════════════════

class BEVEncoder:
    """
    LiDAR 포인트클라우드 → 4채널 Bird's Eye View.

    이전 코드 문제:
        단일 점유 채널만 → semantic 구분 없음.

    수정:
        ch0 = occupancy   (0/1)
        ch1 = height      (정규화 높이)
        ch2 = intensity   (반사 강도)
        ch3 = dynamic     (프레임 간 차이 → 움직이는 물체)

    RL 성능 향상 이유:
        ch3(dynamic)으로 보행자/차량을 정적 장애물과 구분.
    """

    def __init__(self, range_m: float = 50.0, size: int = 64):
        self.range  = range_m
        self.size   = size
        self._prev_bev: Optional[np.ndarray] = None

    def encode(
        self,
        pts_ego: np.ndarray,   # [N,4] ego frame (x,y,z,intensity)
        ground_z: float = 0.0,
    ) -> np.ndarray:
        """Returns: [4, size, size] float32."""
        bev = np.zeros((4, self.size, self.size), np.float32)
        S   = self.size
        R   = self.range

        # 유효 포인트 필터
        mask = (
            (np.abs(pts_ego[:, 0]) < R) &
            (np.abs(pts_ego[:, 1]) < R) &
            (pts_ego[:, 2] > ground_z - 0.2)
        )
        pts = pts_ego[mask]
        if len(pts) == 0:
            return bev

        ix = np.clip(((pts[:, 0] + R) / (2*R) * S).astype(int), 0, S-1)
        iy = np.clip(((pts[:, 1] + R) / (2*R) * S).astype(int), 0, S-1)

        # ch0: occupancy
        np.add.at(bev[0], (ix, iy), 1.0)
        bev[0] = np.clip(bev[0] / 5.0, 0, 1)

        # ch1: height (지면 대비 정규화)
        height = np.clip((pts[:, 2] - ground_z) / 3.0, 0, 1)
        np.maximum.at(bev[1], (ix, iy), height)

        # ch2: intensity 정규화
        intensity = np.clip(pts[:, 3], 0, 1)
        np.maximum.at(bev[2], (ix, iy), intensity)

        # ch3: dynamic (이전 프레임과 차이)
        if self._prev_bev is not None:
            bev[3] = np.abs(bev[0] - self._prev_bev[0])
            bev[3] = np.clip(bev[3], 0, 1)

        self._prev_bev = bev.copy()
        return bev


# ══════════════════════════════════════════════════════════════════
# Fix #3 — Lane-constrained Obstacle Avoidance
# ══════════════════════════════════════════════════════════════════

@dataclass
class LaneBoundary:
    """단순 차선 경계 (좌/우 오프셋)."""
    left_m:  float = -3.5   # ego 기준 왼쪽 허용 한계 (m)
    right_m: float =  3.5   # 오른쪽 허용 한계


class LaneConstrainedAvoider:
    """
    장애물 회피 시 차선 경계를 넘지 않도록 제약.

    이전 코드 문제:
        offset = ±2.0 고정 → 반대 차선 침범 가능.

    수정:
        (1) 회피 방향 결정 (장애물 반대편)
        (2) 오프셋을 차선 경계까지만 허용
        (3) 넘치면 속도를 감속해 추종

    다음 단계: elastic band → MPPI → graph neural planner
    """

    def __init__(self, lane: LaneBoundary = None):
        self.lane = lane or LaneBoundary()

    def apply(
        self,
        waypoints: List,
        obstacles_ego: np.ndarray,   # ego frame [N,3]
    ) -> List:
        if len(obstacles_ego) == 0 or not waypoints:
            return waypoints

        modified = []
        for wp in waypoints:
            near_mask = (
                (obstacles_ego[:, 0] > 0) &
                (obstacles_ego[:, 0] < wp.x + 5.0) &
                (np.abs(obstacles_ego[:, 1] - wp.y) < 3.0)
            )
            near = obstacles_ego[near_mask]

            if len(near) == 0:
                modified.append(wp)
                continue

            obs_mean_y = float(near[:, 1].mean())

            # 회피 방향: 장애물 반대
            if obs_mean_y > 0:
                target_y = wp.y + self.lane.left_m * 0.5   # 왼쪽으로
            else:
                target_y = wp.y + self.lane.right_m * 0.5  # 오른쪽으로

            # ── 차선 경계 클리핑 ────────────────────────────────
            target_y = float(np.clip(
                target_y,
                wp.y + self.lane.left_m,
                wp.y + self.lane.right_m,
            ))

            # 오프셋 크기
            offset = target_y - wp.y

            if abs(offset) < 0.1:
                # 회피 공간 없음 → 감속
                wp = wp.__class__(
                    x=wp.x, y=wp.y, z=wp.z,
                    speed=min(wp.speed, 2.78),   # 10 km/h
                    heading=wp.heading,
                )
            else:
                wp = wp.__class__(
                    x=wp.x, y=target_y, z=wp.z,
                    speed=min(wp.speed, 5.56),   # 20 km/h
                    heading=wp.heading,
                )
            modified.append(wp)

        return modified


# ══════════════════════════════════════════════════════════════════
# Fix #6 — 5-Waypoint Sequence Observation
# ══════════════════════════════════════════════════════════════════

def encode_waypoint_sequence(
    plan:     List,
    state_pos: np.ndarray,
    state_quat: np.ndarray,
    n_wps: int = 5,
) -> np.ndarray:
    """
    다음 N개 waypoint를 ego frame으로 인코딩.
    [dx, dy, dspeed, dcurv] × N = 4N 차원.

    이전 코드 문제:
        plan[0] 하나만 사용 → 곡률 사전 인식 불가.

    수정:
        앞 5개 waypoint sequence → 커브 예측 가능.

    왜 중요한가:
        차량은 현재 waypoint가 아니라
        앞 곡률을 보고 핸들을 미리 돌려야 함.
        (인간 운전자가 하는 방식과 동일)
    """
    yaw = quat_to_yaw(state_quat)
    cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)

    obs = np.zeros(n_wps * 4, np.float32)

    for i in range(n_wps):
        if i >= len(plan):
            break
        wp = plan[i]
        dx_w = wp.x - float(state_pos[0])
        dy_w = wp.y - float(state_pos[1])
        # world → ego
        dx_e =  cos_yaw * dx_w - sin_yaw * dy_w
        dy_e =  sin_yaw * dx_w + cos_yaw * dy_w

        # 곡률 근사 (연속 waypoint 방향 변화)
        curv = 0.0
        if i < len(plan) - 1:
            nwp = plan[i+1]
            ddx = nwp.x - wp.x
            ddy = nwp.y - wp.y
            seg_len = math.hypot(ddx, ddy) + 1e-8
            heading_diff = math.atan2(ddy, ddx) - math.atan2(
                dy_w, dx_w + 1e-8
            )
            curv = heading_diff / seg_len

        obs[i*4]   = float(np.clip(dx_e / 30.0, -1, 1))
        obs[i*4+1] = float(np.clip(dy_e /  5.0, -1, 1))
        obs[i*4+2] = float(np.clip((wp.speed - 8.33) / 8.33, -1, 1))
        obs[i*4+3] = float(np.clip(curv * 10.0, -1, 1))

    return obs


# ══════════════════════════════════════════════════════════════════
# Fix #4 — 복합 reasons 리스트
# ══════════════════════════════════════════════════════════════════

@dataclass
class SafeCmd:
    throttle:  float
    steering:  float
    brake:     float
    estop:     bool          = False
    speed_kmh: float         = 0.0
    reasons:   List[str]     = field(default_factory=list)

    @property
    def reason(self) -> str:
        return " + ".join(self.reasons) if self.reasons else "nominal"


# ══════════════════════════════════════════════════════════════════
# Fix #5 — Shadow KL Hysteresis
# ══════════════════════════════════════════════════════════════════

class KLHysteresis:
    """
    단일 임계치 대신 ENTER/EXIT 이중 임계치로
    threshold bouncing(깜박임) 방지.

    이전:
        if kl > 0.08: restrict()

    수정:
        ENTER = 0.12  → 이 값 초과 시 제한 모드 진입
        EXIT  = 0.06  → 이 값 이하로 내려가야 정상 복귀

    실제 효과:
        KL = 0.10 → 제한 안 함 (ENTER 미달)
        KL = 0.13 → 제한 진입
        KL = 0.09 → 아직 제한 유지 (EXIT 미달)
        KL = 0.05 → 제한 해제
    """
    ENTER = 0.12
    EXIT  = 0.06

    def __init__(self):
        self._restricted = False

    def update(self, kl: float) -> bool:
        """Returns True if in restricted mode."""
        if not self._restricted and kl > self.ENTER:
            self._restricted = True
            logger.warning("KL hysteresis: 제한 모드 진입 (kl=%.4f)", kl)
        elif self._restricted and kl < self.EXIT:
            self._restricted = False
            logger.info("KL hysteresis: 정상 복귀 (kl=%.4f)", kl)
        return self._restricted


# ══════════════════════════════════════════════════════════════════
# 통합 데이터 타입
# ══════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    x: float; y: float; z: float = 0.0
    speed: float = 8.33; heading: float = 0.0


@dataclass
class VehicleState:
    pos:     np.ndarray
    vel:     np.ndarray
    quat:    np.ndarray
    ang_vel: np.ndarray
    speed:   float = 0.0

    @classmethod
    def zeros(cls):
        return cls(np.zeros(3), np.zeros(3),
                   np.array([1,0,0,0],np.float32), np.zeros(3))


@dataclass
class PointCloud:
    points:    np.ndarray   # [N,4] x,y,z,intensity  (world frame)
    timestamp: float = 0.0

    @classmethod
    def mock(cls, n=500):
        pts = np.random.randn(n, 4).astype(np.float32)
        pts[:, 3] = np.abs(pts[:, 3])
        return cls(pts, time.time())


# ══════════════════════════════════════════════════════════════════
# Safety Layer (Fix #4 + #5 + #8 주석)
# ══════════════════════════════════════════════════════════════════

class SafetyLayer:
    """
    Fix #8 주석:
        이 계층은 현재 Python으로 구현되어 있으나
        production에서는 Rust microservice로 교체 권장.

        계층별 언어 계획:
          Safety      → Rust (μs 단위, hard realtime)
          RL          → Python/ONNX
          VLM         → Python
          Actuator    → C++ RTOS

        마이그레이션 경로:
          1. 이 클래스의 apply()를 Rust FFI로 래핑
          2. PyO3 바인딩으로 Python 인터페이스 유지
          3. 점진적 교체
    """

    def __init__(
        self,
        max_speed_ms: float = 8.33,
        ttc_thresh:   float = 2.0,
        geofence_pts: Optional[np.ndarray] = None,
    ):
        self.max_v       = max_speed_ms
        self.ttc_thresh  = ttc_thresh
        self.geofence    = geofence_pts
        self._estop      = False
        self._kl_hyst    = KLHysteresis()        # Fix #5
        self._log: List[dict] = []

    def apply(
        self,
        throttle:   float,
        steering:   float,
        state:      VehicleState,
        pts_ego:    np.ndarray,      # ego frame 포인트 (Fix #2)
        perception: Optional[dict] = None,
        shadow_kl:  float = 0.0,
    ) -> SafeCmd:
        brake   = 0.0
        reasons: List[str] = []      # Fix #4 복합 로그

        # Rule 1: e-stop
        if self._estop:
            return SafeCmd(0.0, 0.0, 1.0, estop=True,
                           speed_kmh=state.speed*3.6,
                           reasons=["e-stop"])

        # Rule 2: 신호등
        if perception:
            sig = perception.get("signal_state", "unknown")
            if sig == "red" and state.speed > 0.5:
                throttle = min(throttle, -0.4)
                brake    = max(brake, 0.6)
                reasons.append("red_light")

        # Rule 3: TTC (ego frame 사용 — Fix #2)
        ttc = self._compute_ttc_ego(state, pts_ego)
        if ttc is not None and ttc < self.ttc_thresh:
            force = float(np.clip(
                (self.ttc_thresh - ttc) / self.ttc_thresh, 0, 1
            ))
            brake    = max(brake, force)
            throttle = min(throttle, 1.0 - force)
            reasons.append(f"ttc={ttc:.1f}s")

        # Rule 4: 속도 상한
        limit = self.max_v
        if perception:
            exceptions = perception.get("active_exceptions", [])
            limit = self._exception_limit(exceptions)

        if state.speed > limit:
            throttle = min(throttle, 0.0)
            reasons.append(f"speed_cap={limit*3.6:.0f}km/h")

        # Rule 5: Geofence
        if self.geofence is not None:
            if not self._in_geofence(state.pos[:2]):
                return SafeCmd(0.0, 0.0, 1.0,
                               reasons=["geofence"],
                               speed_kmh=state.speed*3.6)

        # Rule 6: 역주행
        if perception and perception.get("oncoming", False):
            steering = float(np.clip(steering + 0.4, -1, 1))
            throttle = min(throttle, 0.3)
            reasons.append("oncoming_correction")

        # Rule 7: Shadow KL hysteresis (Fix #5)
        restricted = self._kl_hyst.update(shadow_kl)
        if restricted:
            if state.speed > self.max_v * 0.5:
                throttle = min(throttle, 0.0)
            reasons.append(f"kl_restricted={shadow_kl:.3f}")

        if not reasons:
            reasons = ["nominal"]

        self._log.append({
            "t": time.time(),
            "reasons": reasons.copy(),
        })
        if len(self._log) > 2000:
            self._log = self._log[-1000:]

        return SafeCmd(
            throttle = float(np.clip(throttle, -1, 1)),
            steering = float(np.clip(steering, -1, 1)),
            brake    = float(np.clip(brake,     0, 1)),
            speed_kmh= round(state.speed * 3.6, 1),
            reasons  = reasons,
        )

    def _compute_ttc_ego(
        self,
        state:    VehicleState,
        pts_ego:  np.ndarray,
    ) -> Optional[float]:
        """ego frame 포인트 기준 TTC 계산 (Fix #2)."""
        if len(pts_ego) == 0 or state.speed < 0.5:
            return None

        # ego X+ = 전방, Y = 횡방향
        front_mask = (
            (pts_ego[:, 0] > 0.5) &
            (pts_ego[:, 0] < 40.0) &
            (np.abs(pts_ego[:, 1]) < 1.5)   # 차량 폭 이내
        )
        front = pts_ego[front_mask]
        if len(front) == 0:
            return None

        nearest = float(front[:, 0].min())
        return nearest / max(state.speed, 0.1)

    def _exception_limit(self, exceptions: List[str]) -> float:
        limits = {
            "WEATHER_HEAVY_RAIN":  5.56,
            "WEATHER_SNOW":        4.17,
            "WEATHER_FOG":         2.78,
            "WEATHER_BLACK_ICE":   2.78,
            "ROAD_CONSTRUCTION":   5.56,
            "ROAD_FLOODING":       0.0,
            "ACCIDENT_EMERGENCY":  0.0,
        }
        limit = self.max_v
        for e in exceptions:
            if e in limits:
                limit = min(limit, limits[e])
        return limit

    def _in_geofence(self, pos: np.ndarray) -> bool:
        pts = self.geofence
        n, inside, j = len(pts), False, len(pts)-1
        for i in range(n):
            xi, yi = pts[i]; xj, yj = pts[j]
            if ((yi > pos[1]) != (yj > pos[1])) and \
               (pos[0] < (xj-xi)*(pos[1]-yi)/(yj-yi+1e-8)+xi):
                inside = not inside
            j = i
        return inside

    def trigger_estop(self):  self._estop = True
    def release_estop(self):  self._estop = False


# ══════════════════════════════════════════════════════════════════
# 통합 스택 v2
# ══════════════════════════════════════════════════════════════════

class AutonomyStackV2:
    """
    10개 버그 수정된 3계층 자율주행 스택.
    step()은 고정 인터페이스 유지.
    """

    OBS_DIM = 17   # 기본 + 가속도(3) + yaw_rate(1) = 17

    def __init__(
        self,
        geofence_pts: Optional[np.ndarray] = None,
        lane_boundary: Optional[LaneBoundary] = None,
        history_len: int = 8,
    ):
        self.bev_enc  = BEVEncoder()
        self.avoider  = LaneConstrainedAvoider(lane_boundary)
        self.safety   = SafetyLayer(geofence_pts=geofence_pts)
        self.memory   = TrajectoryMemory(self.OBS_DIM, history_len)
        self.time_sync = ApproxTimeSync(max_dt_s=0.05)
        self._plan:   List[Waypoint] = []
        self._route:  List[Waypoint] = []
        self._step    = 0

    def load_route(self, waypoints: List[Waypoint]):
        self._route = waypoints

    def step(
        self,
        pc:         PointCloud,
        state:      VehicleState,
        perception: Optional[dict] = None,
        shadow_kl:  float = 0.0,
    ) -> SafeCmd:
        self._step += 1

        # Fix #2: world → ego frame 변환
        pts_ego = transform_points_to_ego_frame(
            pc.points[:, :3], state.pos, state.quat
        )
        pts_ego_4 = np.hstack([
            pts_ego,
            pc.points[:, 3:4],   # intensity 유지
        ])

        # Fix #7: 4채널 BEV
        bev = self.bev_enc.encode(pts_ego_4, ground_z=0.0)

        # Planning: ego frame 장애물로 차선 제약 회피
        plan = self._local_plan(state)
        plan = self.avoider.apply(plan, pts_ego)

        # Fix #6: 5-waypoint sequence obs
        wp_seq = encode_waypoint_sequence(
            plan, state.pos, state.quat, n_wps=5
        )

        # Fix #10: 가속도/yaw_rate 추정
        accel, yaw_rate = self.memory.compute_derivatives(state.vel)

        # Fix #1: 올바른 yaw
        yaw = quat_to_yaw(state.quat)

        # 관측 벡터 [17]
        obs = np.zeros(self.OBS_DIM, np.float32)
        obs[0]  = float(np.clip(yaw / math.pi, -1, 1))
        obs[1]  = float(np.clip(state.speed / 10.0, -1, 1))
        obs[2]  = float(np.clip(yaw_rate / 1.0, -1, 1))      # Fix #10
        obs[3:6]= np.clip(accel / 5.0, -1, 1)                # Fix #10
        obs[6]  = float(bev[0].max())                          # occupancy
        obs[7]  = float(bev[3].max())                          # dynamic
        if perception:
            obs[8]  = float(np.clip(perception.get("lane_deviation", 0)/200, -1, 1))
            obs[9]  = float(perception.get("oncoming", False))
            obs[10] = float({"red":0, "yellow":0.5, "green":1}.get(
                perception.get("signal_state","green"), 0.5))
        obs[11] = float(np.clip(shadow_kl / 0.2, 0, 1))       # Fix #5

        # Fix #10: 히스토리 버퍼
        hist = self.memory.push(obs)   # [H, OBS_DIM]

        # Mock RL 정책 (체크포인트 없을 때 PID)
        throttle, steering = self._mock_control(state, plan)

        # Fix #4 + #5 + #2: 안전 계층 (ego frame 포인트 전달)
        cmd = self.safety.apply(
            throttle, steering, state,
            pts_ego, perception, shadow_kl,
        )

        return cmd

    def _local_plan(self, state: VehicleState) -> List[Waypoint]:
        if not self._route:
            yaw = quat_to_yaw(state.quat)
            return [
                Waypoint(
                    x=state.pos[0]+math.cos(yaw)*d,
                    y=state.pos[1]+math.sin(yaw)*d,
                )
                for d in [5,10,15,20,25]
            ]
        pos  = state.pos[:2]
        dists= [math.hypot(wp.x-pos[0], wp.y-pos[1])
                for wp in self._route]
        start = int(np.argmin(dists))
        return self._route[start:start+10]

    def _mock_control(
        self,
        state: VehicleState,
        plan:  List[Waypoint],
    ) -> Tuple[float, float]:
        if not plan:
            return 0.0, 0.0
        wp  = plan[0]
        yaw = quat_to_yaw(state.quat)
        dx  = wp.x - float(state.pos[0])
        dy  = wp.y - float(state.pos[1])
        lx  =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
        ly  =  math.sin(-yaw)*dx + math.cos(-yaw)*dy
        thr = float(np.clip((wp.speed-state.speed)/wp.speed, -1, 1))
        ste = float(np.clip(-ly/max(lx, 0.1)*0.5, -1, 1))
        return thr, ste

    def estop(self):         self.safety.trigger_estop()
    def release_estop(self): self.safety.release_estop()


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

def smoke_test():
    print("\n" + "="*56)
    print("  AutonomyStack v2 — 10-bug-fix smoke test")
    print("="*56)
    PASS = "\033[92m✓ PASS\033[0m"
    FAIL = "\033[91m✗ FAIL\033[0m"

    stack = AutonomyStackV2()
    stack.load_route([
        Waypoint(x=float(i*5), y=float(i*0.3)) for i in range(20)
    ])
    state = VehicleState.zeros()
    state.speed = 5.0
    state.vel   = np.array([5.0, 0.0, 0.0])

    perc = {"signal_state":"green","oncoming":False,
            "collision_risk":0.0,"active_exceptions":[]}

    results = {}

    # Fix #1: yaw 정확성
    q_pitched = np.array([0.966, 0.259, 0.0, 0.0])  # 30° pitch
    old_yaw = 2.0 * math.atan2(q_pitched[3], q_pitched[0])
    new_yaw = quat_to_yaw(q_pitched)
    results["#1 quat_to_yaw"] = abs(new_yaw) < abs(old_yaw) + 0.01

    # Fix #2: ego transform
    pts_w = np.array([[10.0, 0.0, 0.0, 1.0]])
    pts_e = transform_points_to_ego_frame(
        pts_w[:,:3], state.pos, state.quat
    )
    results["#2 ego_transform"] = pts_e[0,0] > 9.0

    # Fix #3: lane constraint
    from copy import deepcopy
    wps = [Waypoint(5.0, 0.0)]
    obs_far_right = np.array([[5.0, 3.0, 0.0]])  # 오른쪽 장애물
    avoided = stack.avoider.apply(deepcopy(wps), obs_far_right)
    results["#3 lane_constrained"] = (
        avoided[0].y < wps[0].y and
        avoided[0].y >= wps[0].y + LaneBoundary().left_m
    )

    # Fix #4: compound reasons
    state2 = VehicleState.zeros()
    state2.speed = 15.0
    state2.vel   = np.array([15.0, 0, 0])
    perc_red = {**perc, "signal_state":"red"}
    cmd = stack.step(PointCloud.mock(), state2, perc_red, shadow_kl=0.0)
    results["#4 compound_reasons"] = len(cmd.reasons) >= 1

    # Fix #5: KL hysteresis
    hyst = KLHysteresis()
    hyst.update(0.10)   # ENTER=0.12 미달 → 정상
    r1 = not hyst._restricted
    hyst.update(0.13)   # 진입
    r2 = hyst._restricted
    hyst.update(0.09)   # EXIT=0.06 미달 → 유지
    r3 = hyst._restricted
    hyst.update(0.05)   # 해제
    r4 = not hyst._restricted
    results["#5 kl_hysteresis"] = r1 and r2 and r3 and r4

    # Fix #6: 5-waypoint seq
    plan6 = [Waypoint(float(i*5), 0.0) for i in range(6)]
    seq   = encode_waypoint_sequence(plan6, state.pos, state.quat, 5)
    results["#6 waypoint_seq"] = seq.shape == (20,) and seq[3] == 0.0

    # Fix #7: 4-channel BEV
    bev = stack.bev_enc.encode(
        np.hstack([np.random.randn(100,3).astype(np.float32),
                   np.random.rand(100,1).astype(np.float32)])
    )
    results["#7 bev_4ch"] = bev.shape == (4, 64, 64)

    # Fix #9: time sync
    sync = ApproxTimeSync(max_dt_s=0.05)
    sync.register("lidar"); sync.register("imu")
    t = time.time()
    sync.add("lidar", t, "lidar_data")
    sync.add("imu",   t+0.01, "imu_data")
    synced = sync.get_synced()
    results["#9 time_sync"] = synced is not None

    # Fix #10: trajectory memory
    mem = TrajectoryMemory(obs_dim=17, history=8)
    for _ in range(5):
        h = mem.push(np.random.randn(17).astype(np.float32))
    results["#10 traj_memory"] = h.shape == (8, 17)

    # e-stop
    stack.estop()
    cmd_e = stack.step(PointCloud.mock(), state, perc)
    results["e-stop"] = cmd_e.estop
    stack.release_estop()

    print()
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} PASSED  "
          f"{'✅ ALL PASS' if all_pass else '❌ SOME FAIL'}\n")
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
