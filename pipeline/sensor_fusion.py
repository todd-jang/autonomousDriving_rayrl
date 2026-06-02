"""
pipeline/sensor_fusion.py
──────────────────────────
Sensor Fusion (EKF) — 다이어그램 최상단 레이어

LiDAR 포인트클라우드 + Garmin GPS + Camera + IMU
  → world state [N×13]

Fix #9 통합: NsTimestamp 기반 ApproxTimeSync
robot_localization 패키지 없이 Python-native EKF.

상태 벡터 [13]:
  [0:3]   pos   (x, y, z)   world frame, meter
  [3:6]   vel   (vx,vy,vz)  world frame, m/s
  [6:10]  quat  (w,x,y,z)
  [10:13] accel (ax,ay,az)  body frame, m/s²
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# 타임스탬프 동기화 (Fix #9)
# ══════════════════════════════════════════════════════════════════

class SensorRingBuffer:
    """고정 크기 링 버퍼 — 타임스탬프 기반 nearest 검색."""

    def __init__(self, maxlen: int = 16):
        self._buf: list = []
        self._maxlen = maxlen

    def add(self, stamp: float, data):
        self._buf.append((stamp, data))
        if len(self._buf) > self._maxlen:
            self._buf.pop(0)

    def nearest(self, ref_t: float, max_dt: float):
        if not self._buf:
            return None
        best = min(self._buf, key=lambda x: abs(x[0] - ref_t))
        return best[1] if abs(best[0] - ref_t) <= max_dt else None

    def has_data(self) -> bool:
        return len(self._buf) > 0


# ══════════════════════════════════════════════════════════════════
# EKF (Extended Kalman Filter)
# ══════════════════════════════════════════════════════════════════

class EKFSensorFusion:
    """
    13차원 상태 벡터 EKF.

    센서 주기:
      LiDAR  10 Hz  → 위치 보정
      GPS    1 Hz   → 위치 + 고도
      IMU    200 Hz → 자세 + 가속도 (주요 예측)
      Camera 30 Hz  → 자세 보정 (visual odometry 근사)
    """

    DIM = 13   # 상태 벡터 차원

    # 프로세스 노이즈
    Q_POS   = 0.01    # 위치
    Q_VEL   = 0.1     # 속도
    Q_QUAT  = 0.001   # 자세
    Q_ACCEL = 0.5     # 가속도

    # 관측 노이즈
    R_GPS    = 2.0    # GPS 위치 노이즈 (m)
    R_LIDAR  = 0.3    # LiDAR 위치 노이즈 (m)
    R_IMU    = 0.05   # IMU 자세 노이즈

    def __init__(self):
        self._x = np.zeros(self.DIM)          # 상태 벡터
        self._x[6] = 1.0                       # quat w = 1 (단위 쿼터니언)
        self._P = np.eye(self.DIM) * 10.0     # 공분산 행렬
        self._initialized = False
        self._last_t = time.time()

        # 센서 버퍼
        self._gps_buf    = SensorRingBuffer(8)
        self._lidar_buf  = SensorRingBuffer(4)
        self._imu_buf    = SensorRingBuffer(32)
        self._camera_buf = SensorRingBuffer(8)

    # ── 센서 데이터 수신 ──────────────────────────────────────────

    def add_gps(self, stamp: float, lat: float, lon: float,
                alt: float, origin_utm: Tuple[float, float]):
        """Garmin GPS → UTM → EKF 관측."""
        from pipeline.garmin_bridge import ll_to_utm_local
        x, y = ll_to_utm_local(lat, lon, origin_utm)
        self._gps_buf.add(stamp, np.array([x, y, alt]))

    def add_lidar_pos(self, stamp: float, pos: np.ndarray):
        """LiDAR odometry 위치 추정치."""
        self._lidar_buf.add(stamp, pos)

    def add_imu(self, stamp: float, accel: np.ndarray,
                gyro: np.ndarray, quat: np.ndarray):
        """IMU 가속도 + 자이로 + 자세."""
        self._imu_buf.add(stamp, {
            "accel": accel, "gyro": gyro, "quat": quat
        })

    def add_camera_quat(self, stamp: float, quat: np.ndarray):
        """Visual Odometry 자세 추정."""
        self._camera_buf.add(stamp, quat)

    # ── 메인 업데이트 ─────────────────────────────────────────────

    def update(self) -> np.ndarray:
        """
        현재 시각 기준 EKF predict + update.
        Returns: world state [13]
        """
        now = time.time()
        dt  = min(now - self._last_t, 0.1)   # 최대 100ms 클리핑
        self._last_t = now

        # 1. Predict (IMU 기반)
        imu = self._imu_buf.nearest(now, max_dt=0.01)
        if imu:
            self._predict_imu(dt, imu["accel"], imu["gyro"])
        else:
            self._predict_const_vel(dt)

        # 2. Update — GPS (1 Hz, 2m 정확도)
        gps = self._gps_buf.nearest(now, max_dt=0.5)
        if gps is not None:
            self._update_position(gps, noise=self.R_GPS)

        # 3. Update — LiDAR (10 Hz, 0.3m 정확도)
        lidar_pos = self._lidar_buf.nearest(now, max_dt=0.05)
        if lidar_pos is not None:
            self._update_position(lidar_pos, noise=self.R_LIDAR)

        # 4. Update — Camera 자세 (30 Hz)
        cam_q = self._camera_buf.nearest(now, max_dt=0.02)
        if cam_q is not None:
            self._update_quat(cam_q, noise=self.R_IMU)
        elif imu:
            self._update_quat(imu["quat"], noise=self.R_IMU * 0.5)

        # 쿼터니언 정규화
        q = self._x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            self._x[6:10] = q / q_norm

        self._initialized = True
        return self._x.copy()

    # ── EKF 수학 ─────────────────────────────────────────────────

    def _predict_imu(
        self,
        dt: float,
        accel: np.ndarray,  # body frame
        gyro:  np.ndarray,  # body frame, rad/s
    ):
        """IMU 기반 상태 전이."""
        # 자세 회전 (body → world)
        q = self._x[6:10]
        R = self._quat_to_rotmat(q)

        # 가속도 world frame 변환 (중력 제거)
        g_world = np.array([0, 0, -9.81])
        acc_world = R @ accel + g_world

        # 상태 전이
        self._x[0:3] += self._x[3:6] * dt + 0.5 * acc_world * dt**2
        self._x[3:6] += acc_world * dt
        self._x[10:13] = accel   # body frame 가속도 저장

        # 쿼터니언 적분 (1차 근사)
        omega = gyro
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-6:
            angle = omega_norm * dt
            axis  = omega / omega_norm
            dq    = np.array([
                math.cos(angle/2),
                axis[0]*math.sin(angle/2),
                axis[1]*math.sin(angle/2),
                axis[2]*math.sin(angle/2),
            ])
            self._x[6:10] = self._quat_mult(q, dq)

        # 공분산 전파 (대각 근사로 WCET 보장)
        Q = np.diag([
            self.Q_POS]*3 + [self.Q_VEL]*3 +
            [self.Q_QUAT]*4 + [self.Q_ACCEL]*3
        )
        self._P += Q * dt

    def _predict_const_vel(self, dt: float):
        """IMU 없을 때: 등속 운동 모델."""
        self._x[0:3] += self._x[3:6] * dt
        self._P += np.eye(self.DIM) * 0.1 * dt

    def _update_position(self, z: np.ndarray, noise: float):
        """위치 관측 업데이트 (x, y, z)."""
        H = np.zeros((3, self.DIM))
        H[0,0] = H[1,1] = H[2,2] = 1.0
        R = np.eye(3) * noise**2

        innovation = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x += K @ innovation
        self._P  = (np.eye(self.DIM) - K @ H) @ self._P

    def _update_quat(self, q_obs: np.ndarray, noise: float):
        """쿼터니언 관측 업데이트."""
        H = np.zeros((4, self.DIM))
        H[0,6] = H[1,7] = H[2,8] = H[3,9] = 1.0
        R = np.eye(4) * noise**2

        innovation = q_obs - H @ self._x
        # 쿼터니언 부호 정렬 (antipodal 모호성)
        if np.dot(q_obs, self._x[6:10]) < 0:
            innovation = -q_obs - H @ self._x

        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x += K @ innovation
        self._P  = (np.eye(self.DIM) - K @ H) @ self._P

    @staticmethod
    def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
        w,x,y,z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def get_vehicle_state_dict(self) -> dict:
        """AutonomousProjectKernel obs 형식으로 변환."""
        x = self._x
        speed = float(np.linalg.norm(x[3:6]))
        return {
            "ego_position":    x[0:3].copy(),
            "ego_orientation": x[6:10].copy(),
            "velocity_m_s":    speed,
            "velocity_vec":    x[3:6].copy(),
            "acceleration":    x[10:13].copy(),
        }
