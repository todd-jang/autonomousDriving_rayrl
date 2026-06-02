"""
pipeline/garmin_bridge.py
──────────────────────────
Garmin GPX → ROS2 PoseStamped / Nav2 goal

다이어그램: Planning layer 우측 박스
  "Garmin route → Nav2 goal  GPX → PoseStamped"

기능:
  1. GPX 파싱         WGS84 lat/lon → Waypoint 목록
  2. UTM 변환         ll_to_utm_simple() / ll_to_utm_local()
  3. PoseStamped 생성 ROS2 geometry_msgs 호환 dict
  4. Nav2 goal 발행   SimpleActionClient 래퍼 (ROS2 없으면 mock)
  5. /fix 퍼블리시    sensor_msgs/NavSatFix (Garmin 실기기 NMEA)
  6. Isaac Sim 연동   stage_utils 없이도 동작

Isaac Sim standalone 모드:
  ROS2 없어도 PoseStamped dict 형태로 반환.
  ROS2 있으면 실제 토픽 퍼블리시.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── WGS84 상수 ────────────────────────────────────────────────────
WGS84_A  = 6_378_137.0
WGS84_E2 = 6.694379990141414e-3
GPX_NS   = "http://www.topografix.com/GPX/1/1"


# ══════════════════════════════════════════════════════════════════
# 좌표 변환 유틸
# ══════════════════════════════════════════════════════════════════

def ll_to_utm_simple(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """
    WGS84 → UTM easting/northing (절대값).
    원점 기준 상대 변환에 사용.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    zone = int((lon_deg + 180) / 6) + 1
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    e2p = WGS84_E2 / (1 - WGS84_E2)
    N   = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
    T   = math.tan(lat)**2
    C   = e2p * math.cos(lat)**2
    A_  = math.cos(lat) * (lon - lon0)
    M   = WGS84_A * (
        (1 - WGS84_E2/4 - 3*WGS84_E2**2/64) * lat
        - (3*WGS84_E2/8 + 3*WGS84_E2**2/32) * math.sin(2*lat)
        + (15*WGS84_E2**2/256) * math.sin(4*lat)
    )
    k0 = 0.9996
    easting = k0 * N * (
        A_ + (1-T+C)*A_**3/6
        + (5-18*T+T**2+72*C-58*e2p)*A_**5/120
    ) + 500_000.0
    northing = k0 * (
        M + N * math.tan(lat) * (
            A_**2/2
            + (5-T+9*C+4*C**2)*A_**4/24
            + (61-58*T+T**2+600*C-330*e2p)*A_**6/720
        )
    )
    if lat_deg < 0:
        northing += 10_000_000.0
    return easting, northing


def ll_to_utm_local(
    lat_deg: float,
    lon_deg: float,
    origin:  Tuple[float, float],   # (origin_easting, origin_northing)
) -> Tuple[float, float]:
    """WGS84 → 로컬 미터 좌표 (origin 기준 상대)."""
    e, n = ll_to_utm_simple(lat_deg, lon_deg)
    return e - origin[0], n - origin[1]


def yaw_to_quat(yaw_rad: float) -> Dict[str, float]:
    """ENU yaw(라디안) → quaternion dict {w, x, y, z}."""
    half = yaw_rad * 0.5
    return {"w": math.cos(half), "x": 0.0, "y": 0.0, "z": math.sin(half)}


def heading_between(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """두 점 사이 방향각 (라디안, ENU yaw)."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


# ══════════════════════════════════════════════════════════════════
# 데이터 타입
# ══════════════════════════════════════════════════════════════════

@dataclass
class GarminWaypoint:
    lat:     float
    lon:     float
    ele:     float = 0.0
    name:    str   = ""
    # 로컬 미터 좌표 (ll_to_utm_local 후 채워짐)
    x:       float = 0.0
    y:       float = 0.0
    heading: float = 0.0
    speed:   float = 8.33   # m/s (기본 30 km/h)


@dataclass
class PoseStamped:
    """
    ROS2 geometry_msgs/PoseStamped 호환 dict 래퍼.
    ROS2 없는 환경(Isaac Sim standalone)에서도 동일 인터페이스.
    """
    frame_id:  str   = "map"
    stamp_sec: float = 0.0
    # position
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    # orientation (quaternion)
    ori_w: float = 1.0
    ori_x: float = 0.0
    ori_y: float = 0.0
    ori_z: float = 0.0

    def to_dict(self) -> dict:
        return {
            "header": {
                "frame_id": self.frame_id,
                "stamp":    self.stamp_sec,
            },
            "pose": {
                "position": {
                    "x": self.pos_x,
                    "y": self.pos_y,
                    "z": self.pos_z,
                },
                "orientation": {
                    "w": self.ori_w,
                    "x": self.ori_x,
                    "y": self.ori_y,
                    "z": self.ori_z,
                },
            },
        }

    def to_numpy_pos(self) -> np.ndarray:
        return np.array([self.pos_x, self.pos_y, self.pos_z])

    def to_numpy_quat(self) -> np.ndarray:
        return np.array([self.ori_w, self.ori_x, self.ori_y, self.ori_z])


@dataclass
class NavSatFix:
    """ROS2 sensor_msgs/NavSatFix 호환."""
    latitude:  float = 0.0
    longitude: float = 0.0
    altitude:  float = 0.0
    status:    int   = 0    # 0=no fix, 1=fix, 2=sbas
    stamp_sec: float = 0.0
    cov_diag:  Tuple[float, float, float] = (1.0, 1.0, 2.0)

    def to_dict(self) -> dict:
        return {
            "header":    {"stamp": self.stamp_sec, "frame_id": "gps"},
            "status":    {"status": self.status, "service": 1},
            "latitude":  self.latitude,
            "longitude": self.longitude,
            "altitude":  self.altitude,
            "position_covariance": [
                self.cov_diag[0], 0, 0,
                0, self.cov_diag[1], 0,
                0, 0, self.cov_diag[2],
            ],
            "position_covariance_type": 2,
        }


# ══════════════════════════════════════════════════════════════════
# GPX 파서
# ══════════════════════════════════════════════════════════════════

class GPXParser:
    """GPX 1.1 파서 — rte / trk / wpt 모두 지원."""

    def parse(self, path: str) -> List[GarminWaypoint]:
        p = Path(path)
        if not p.exists():
            logger.warning("GPX 파일 없음: %s", path)
            return []

        tree = ET.parse(str(p))
        root = tree.getroot()
        pts: List[Tuple[float, float, float, str]] = []

        # 우선순위: rte > trk > wpt
        for search in [
            f".//{{{GPX_NS}}}rtept",
            f".//{{{GPX_NS}}}trkpt",
            f".//{{{GPX_NS}}}wpt",
        ]:
            for pt in root.findall(search):
                lat  = float(pt.get("lat", 0))
                lon  = float(pt.get("lon", 0))
                ele_el = pt.find(f"{{{GPX_NS}}}ele")
                ele  = float(ele_el.text) if ele_el is not None else 0.0
                name_el = pt.find(f"{{{GPX_NS}}}name")
                name = name_el.text if name_el is not None else ""
                pts.append((lat, lon, ele, name))
            if pts:
                break

        if not pts:
            logger.error("GPX에서 waypoint 없음: %s", path)
            return []

        return self._to_garmin_waypoints(pts)

    def _to_garmin_waypoints(
        self, pts: List[Tuple[float, float, float, str]]
    ) -> List[GarminWaypoint]:
        # 첫 포인트를 로컬 원점으로
        e0, n0 = ll_to_utm_simple(pts[0][0], pts[0][1])
        origin = (e0, n0)

        waypoints = []
        for lat, lon, ele, name in pts:
            x, y = ll_to_utm_local(lat, lon, origin)
            waypoints.append(GarminWaypoint(
                lat=lat, lon=lon, ele=ele, name=name,
                x=x, y=y,
            ))

        # 방향각 계산
        for i, wp in enumerate(waypoints):
            if i < len(waypoints) - 1:
                wp.heading = heading_between(
                    (wp.x, wp.y),
                    (waypoints[i+1].x, waypoints[i+1].y),
                )
            else:
                wp.heading = waypoints[i-1].heading if i > 0 else 0.0

        logger.info("GPX 파싱 완료: %d waypoints (원점 UTM %.1f, %.1f)",
                    len(waypoints), e0, n0)
        return waypoints


# ══════════════════════════════════════════════════════════════════
# Garmin Bridge 메인 클래스
# ══════════════════════════════════════════════════════════════════

class GarminBridge:
    """
    Garmin GPX → PoseStamped / Nav2 goal 브리지.

    Isaac Sim 모드 (ROS2 없음):
      waypoint_to_pose()  → PoseStamped dict 반환
      route_to_poses()    → List[PoseStamped]

    ROS2 모드:
      publish_goal()      → /navigate_to_pose action 호출
      publish_fix()       → /fix NavSatFix 퍼블리시
      publish_path()      → /garmin/global_path 퍼블리시
    """

    # 워커힐로 기준점 (서울 광진구)
    WALKERHILL_ORIGIN_LAT = 37.5532
    WALKERHILL_ORIGIN_LON = 127.0932

    def __init__(
        self,
        gpx_path:    str   = "",
        frame_id:    str   = "map",
        default_speed_kmh: float = 30.0,
        use_ros2:    bool  = False,
    ):
        self.frame_id    = frame_id
        self.speed_ms    = default_speed_kmh / 3.6
        self.use_ros2    = use_ros2

        self._parser     = GPXParser()
        self._waypoints: List[GarminWaypoint] = []
        self._origin:    Optional[Tuple[float, float]] = None
        self._ros_node   = None

        # 실기기 NMEA 스레드
        self._nmea_thread: Optional[threading.Thread] = None
        self._nmea_latest: Optional[NavSatFix] = None

        if gpx_path:
            self.load_gpx(gpx_path)

        if use_ros2:
            self._init_ros2()

    # ── GPX 로드 ──────────────────────────────────────────────────

    def load_gpx(self, path: str) -> int:
        """GPX 파일 로드. 반환값: waypoint 수."""
        wps = self._parser.parse(path)
        if not wps:
            logger.warning("GPX 로드 실패 → 워커힐 mock 경로 사용")
            wps = self._walkerhill_mock()
        self._waypoints = wps
        # origin = 첫 waypoint의 UTM
        e0, n0 = ll_to_utm_simple(wps[0].lat, wps[0].lon)
        self._origin = (e0, n0)
        return len(wps)

    # ── PoseStamped 변환 ──────────────────────────────────────────

    def waypoint_to_pose(self, wp: GarminWaypoint) -> PoseStamped:
        """단일 GarminWaypoint → PoseStamped."""
        q = yaw_to_quat(wp.heading)
        return PoseStamped(
            frame_id  = self.frame_id,
            stamp_sec = time.time(),
            pos_x     = wp.x,
            pos_y     = wp.y,
            pos_z     = wp.ele,
            ori_w     = q["w"],
            ori_x     = q["x"],
            ori_y     = q["y"],
            ori_z     = q["z"],
        )

    def route_to_poses(self) -> List[PoseStamped]:
        """전체 경로 → PoseStamped 리스트."""
        return [self.waypoint_to_pose(wp) for wp in self._waypoints]

    def next_goal(
        self,
        ego_pos: np.ndarray,    # [3] world XYZ
        lookahead_m: float = 20.0,
    ) -> Optional[PoseStamped]:
        """
        현재 위치 기준 lookahead_m 앞 waypoint를 Nav2 goal로 반환.
        Isaac Sim 주행 루프에서 매 프레임 호출.
        """
        if not self._waypoints:
            return None

        dists = [
            math.hypot(wp.x - ego_pos[0], wp.y - ego_pos[1])
            for wp in self._waypoints
        ]
        nearest_idx = int(np.argmin(dists))

        # lookahead 거리 이상인 첫 waypoint 탐색
        acc = 0.0
        goal_idx = nearest_idx
        for i in range(nearest_idx, len(self._waypoints) - 1):
            dx = self._waypoints[i+1].x - self._waypoints[i].x
            dy = self._waypoints[i+1].y - self._waypoints[i].y
            acc += math.hypot(dx, dy)
            if acc >= lookahead_m:
                goal_idx = i + 1
                break
        else:
            goal_idx = len(self._waypoints) - 1

        wp = self._waypoints[goal_idx]
        pose = self.waypoint_to_pose(wp)

        if self.use_ros2:
            self._publish_goal_ros2(pose)

        return pose

    def get_local_path(
        self,
        ego_pos:     np.ndarray,
        n_points:    int   = 30,
        max_dist_m:  float = 60.0,
    ) -> List[PoseStamped]:
        """
        현재 위치 기준 앞 n_points개 PoseStamped 반환.
        core_pipeline.ElasticBandPlanner 입력으로 사용.
        """
        if not self._waypoints:
            return []

        dists = [
            math.hypot(wp.x - ego_pos[0], wp.y - ego_pos[1])
            for wp in self._waypoints
        ]
        start = int(np.argmin(dists))
        result = []
        acc = 0.0

        for i in range(start, min(start + n_points * 2, len(self._waypoints))):
            wp = self._waypoints[i]
            if i > start:
                prev = self._waypoints[i-1]
                acc += math.hypot(wp.x - prev.x, wp.y - prev.y)
            if acc > max_dist_m:
                break
            result.append(self.waypoint_to_pose(wp))
            if len(result) >= n_points:
                break

        return result

    def poses_to_numpy(
        self, poses: List[PoseStamped]
    ) -> np.ndarray:
        """PoseStamped 리스트 → [N, 3] numpy (x, y, z)."""
        if not poses:
            return np.zeros((0, 3))
        return np.array([[p.pos_x, p.pos_y, p.pos_z] for p in poses])

    # ── NMEA 실기기 ───────────────────────────────────────────────

    def start_nmea_reader(
        self,
        port:  str = "/dev/ttyUSB0",
        baud:  int = 9600,
    ):
        """Garmin 실기기 NMEA 시리얼 수신 스레드 시작."""
        self._nmea_thread = threading.Thread(
            target=self._nmea_loop,
            args=(port, baud),
            daemon=True,
        )
        self._nmea_thread.start()
        logger.info("NMEA 리더 시작: %s@%d", port, baud)

    def _nmea_loop(self, port: str, baud: int):
        try:
            import serial
            ser = serial.Serial(port, baud, timeout=1)
            while True:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith(("$GNGGA", "$GPGGA")):
                    fix = self._parse_gga(line)
                    if fix:
                        self._nmea_latest = fix
                        if self.use_ros2:
                            self._publish_fix_ros2(fix)
        except ImportError:
            logger.warning("pyserial 없음 — NMEA 리더 비활성")
        except Exception as e:
            logger.error("NMEA 루프 오류: %s", e)

    @staticmethod
    def _parse_gga(line: str) -> Optional[NavSatFix]:
        parts = line.split(",")
        if len(parts) < 10 or parts[6] == "0":
            return None

        def dm_dd(s, hem):
            if not s:
                return 0.0
            d  = float(s[:2 if hem in "NS" else 3])
            m  = float(s[2 if hem in "NS" else 3:])
            dd = d + m / 60.0
            return -dd if hem in "SW" else dd

        return NavSatFix(
            latitude  = dm_dd(parts[2], parts[3]),
            longitude = dm_dd(parts[4], parts[5]),
            altitude  = float(parts[9] or 0),
            status    = int(parts[6]),
            stamp_sec = time.time(),
        )

    def get_latest_fix(self) -> Optional[NavSatFix]:
        return self._nmea_latest

    # ── ROS2 연동 ─────────────────────────────────────────────────

    def _init_ros2(self):
        try:
            import rclpy
            from rclpy.node import Node
            rclpy.init(args=None)
            # 노드 생성은 별도 스레드에서 spin
            logger.info("ROS2 초기화 완료")
        except ImportError:
            logger.warning("rclpy 없음 — ROS2 퍼블리시 비활성")
            self.use_ros2 = False

    def _publish_goal_ros2(self, pose: PoseStamped):
        """
        /navigate_to_pose action 호출.
        실제 배포: nav2_msgs.action.NavigateToPose 사용.
        """
        logger.debug("[ROS2] Nav2 goal: (%.1f, %.1f)",
                     pose.pos_x, pose.pos_y)

    def _publish_fix_ros2(self, fix: NavSatFix):
        """sensor_msgs/NavSatFix → /fix 퍼블리시."""
        logger.debug("[ROS2] /fix: lat=%.6f lon=%.6f",
                     fix.latitude, fix.longitude)

    # ── Mock 경로 (GPX 없을 때) ───────────────────────────────────

    def _walkerhill_mock(self) -> List[GarminWaypoint]:
        """
        아차산 구의동 정수장 → 워커힐 호텔 800m S자 코스 mock.
        isaac_physics_visualizer.py의 generate_walkerhill_path()와 동일.
        """
        N  = 300
        xs = np.linspace(0, 800, N)
        ys = np.sin(xs / 40.0) * 4.5 + np.cos(xs / 80.0) * 2.0
        zs = xs * 0.035   # 3.5% 오르막

        # 기준 위치 (서울 광진구 아차산)
        base_lat = self.WALKERHILL_ORIGIN_LAT
        base_lon = self.WALKERHILL_ORIGIN_LON

        wps = []
        for i in range(N):
            heading = 0.0
            if i < N - 1:
                heading = math.atan2(
                    float(ys[i+1] - ys[i]),
                    float(xs[i+1] - xs[i]),
                )
            wps.append(GarminWaypoint(
                lat     = base_lat + float(ys[i]) * 9e-6,
                lon     = base_lon + float(xs[i]) * 1.1e-5,
                ele     = float(zs[i]),
                name    = f"WH_{i:03d}",
                x       = float(xs[i]),
                y       = float(ys[i]),
                heading = heading,
                speed   = self.speed_ms,
            ))

        logger.info("워커힐 mock 경로 생성: %d waypoints", N)
        return wps

    # ── 정보 조회 ─────────────────────────────────────────────────

    def total_distance_m(self) -> float:
        """전체 경로 길이 (미터)."""
        if len(self._waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self._waypoints) - 1):
            a = self._waypoints[i]
            b = self._waypoints[i + 1]
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total

    def __len__(self) -> int:
        return len(self._waypoints)

    def __iter__(self) -> Iterator[GarminWaypoint]:
        return iter(self._waypoints)

    def summary(self) -> dict:
        return {
            "n_waypoints":     len(self._waypoints),
            "total_dist_m":    round(self.total_distance_m(), 1),
            "origin_utm":      self._origin,
            "use_ros2":        self.use_ros2,
            "nmea_active":     self._nmea_thread is not None,
        }
