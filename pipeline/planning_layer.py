"""
pipeline/planning_layer.py
────────────────────────────
Planning Layer — 다이어그램 2번째 레이어

세 개의 서브모듈:
  1. HybridAStarPlanner   classical path planning
  2. VLMReasoningModule   scene understanding (Claude Vision API)
  3. GarminRouteBridge    GPX → waypoints [W×3]

출력: List[Waypoint] → Low-level RL 레이어 전달
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GPX_NS = "http://www.topografix.com/GPX/1/1"
WGS84_A  = 6_378_137.0
WGS84_E2 = 6.694379990141e-3


# ══════════════════════════════════════════════════════════════════
# 공통 타입
# ══════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    x: float; y: float; z: float = 0.0
    speed: float = 8.33   # 30 km/h
    heading: float = 0.0


# ══════════════════════════════════════════════════════════════════
# 1. Garmin GPX → Waypoint 브리지
# ══════════════════════════════════════════════════════════════════

def ll_to_utm_local(
    lat: float, lon: float,
    origin: Tuple[float, float],  # (origin_easting, origin_northing)
) -> Tuple[float, float]:
    """WGS84 → 로컬 미터 좌표 (origin 기준)."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    zone  = int((lon + 180) / 6) + 1
    lon0  = math.radians((zone - 1) * 6 - 180 + 3)
    e2p   = WGS84_E2 / (1 - WGS84_E2)
    N     = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
    T     = math.tan(lat_r)**2
    C     = e2p * math.cos(lat_r)**2
    A     = math.cos(lat_r) * (lon_r - lon0)
    M     = WGS84_A * (
        (1 - WGS84_E2/4 - 3*WGS84_E2**2/64) * lat_r
        - (3*WGS84_E2/8 + 3*WGS84_E2**2/32) * math.sin(2*lat_r)
    )
    k0 = 0.9996
    e  = k0 * N * (A + (1-T+C)*A**3/6) + 500_000.0
    n  = k0 * (M + N * math.tan(lat_r) * (A**2/2 + (5-T+9*C)*A**4/24))
    if lat < 0:
        n += 10_000_000.0
    return e - origin[0], n - origin[1]


class GarminRouteBridge:
    """
    GPX 파일 → Waypoint 목록.
    워커힐길 GPX를 로컬 미터 좌표로 변환.
    """

    def __init__(self):
        self._origin: Optional[Tuple[float, float]] = None

    def load_gpx(self, path: str) -> List[Waypoint]:
        p = Path(path)
        if not p.exists():
            logger.warning("GPX 없음 → 테스트 경로 생성: %s", path)
            return self._walkerhill_mock()

        tree = ET.parse(str(p))
        root = tree.getroot()
        pts  = []

        for tag in ["rte/rtept", "trk/trkseg/trkpt", "wpt"]:
            for pt in root.findall(f".//{{{GPX_NS}}}{tag.split('/')[-1]}"):
                pts.append((
                    float(pt.get("lat", 0)),
                    float(pt.get("lon", 0)),
                    float(getattr(pt.find(f"{{{GPX_NS}}}ele"), "text", 0) or 0),
                ))
            if pts:
                break

        if not pts:
            return self._walkerhill_mock()

        return self._to_waypoints(pts)

    def _to_waypoints(self, pts: list) -> List[Waypoint]:
        # 첫 포인트를 원점으로
        from pipeline.garmin_bridge import ll_to_utm_simple
        e0, n0 = ll_to_utm_simple(pts[0][0], pts[0][1])
        self._origin = (e0, n0)

        waypoints = []
        for lat, lon, ele in pts:
            x, y = ll_to_utm_local(lat, lon, self._origin)
            waypoints.append(Waypoint(x=x, y=y, z=ele))

        # 방향각 계산
        for i, wp in enumerate(waypoints):
            if i < len(waypoints) - 1:
                dx = waypoints[i+1].x - wp.x
                dy = waypoints[i+1].y - wp.y
                wp.heading = math.atan2(dy, dx)
            else:
                wp.heading = waypoints[i-1].heading if i > 0 else 0.0

        logger.info("GPX 로드: %d waypoints", len(waypoints))
        return waypoints

    def _walkerhill_mock(self) -> List[Waypoint]:
        """아차산 구의동 → 워커힐 800m S자 경로 (mock)."""
        N = 300
        xs = np.linspace(0, 800, N)
        ys = np.sin(xs / 40.0) * 4.5 + np.cos(xs / 80.0) * 2.0
        zs = xs * 0.035   # 3.5% 오르막
        wps = []
        for i in range(N):
            heading = 0.0
            if i < N - 1:
                heading = math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i])
            wps.append(Waypoint(
                x=float(xs[i]), y=float(ys[i]), z=float(zs[i]),
                heading=heading,
            ))
        return wps


# ══════════════════════════════════════════════════════════════════
# 2. Hybrid-A* Planner (Ackermann 기구학 인식)
# ══════════════════════════════════════════════════════════════════

class HybridAStarPlanner:
    """
    Ackermann 차량 기구학을 고려한 경로 계획.
    Nav2 SmacPlannerHybrid와 동일한 개념을 Python으로 경량 구현.

    실제 배포: Nav2 SmacPlannerHybrid 사용 (더 빠름).
    여기서는 Isaac Sim standalone 모드용 Python 구현.
    """

    def __init__(
        self,
        wheelbase:     float = 2.9,       # m (Porsche Taycan)
        min_turn_r:    float = 5.5,        # m
        steer_angles:  int   = 5,          # 이산 조향 각도 수
        grid_res:      float = 0.5,        # m
        angle_bins:    int   = 36,         # 방향 이산화
        max_iters:     int   = 5000,
    ):
        self.wheelbase  = wheelbase
        self.min_r      = min_turn_r
        self.n_steer    = steer_angles
        self.res        = grid_res
        self.n_angle    = angle_bins
        self.max_iters  = max_iters
        self.max_steer  = math.asin(wheelbase / min_turn_r)

    def plan(
        self,
        start: Waypoint,
        goal:  Waypoint,
        obstacles: np.ndarray = None,   # [M, 2] world XY
    ) -> List[Waypoint]:
        """
        Hybrid-A* 경로 계획.
        장애물이 없으면 직접 직선+곡선 연결.
        """
        if obstacles is None or len(obstacles) == 0:
            return self._direct_path(start, goal)

        # 장애물 있으면 A* 탐색
        return self._astar(start, goal, obstacles)

    def _direct_path(
        self, start: Waypoint, goal: Waypoint, n: int = 20
    ) -> List[Waypoint]:
        """장애물 없음: 부드러운 보간 경로."""
        wps = []
        for i in range(n + 1):
            t = i / n
            # Cubic Hermite 보간
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            scale = math.hypot(goal.x-start.x, goal.y-start.y)
            tx0   = math.cos(start.heading) * scale
            ty0   = math.sin(start.heading) * scale
            tx1   = math.cos(goal.heading)  * scale
            ty1   = math.sin(goal.heading)  * scale
            x = h00*start.x + h10*tx0 + h01*goal.x + h11*tx1
            y = h00*start.y + h10*ty0 + h01*goal.y + h11*ty1
            z = start.z + (goal.z - start.z) * t
            wps.append(Waypoint(x=x, y=y, z=z, speed=start.speed))
        return wps

    def _astar(
        self,
        start: Waypoint,
        goal:  Waypoint,
        obstacles: np.ndarray,
    ) -> List[Waypoint]:
        """
        간략화된 Hybrid-A* (실제 Nav2 수준은 아님, 데모용).
        실제: Nav2 SmacPlannerHybrid → ROS2 service 호출.
        """
        import heapq

        def state_to_key(x, y, yaw):
            ix = int(x / self.res)
            iy = int(y / self.res)
            ia = int(((yaw % (2*math.pi)) / (2*math.pi)) * self.n_angle)
            return (ix, iy, ia)

        def heuristic(x, y):
            return math.hypot(goal.x - x, goal.y - y)

        def is_collision(x, y):
            if len(obstacles) == 0:
                return False
            dists = np.linalg.norm(obstacles - [x, y], axis=1)
            return bool(dists.min() < 2.0)

        open_set = []
        heapq.heappush(open_set, (0, start.x, start.y, start.heading, []))
        visited = set()

        for _ in range(self.max_iters):
            if not open_set:
                break
            cost, cx, cy, cyaw, path = heapq.heappop(open_set)
            key = state_to_key(cx, cy, cyaw)
            if key in visited:
                continue
            visited.add(key)

            if math.hypot(cx - goal.x, cy - goal.y) < 2.0:
                path.append(Waypoint(x=cx, y=cy, heading=cyaw))
                return path if path else self._direct_path(start, goal)

            for steer in np.linspace(-self.max_steer, self.max_steer, self.n_steer):
                STEP = self.res * 2
                nx = cx + STEP * math.cos(cyaw)
                ny = cy + STEP * math.sin(cyaw)
                if self.wheelbase > 0 and abs(steer) > 1e-4:
                    R    = self.wheelbase / math.tan(steer)
                    nyaw = cyaw + STEP / R
                else:
                    nyaw = cyaw

                if is_collision(nx, ny):
                    continue
                new_path = path + [Waypoint(x=cx, y=cy, heading=cyaw)]
                g = cost + STEP
                h = heuristic(nx, ny)
                heapq.heappush(open_set, (g+h, nx, ny, nyaw, new_path))

        # 탐색 실패 → 직선 폴백
        logger.warning("Hybrid-A* 탐색 실패 → 직선 경로 폴백")
        return self._direct_path(start, goal)


# ══════════════════════════════════════════════════════════════════
# 3. VLM Reasoning Module (Claude Vision API)
# ══════════════════════════════════════════════════════════════════

class VLMReasoningModule:
    """
    카메라 프레임 + 현재 상황 → Claude Vision → 경로 수정 결정.

    예외 상황 처리:
      사고 현장, 침수 구간, 예상치 못한 공사 등
      classical planner가 처리 못하는 케이스를 VLM이 보완.

    출력: waypoint override (있을 때만)
    """

    SYSTEM_PROMPT = """You are an autonomous driving scene analyzer.
Analyze the road scene and output ONLY a JSON object.
Required format:
{
  "scene_type": "normal|accident|construction|flooding|emergency",
  "hazard_detected": true|false,
  "recommended_action": "continue|slow_down|stop|reroute|yield",
  "speed_override_kmh": null or number,
  "lateral_offset_m": 0.0,
  "reasoning": "brief reason"
}"""

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._cache: Optional[dict] = None
        self._last_call = 0.0
        self._min_interval = 2.0   # VLM은 0.5 Hz (느려도 됨)

    def analyze(
        self,
        frame_rgb: Optional[np.ndarray],  # [H, W, 3] uint8
        context: dict,
    ) -> Optional[dict]:
        """
        Returns VLM 분석 결과 dict 또는 None (캐시/실패).
        """
        now = time.time()
        if now - self._last_call < self._min_interval:
            return self._cache   # 이전 결과 재사용

        if not self._key or frame_rgb is None:
            return self._mock_analysis(context)

        try:
            import urllib.request
            b64 = self._encode_frame(frame_rgb)
            prompt = (
                f"Scene context: speed={context.get('speed_kmh',0):.1f}km/h, "
                f"weather_mask={context.get('weather_mask',0)}, "
                f"oncoming={context.get('oncoming_flag',0)}\n"
                f"Analyze this road scene."
            )
            payload = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ]}],
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key":         self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read())
            raw  = data["content"][0]["text"].strip()
            raw  = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            result = json.loads(raw)
            self._cache    = result
            self._last_call = now
            logger.info("[VLM] %s → %s",
                        result.get("scene_type"), result.get("recommended_action"))
            return result

        except Exception as e:
            logger.debug("[VLM] 분석 실패 → mock: %s", e)
            return self._mock_analysis(context)

    def _mock_analysis(self, context: dict) -> dict:
        """VLM 없을 때 규칙 기반 mock."""
        weather = int(context.get("weather_mask", 0))
        oncoming = int(context.get("oncoming_flag", 0))
        scene = "normal"
        action = "continue"
        speed_override = None

        if weather & 0x04:   # fog
            scene  = "adverse_weather"
            action = "slow_down"
            speed_override = 20.0
        if oncoming:
            scene  = "oncoming_traffic"
            action = "yield"
            speed_override = 15.0

        return {
            "scene_type":        scene,
            "hazard_detected":   oncoming or bool(weather),
            "recommended_action": action,
            "speed_override_kmh": speed_override,
            "lateral_offset_m":  0.0,
            "reasoning":         "mock_rule_based",
        }

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        import cv2
        small = cv2.resize(frame, (320, 180))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════════════════
# 통합 Planning Layer
# ══════════════════════════════════════════════════════════════════

class PlanningLayer:
    """
    다이어그램 2번째 레이어 — 완전판.

    plan() 출력 → core_pipeline.AutonomousProjectKernel에 전달.
    """

    def __init__(
        self,
        gpx_path: str = "",
        vlm_api_key: str = "",
        mode: str = "hybrid",   # "classical" | "vlm" | "hybrid"
    ):
        self.garmin  = GarminRouteBridge()
        self.astar   = HybridAStarPlanner()
        self.vlm     = VLMReasoningModule(vlm_api_key)
        self.mode    = mode

        self._route: List[Waypoint] = []
        if gpx_path:
            self._route = self.garmin.load_gpx(gpx_path)

    def load_route(self, gpx_path: str):
        self._route = self.garmin.load_gpx(gpx_path)

    def plan(
        self,
        state:     dict,            # EKF 출력 dict
        obstacles: np.ndarray,      # ego frame [M, 3]
        frame_rgb: Optional[np.ndarray] = None,
    ) -> List[Waypoint]:
        """
        Planning 주기 (5 Hz).
        Returns: local waypoint list (앞 30m)
        """
        pos  = np.array(state["ego_position"])
        quat = np.array(state["ego_orientation"])

        # 1. Garmin 경로에서 현재 위치 기준 로컬 추출
        local = self._extract_local(pos)

        # 2. VLM 분석 (mode에 따라)
        vlm_result = None
        if self.mode in ("vlm", "hybrid") and frame_rgb is not None:
            vlm_result = self.vlm.analyze(frame_rgb, {
                "speed_kmh":    state.get("velocity_m_s", 0) * 3.6,
                "weather_mask": state.get("weather_mask", 0),
                "oncoming_flag": state.get("oncoming_flag", 0),
            })

        # 3. VLM 결과로 경로 수정
        if vlm_result:
            local = self._apply_vlm(local, vlm_result)

        # 4. Hybrid-A* (장애물 있을 때)
        if len(obstacles) > 0 and len(local) >= 2:
            # obstacles를 world frame으로 역변환 (ego → world 근사)
            obs_world = obstacles[:, :2] + pos[:2]
            start = local[0]
            goal  = local[-1]
            local = self.astar.plan(start, goal, obs_world)

        return local

    def _extract_local(self, pos: np.ndarray) -> List[Waypoint]:
        """현재 위치 기준 앞 30m 경로 추출."""
        if not self._route:
            # 기본: 전방 직진
            return [
                Waypoint(x=pos[0]+d, y=pos[1], z=pos[2])
                for d in [5, 10, 15, 20, 25, 30]
            ]
        dists = [math.hypot(wp.x-pos[0], wp.y-pos[1])
                 for wp in self._route]
        start = int(np.argmin(dists))
        return self._route[start:start+20]

    def _apply_vlm(
        self,
        wps: List[Waypoint],
        vlm: dict,
    ) -> List[Waypoint]:
        """VLM 결과를 waypoint에 반영."""
        action   = vlm.get("recommended_action", "continue")
        speed_ov = vlm.get("speed_override_kmh")
        lat_off  = float(vlm.get("lateral_offset_m", 0.0))

        for wp in wps:
            if speed_ov:
                wp.speed = min(wp.speed, speed_ov / 3.6)
            if action == "stop":
                wp.speed = 0.0
            wp.y += lat_off   # 횡방향 오프셋

        if action in ("stop", "yield") and wps:
            # 앞 waypoint들을 감속 프로파일로 재설정
            for i, wp in enumerate(wps):
                wp.speed = max(0.0, wp.speed * (1 - i / len(wps)))

        return wps
