"""
isaac_physics_visualizer.py
─────────────────────────────
Isaac Sim Physics Visualizer — 완전판

제공된 stub을 완전히 대체:
  - SimulationApp 초기화 (headless/UI 선택)
  - 워커힐 3D 경로 (generate_walkerhill_path)
  - Garmin Bridge → PoseStamped → Nav2 goal
  - EKF Sensor Fusion 통합
  - Planning Layer (Hybrid-A* + VLM)
  - core_pipeline.AutonomousProjectKernel
  - Recovery Manager
  - C Safety Layer ctypes 연동
  - TensorBoard 실시간 로깅
  - Slack 알림 (@장기3)
  - 30fps HUD 오버레이

실행:
  # Isaac Sim Python 환경
  ./python.sh isaac_physics_visualizer.py

  # headless (서버)
  ./python.sh isaac_physics_visualizer.py --headless

  # mock 모드 (Isaac Sim 없이 파이프라인 테스트)
  python isaac_physics_visualizer.py --mock
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── CLI 파싱 (SimulationApp 초기화 전에) ─────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--headless",  action="store_true")
ap.add_argument("--mock",      action="store_true",
                help="Isaac Sim 없이 파이프라인만 실행")
ap.add_argument("--gpx",       default="garmin_maps/walkerhill.gpx")
ap.add_argument("--so",        default="./libsafety_core.so")
ap.add_argument("--width",     type=int, default=1920)
ap.add_argument("--height",    type=int, default=1080)
ap.add_argument("--max-steps", type=int, default=0,   help="0=무한")
args = ap.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("visualizer")

# ── Isaac Sim 초기화 ─────────────────────────────────────────────
simulation_app = None
ISAAC_AVAILABLE = False

if not args.mock:
    try:
        from omni.isaac.kit import SimulationApp
        rendering_config = {
            "headless": args.headless,
            "width":    args.width,
            "height":   args.height,
        }
        simulation_app  = SimulationApp(rendering_config)
        ISAAC_AVAILABLE = True
        logger.info("Isaac Sim 초기화 완료 (%dx%d headless=%s)",
                    args.width, args.height, args.headless)
    except ImportError:
        logger.warning("Isaac Sim 미감지 → mock 모드로 전환")
        args.mock = True

# ── Isaac Sim 의존 임포트 (초기화 이후) ──────────────────────────
if ISAAC_AVAILABLE:
    import omni.timeline
    import omni.kit.commands
    from omni.isaac.core import SimulationContext
    from omni.isaac.core.utils.stage import add_reference_to_stage

# ── 프로젝트 모듈 ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.garmin_bridge   import GarminBridge, NavSatFix
from pipeline.sensor_fusion   import EKFSensorFusion
from pipeline.planning_layer  import PlanningLayer
from pipeline.core_pipeline   import AutonomousProjectKernel
from recovery.recovery_control_kernel import (
    AutonomousRecoveryManager,
    execution_loop_with_recovery,
)


# ══════════════════════════════════════════════════════════════════
# 워커힐 경로 생성 (stub 함수 — 이제 Garmin Bridge 사용)
# ══════════════════════════════════════════════════════════════════

def generate_walkerhill_path() -> np.ndarray:
    """
    아차산 구의동 정수장 → 워커힐 호텔 3D 궤적.
    Garmin Bridge에서 로드, 없으면 수학적 mock.
    """
    bridge = GarminBridge(gpx_path=args.gpx)
    if len(bridge) > 0:
        poses = bridge.route_to_poses()
        path  = bridge.poses_to_numpy(poses)
        logger.info("Garmin 경로 로드: %d pts, %.0fm",
                    len(path), bridge.total_distance_m())
        return path

    # GPX 없으면 수학적 생성
    N  = 300
    xs = np.linspace(0, 800, N)
    ys = np.sin(xs / 40.0) * 4.5 + np.cos(xs / 80.0) * 2.0
    zs = xs * 0.035
    logger.info("워커힐 mock 경로 생성: %d pts", N)
    return np.column_stack([xs, ys, zs])


# ══════════════════════════════════════════════════════════════════
# LaneGraphConstrained (stub → 실제 구현)
# ══════════════════════════════════════════════════════════════════

class LaneGraphConstrained:
    """
    워커힐로 왕복 2차선 차선 경계 제약.
    Fix #3: 차선 내 회피만 허용 (±3.0m).
    """
    LANE_HALF_WIDTH = 3.0   # m

    def is_inside_lane(self, pos: np.ndarray) -> bool:
        """횡방향 변위 ±3.0m 기준 차선 이탈 여부."""
        return bool(abs(pos[1]) < self.LANE_HALF_WIDTH)

    def clamp_to_lane(self, pos: np.ndarray) -> np.ndarray:
        """차선 경계 내로 위치 클리핑."""
        clamped = pos.copy()
        clamped[1] = float(np.clip(pos[1],
                                   -self.LANE_HALF_WIDTH,
                                    self.LANE_HALF_WIDTH))
        return clamped

    def lateral_margin(self, pos: np.ndarray) -> float:
        """차선 중앙에서 거리 (m). 0 = 완벽 중앙."""
        return abs(float(pos[1]))


# ══════════════════════════════════════════════════════════════════
# TensorBoard 실시간 로거
# ══════════════════════════════════════════════════════════════════

class TBLogger:
    def __init__(self, log_dir: str = "runs/walkerhill"):
        self._writer = None
        self._step   = 0
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir)
            logger.info("TensorBoard: %s", log_dir)
        except ImportError:
            logger.debug("TensorBoard 없음 — 콘솔 로깅만")

    def log(self, tag: str, value: float, step: int = None):
        s = step if step is not None else self._step
        if self._writer:
            self._writer.add_scalar(tag, value, s)

    def log_dict(self, d: dict, step: int = None):
        s = step if step is not None else self._step
        for k, v in d.items():
            self.log(k, v, s)
        self._step = s + 1

    def close(self):
        if self._writer:
            self._writer.close()


# ══════════════════════════════════════════════════════════════════
# Slack 알림
# ══════════════════════════════════════════════════════════════════

class SlackNotifier:
    def __init__(self):
        self._token   = os.getenv("SLACK_BOT_TOKEN", "")
        self._channel = os.getenv("SLACK_ALERT_CHANNEL", "C09H6HV7GGY")
        self._eng_id  = os.getenv("ENGINEER_SLACK_ID",  "U09HNFL0B9S")
        self._last_t  = 0.0
        self._min_interval = 10.0   # 최소 10초 간격

    def alert(self, msg: str, level: str = "info"):
        now = time.time()
        if now - self._last_t < self._min_interval:
            return
        self._last_t = now
        if not self._token:
            logger.info("[Slack mock] %s: %s", level, msg)
            return
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "channel": self._channel,
                "text":    f"<@{self._eng_id}> {msg}",
            }).encode()
            req = urllib.request.Request(
                "https://slack.com/api/chat.postMessage",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type":  "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception as e:
            logger.debug("Slack 전송 실패: %s", e)


# ══════════════════════════════════════════════════════════════════
# HUD 오버레이 (헤드리스 모드에서는 콘솔 출력)
# ══════════════════════════════════════════════════════════════════

def print_hud(step: int, pos: np.ndarray, cmd: dict,
              recovery_report: dict, elapsed: float):
    if step % 30 != 0:
        return
    rules = cmd.get("active_rules", 0)
    rule_str = []
    if rules & 0x01: rule_str.append("ESTOP")
    if rules & 0x02: rule_str.append("RED")
    if rules & 0x04: rule_str.append("TTC")
    if rules & 0x08: rule_str.append("SPD")
    if rules & 0x20: rule_str.append("ONC")
    if rules & 0x40: rule_str.append("KL")

    print(
        f"\r[{step:04d}] "
        f"pos=({pos[0]:6.1f},{pos[1]:5.1f},{pos[2]:4.1f}) "
        f"thr={cmd['throttle']:+.2f} "
        f"ste={cmd['steering']:+.2f} "
        f"brk={cmd['brake']:.2f} "
        f"spd={cmd.get('speed_kmh',0):4.1f}km/h "
        f"rules=[{','.join(rule_str) or 'OK'}] "
        f"faults={recovery_report['consecutive_faults']} "
        f"fps={1/(elapsed+1e-9):.0f}",
        end="", flush=True,
    )


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 경로 & 레인 그래프 ────────────────────────────────────────
    walkerhill_3d_path = generate_walkerhill_path()
    lane_map = LaneGraphConstrained()
    garmin   = GarminBridge(gpx_path=args.gpx)

    # ── EKF ──────────────────────────────────────────────────────
    ekf = EKFSensorFusion()

    # ── Planning Layer ────────────────────────────────────────────
    planner = PlanningLayer(
        gpx_path    = args.gpx,
        vlm_api_key = os.getenv("ANTHROPIC_API_KEY", ""),
        mode        = "hybrid",
    )
    planner.load_route(args.gpx)

    # ── Core Kernel ───────────────────────────────────────────────
    autonomous_kernel = AutonomousProjectKernel(
        gpx_waypoints = walkerhill_3d_path,
        lane_graph    = lane_map,
        so_path       = args.so,
        K_samples     = 100,
        T_horizon     = 15,
    )

    # ── Recovery Manager ──────────────────────────────────────────
    recovery_manager = AutonomousRecoveryManager(autonomous_kernel)

    # ── 모니터링 ──────────────────────────────────────────────────
    tb      = TBLogger("runs/walkerhill")
    slack   = SlackNotifier()
    slack.alert("🚀 Isaac Sim 워커힐 시뮬레이션 시작")

    # ── Isaac Sim 컨텍스트 ────────────────────────────────────────
    if ISAAC_AVAILABLE:
        sim_context = SimulationContext(
            stage_units_in_meters = 1.0,
            physics_dt            = 1/60.0,
            rendering_dt          = 1/60.0,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        logger.info("Isaac Sim timeline 재생 시작")
    else:
        sim_context = None
        timeline    = None

    # ── 시뮬레이션 상태 ───────────────────────────────────────────
    sim_step   = 0
    start_time = time.time()

    # 차량 초기 위치
    ego_position    = np.array([0.0, 0.0, 0.0])
    ego_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    ego_velocity    = 0.0

    logger.info("시뮬레이션 루프 시작 (max_steps=%d)", args.max_steps)

    # ── 메인 루프 ─────────────────────────────────────────────────
    while True:
        t0 = time.time()

        # Isaac Sim 종료 감지
        if ISAAC_AVAILABLE:
            if not simulation_app.is_running():
                logger.info("Isaac Sim 창 닫힘 — 종료")
                break
            sim_context.step(render=True)
            if not timeline.is_playing():
                time.sleep(0.016)
                continue

        # ── 차량 상태 계산 ────────────────────────────────────────
        elapsed   = time.time() - start_time
        current_x = elapsed * 12.0   # 초속 12m (43 km/h)

        idx = min(int(current_x / 800 * 300), len(walkerhill_3d_path) - 1)
        target_pos = walkerhill_3d_path[idx]

        # 노면 진동 노이즈 (고주파)
        ego_position = np.array([
            float(target_pos[0]),
            float(target_pos[1]) + np.random.normal(0, 0.02),
            float(target_pos[2]),
        ])

        # Fix #1: 진행 방향으로 yaw 계산
        if idx < len(walkerhill_3d_path) - 1:
            dp  = walkerhill_3d_path[idx+1] - walkerhill_3d_path[idx]
            yaw = math.atan2(float(dp[1]), float(dp[0]))
        else:
            yaw = 0.0
        half = yaw * 0.5
        ego_orientation = np.array([
            math.cos(half), 0.0, 0.0, math.sin(half)
        ])
        ego_velocity = 12.0 + np.random.normal(0, 0.3)

        # ── EKF 업데이트 ──────────────────────────────────────────
        ekf.add_imu(
            stamp = time.time(),
            accel = np.array([0.1, 0.0, 9.81]),
            gyro  = np.array([0.0, 0.0, yaw * 0.01]),
            quat  = ego_orientation,
        )
        ekf_state = ekf.update()

        # ── Garmin: 다음 Nav2 goal ─────────────────────────────────
        next_goal = garmin.next_goal(ego_position, lookahead_m=20.0)

        # ── LiDAR 포인트 (전방 장애물 mock) ──────────────────────
        lidar_points = np.array([
            [15.0,  0.8, 0.0],
            [28.0, -1.2, 0.2],
        ])
        # Fix #2: lidar는 world frame — core_pipeline이 ego로 변환

        # ── Shadow KL 동적 변화 ───────────────────────────────────
        shadow_kl = 0.03 + math.sin(sim_step / 10.0) * 0.01

        # ── Isaac raw obs 패킷 ────────────────────────────────────
        isaac_raw_obs = {
            "ego_position":    ego_position,
            "ego_orientation": ego_orientation,
            "velocity_m_s":    ego_velocity,
            "lidar_scan_points": lidar_points,
            "traffic_light":   0,      # 녹색
            "oncoming_flag":   0,
            "weather_mask":    0,
            "shadow_kl":       shadow_kl,
        }

        # ── End-to-End 파이프라인 ─────────────────────────────────
        final_cmd = execution_loop_with_recovery(
            autonomous_kernel,
            isaac_raw_obs,
            recovery_manager,
        )

        # ── 차선 경계 검증 ────────────────────────────────────────
        if not lane_map.is_inside_lane(ego_position):
            slack.alert(
                f"⚠️ 차선 이탈! pos=({ego_position[0]:.1f},{ego_position[1]:.1f})",
                level="warning",
            )

        # ── TensorBoard 로깅 ──────────────────────────────────────
        tb.log_dict({
            "control/throttle":   final_cmd["throttle"],
            "control/steering":   final_cmd["steering"],
            "control/brake":      final_cmd["brake"],
            "state/speed_kmh":    final_cmd.get("speed_kmh", 0),
            "state/pos_x":        ego_position[0],
            "state/pos_y":        ego_position[1],
            "safety/shadow_kl":   shadow_kl,
            "safety/active_rules": final_cmd.get("active_rules", 0),
            "recovery/faults":    recovery_manager.stats.consecutive_faults,
            "recovery/total":     recovery_manager.stats.total_interventions,
            "lane/deviation_m":   lane_map.lateral_margin(ego_position),
        }, step=sim_step)

        # ── HUD 출력 ──────────────────────────────────────────────
        elapsed_frame = time.time() - t0
        print_hud(
            sim_step, ego_position, final_cmd,
            recovery_manager.report(), elapsed_frame,
        )

        # ── MRM 알림 ──────────────────────────────────────────────
        if recovery_manager.stats.mrm_triggered:
            slack.alert(
                "🛑 MRM 발동! 차량 긴급 정지. 확인 필요.",
                level="critical",
            )
            logger.critical("MRM 발동 — 루프 종료")
            break

        sim_step += 1

        # 최대 스텝 체크
        if args.max_steps > 0 and sim_step >= args.max_steps:
            logger.info("max_steps=%d 도달 — 정상 종료", args.max_steps)
            break

        # mock 모드: 60Hz 맞추기
        if args.mock:
            elapsed_frame = time.time() - t0
            sleep = max(0, 1/60.0 - elapsed_frame)
            time.sleep(sleep)

    # ── 종료 처리 ─────────────────────────────────────────────────
    print()   # HUD 줄바꿈
    if ISAAC_AVAILABLE and timeline:
        timeline.stop()
    if ISAAC_AVAILABLE and simulation_app:
        simulation_app.close()

    tb.close()

    # 최종 리포트
    report = recovery_manager.report()
    logger.info("=== 시뮬레이션 완료 ===")
    logger.info("총 스텝:      %d", sim_step)
    logger.info("총 개입:      %d", report["total_interventions"])
    logger.info("MRM 발동:     %s", report["mrm_triggered"])
    logger.info("시나리오 통계: %s", report["scenario_counts"])
    logger.info("경로 완주율:  %.1f%%",
                min(100.0, ego_position[0] / 800.0 * 100))

    slack.alert(
        f"🏁 시뮬레이션 완료\n"
        f"스텝: {sim_step} | 개입: {report['total_interventions']} | "
        f"완주: {min(100.0, ego_position[0]/800.0*100):.0f}%"
    )


if __name__ == "__main__":
    main()
