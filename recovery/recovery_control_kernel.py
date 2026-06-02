"""
recovery/recovery_control_kernel.py
──────────────────────────────────────
AutonomousRecoveryManager — 완전판

C 안전 레이어 개입 신호 → 자가 복구 메커니즘

복구 시나리오:
  A. shadow_kl_divergence only    → MPPI 샘플 확장 + 30% 감속
  B. ttc_breach + kl              → Elastic Band 재조정 + 긴급 제동
  C. persistent failure (>10회)   → MRM (Minimum Risk Maneuver)
  D. red_light                    → 점진적 감속 보장
  E. oncoming                     → 조향 보정 + 속도 제한
  F. stale_sensor                 → 보수적 저속 모드
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 복구 상태 추적
# ══════════════════════════════════════════════════════════════════

@dataclass
class RecoveryStats:
    total_interventions:   int   = 0
    consecutive_faults:    int   = 0
    last_fault_time:       float = 0.0
    last_recovery_action:  str   = "none"
    mrm_triggered:         bool  = False

    # 시나리오별 누적 카운트
    counts: Dict[str, int] = field(default_factory=lambda: {
        "kl_only":     0,
        "ttc_breach":  0,
        "persistent":  0,
        "red_light":   0,
        "oncoming":    0,
        "stale":       0,
    })


# ══════════════════════════════════════════════════════════════════
# Recovery Manager
# ══════════════════════════════════════════════════════════════════

class AutonomousRecoveryManager:
    """
    C Safety Layer 위에 올라타는 Python 자가 복구 레이어.

    isaac_physics_visualizer.py 루프:
      raw_cmd  = kernel.execute_control_cycle(obs)
      safe_cmd = recovery_manager.resolve_system_faults(raw_cmd)
      → safe_cmd을 차량에 인가
    """

    MRM_THRESHOLD      = 10     # 연속 fault 횟수 → MRM 전환
    STABILIZE_WINDOW   = 3.0    # 초: 이 시간 fault 없으면 카운트 감소
    KL_ONLY_DECEL      = 0.7    # KL 단독 → 30% 감속
    KL_ONLY_STEER_DAMP = 0.5    # KL 단독 → 조향 50% 댐핑
    TTC_BRAKE_FORCE    = 0.4    # TTC 위반 → 40% 제동
    TTC_SPEED_CAP      = 4.17   # TTC 회복 중 속도 상한 (15 km/h)

    def __init__(self, core_kernel):
        self.kernel = core_kernel
        self.stats  = RecoveryStats()
        self._last_stable_t = time.time()

    # ── 메인 진입점 ──────────────────────────────────────────────

    def resolve_system_faults(self, safe_cmd: dict) -> dict:
        """
        C 레이어 출력 safe_cmd를 수신하여
        개입 여부에 따라 복구 액션 적용 후 반환.
        """
        now = time.time()

        if not safe_cmd.get("intervened", False):
            # 정상 주행 — 안정화 (카운트 감소)
            if now - self._last_stable_t > self.STABILIZE_WINDOW:
                self.stats.consecutive_faults = max(
                    0, self.stats.consecutive_faults - 1
                )
                self._last_stable_t = now
            return safe_cmd

        # ── fault 발생 ────────────────────────────────────────────
        reasons = safe_cmd.get("reason", [])
        self.stats.total_interventions += 1
        self.stats.consecutive_faults  += 1
        self.stats.last_fault_time      = now

        logger.warning(
            "[Recovery] fault #%d (consecutive=%d) reasons=%s",
            self.stats.total_interventions,
            self.stats.consecutive_faults,
            reasons,
        )

        # ── MRM: 연속 fault 임계치 초과 → 최소 위험 기동 ─────────
        if self.stats.consecutive_faults > self.MRM_THRESHOLD:
            return self._mrm(safe_cmd, reasons)

        # ── 시나리오별 분기 ───────────────────────────────────────
        has_kl  = any("kl" in r or "shadow" in r for r in reasons)
        has_ttc = any("ttc" in r for r in reasons)
        has_red = "red_light" in reasons
        has_onc = "oncoming_correction" in reasons
        has_stl = any("stale" in r for r in reasons)

        if has_stl:
            safe_cmd = self._recover_stale_sensor(safe_cmd)
        if has_ttc:
            safe_cmd = self._recover_ttc(safe_cmd, reasons)
        elif has_kl and not has_ttc:
            safe_cmd = self._recover_kl_only(safe_cmd)
        if has_red:
            safe_cmd = self._recover_red_light(safe_cmd)
        if has_onc:
            safe_cmd = self._recover_oncoming(safe_cmd)

        self.stats.last_recovery_action = "+".join(reasons[:3])
        return safe_cmd

    # ── 복구 시나리오 A: KL 단독 ─────────────────────────────────

    def _recover_kl_only(self, cmd: dict) -> dict:
        """AI 판단 불확실성 상승 → MPPI 정밀도 향상 + 감속."""
        logger.info("[Recovery A] KL only → MPPI 샘플 확장")

        # MPPI 샘플 수 2배, 수평선 단축 (빠른 수렴)
        self.kernel.K_samples = min(
            self.kernel.K_samples * 2, 500
        )
        self.kernel.T_horizon = max(
            self.kernel.T_horizon - 3, 8
        )

        cmd["throttle"] = cmd["throttle"] * self.KL_ONLY_DECEL
        cmd["steering"] = cmd["steering"] * self.KL_ONLY_STEER_DAMP
        self.stats.counts["kl_only"] += 1
        return cmd

    # ── 복구 시나리오 B: TTC 위반 ────────────────────────────────

    def _recover_ttc(self, cmd: dict, reasons: List[str]) -> dict:
        """전방 충돌 위험 → Elastic Band 재조정 + 제동."""
        logger.info("[Recovery B] TTC breach → EB 재조정 + 제동")

        # Elastic Band 반발력 증가 (더 적극적 회피)
        self.kernel.elastic_band.repulsion = min(
            self.kernel.elastic_band.repulsion * 1.5, 8.0
        )
        self.kernel.elastic_band.obs_radius = min(
            self.kernel.elastic_band.obs_radius + 1.0, 8.0
        )

        # MPPI: 장애물 비용 가중치 높이기
        self.kernel.mppi.sigma_w = min(
            self.kernel.mppi.sigma_w * 1.3, 1.0
        )

        cmd["brake"]    = max(cmd["brake"], self.TTC_BRAKE_FORCE)
        cmd["throttle"] = min(cmd["throttle"], 0.0)

        # 속도가 높으면 더 강한 제동
        speed_kmh = cmd.get("speed_kmh", 0.0)
        if speed_kmh > 30.0:
            cmd["brake"] = max(cmd["brake"], 0.7)

        self.stats.counts["ttc_breach"] += 1
        return cmd

    # ── 복구 시나리오 C: MRM ─────────────────────────────────────

    def _mrm(self, cmd: dict, reasons: List[str]) -> dict:
        """
        Minimum Risk Maneuver — 연속 fault 지속.
        차량을 도로 갓길에 안전하게 정지.
        """
        if not self.stats.mrm_triggered:
            logger.critical(
                "[Recovery C] MRM 발동! consecutive_faults=%d reasons=%s",
                self.stats.consecutive_faults, reasons,
            )
            self.stats.mrm_triggered = True

        cmd["throttle"] = 0.0
        cmd["brake"]    = 1.0
        cmd["steering"] = 0.0   # 직진 정지
        cmd["estop"]    = True
        cmd["reason"]   = ["MRM_persistent_failure"]
        self.stats.counts["persistent"] += 1
        return cmd

    # ── 복구 시나리오 D: 빨간 신호 ───────────────────────────────

    def _recover_red_light(self, cmd: dict) -> dict:
        """빨간불 통과 방지 — 점진적 감속 보장."""
        speed_kmh = cmd.get("speed_kmh", 0.0)
        if speed_kmh > 5.0:
            decel = min(0.8, speed_kmh / 30.0)
            cmd["brake"]    = max(cmd["brake"], decel)
            cmd["throttle"] = min(cmd["throttle"], -0.3)
        self.stats.counts["red_light"] += 1
        return cmd

    # ── 복구 시나리오 E: 역주행 ──────────────────────────────────

    def _recover_oncoming(self, cmd: dict) -> dict:
        """역주행 감지 → 조향 보정 강화 + 속도 제한."""
        cmd["steering"] = float(
            np.clip(cmd["steering"] + 0.15, -1, 1)
        )
        cmd["throttle"] = min(cmd["throttle"], 0.25)
        self.stats.counts["oncoming"] += 1
        return cmd

    # ── 복구 시나리오 F: 오래된 센서 ─────────────────────────────

    def _recover_stale_sensor(self, cmd: dict) -> dict:
        """센서 데이터 지연 → 보수적 저속 모드."""
        cmd["throttle"] = min(cmd["throttle"], 0.15)
        cmd["steering"] = cmd["steering"] * 0.6
        self.stats.counts["stale"] += 1
        return cmd

    # ── 리포트 ───────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "total_interventions":  self.stats.total_interventions,
            "consecutive_faults":   self.stats.consecutive_faults,
            "mrm_triggered":        self.stats.mrm_triggered,
            "last_action":          self.stats.last_recovery_action,
            "scenario_counts":      dict(self.stats.counts),
        }

    def reset_mrm(self):
        """수동 MRM 해제 (엔지니어 승인 후)."""
        self.stats.mrm_triggered   = False
        self.stats.consecutive_faults = 0
        logger.info("[Recovery] MRM 해제됨")


# ── 편의 함수 ──────────────────────────────────────────────────────

def execution_loop_with_recovery(
    autonomous_kernel,
    visualizer_obs: dict,
    recovery_manager: Optional[AutonomousRecoveryManager] = None,
) -> dict:
    """
    isaac_physics_visualizer.py 메인 루프에서 호출하는 단일 진입점.

    1. core_pipeline.execute_control_cycle()
    2. recovery_manager.resolve_system_faults()
    """
    if recovery_manager is None:
        recovery_manager = AutonomousRecoveryManager(autonomous_kernel)

    raw_cmd   = autonomous_kernel.execute_control_cycle(visualizer_obs)
    final_cmd = recovery_manager.resolve_system_faults(raw_cmd)
    return final_cmd
