"""
dynamics_wrapper.py
────────────────────
Boltzmann Physics Domain Randomization

세 가지 현실-시뮬 갭(Reality Gap) 브리지:
  1. BoltzmannTerrainWrapper   : 미시 입자 산란 기반 동적 마찰
  2. BoltzmannDynamicsWrapper  : 토크 감쇠 + 제어 지연 (FR3 로봇 팔 / 자율주행)
  3. PhysicsDisruptionEnv      : Gymnasium 환경 래퍼 — 중력 + 바람 랜덤화

자율주행 적용:
  - BoltzmannTerrainWrapper → 포르쉐 Taycan 타이어 마찰 모델
  - PhysicsDisruptionEnv   → Isaac Sim 도메인 랜덤화 파라미터
  - BoltzmannDynamicsWrapper → ROS2 actuator latency 모델
"""
from __future__ import annotations
import time
import random
import gymnasium as gym
import numpy as np


# ══════════════════════════════════════════════════════════════════
# 1. Boltzmann Terrain — 동적 마찰 계수
# ══════════════════════════════════════════════════════════════════

class BoltzmannTerrainWrapper:
    """
    미시 입자 산란(Boltzmann collision approximation) 기반
    동적 타이어 마찰 계수 계산.

    자율주행 적용:
      - C Safety Layer speed_limit() 입력
      - Isaac Sim PhysX 재질 파라미터 갱신
      - RL reward modifier (마찰 낮을수록 역주행 패널티 증폭)
    """

    def __init__(self, base_friction: float = 0.6):
        self.base_friction = base_friction

    def calculate_boltzmann_drag(
        self,
        wheel_velocity: np.ndarray,
        particle_density: float = 1.2,
    ) -> np.ndarray:
        """
        바퀴 속도 → 유효 마찰 계수.

        Args:
            wheel_velocity: 바퀴 각속도 배열 [rad/s]
            particle_density: 도로 표면 입자 밀도 [g/m³]

        Returns:
            effective_friction: 각 바퀴의 유효 마찰 계수
        """
        # 입자 충돌 완화 시간 (relaxation time)
        tau = 0.05   # s

        # 열적/통계적 노이즈 (Boltzmann 분포 근사)
        statistical_variance = np.random.normal(0, 0.1, size=wheel_velocity.shape)

        # 거시 입자 충돌 비선형 모멘텀 감쇠
        collision_drag = (
            particle_density * wheel_velocity**2 * tau
        ) + statistical_variance

        # 입자 산란으로 인한 유효 마찰 감소
        effective_friction = self.base_friction - (0.02 * collision_drag)
        return np.clip(effective_friction, 0.1, 1.2)

    def get_isaac_physics_params(self, wheel_speeds: np.ndarray) -> dict:
        """Isaac Sim PhysX 재질 업데이트용 파라미터 반환."""
        frictions = self.calculate_boltzmann_drag(wheel_speeds)
        avg = float(frictions.mean())
        return {
            "static_friction":  avg,
            "dynamic_friction": avg * 0.85,
            "restitution":      0.1,
        }


# ══════════════════════════════════════════════════════════════════
# 2. Boltzmann Dynamics — 토크 감쇠 + 제어 지연
# ══════════════════════════════════════════════════════════════════

class BoltzmannDynamicsWrapper:
    """
    로봇/차량 액추에이터의 현실-시뮬 갭:
      A. 비선형 토크 감쇠 (미시 입자 충돌)
      B. 제어 지연 (Pi5 → FR3 / ROS2 latency)

    FR3 Joint 1 기준값: base_mass=3.06kg, max_effort=87Nm
    포르쉐 Taycan: base_mass=2295kg, max_effort=320Nm (모터)
    """

    def __init__(
        self,
        base_mass:          float,
        max_effort:         float,
        expected_latency_ms: float = 20.0,
    ):
        self.ideal_mass       = base_mass
        self.ideal_max_effort = max_effort
        self.latency_seconds  = expected_latency_ms / 1000.0
        self.command_history: list = []

    def compute_statistical_force(
        self,
        intended_torque: float,
        velocity:        float,
    ) -> float:
        """
        의도한 토크 → 실제 실행 토크 (Boltzmann 감쇠 적용).

        자율주행 적용:
          intended_torque = MPPI 출력 throttle × 최대토크
          velocity        = 현재 차량 속도 [m/s]
        """
        # 미세 충돌 산란 인자 (Gaussian, mean=1, std=0.15)
        collision_scatter = np.random.normal(loc=1.0, scale=0.15)

        # Boltzmann 운송 감쇠: 속도² 비례 비선형 마찰
        damping = 0.05 * (velocity ** 2) * collision_scatter

        # 실효 토크
        effective_torque = (intended_torque * collision_scatter) - damping

        # 열 드리프트로 인한 동적 최대값 변동 (±8%)
        dynamic_max = self.ideal_max_effort * np.random.uniform(0.92, 1.0)
        return float(np.clip(effective_torque, -dynamic_max, dynamic_max))

    def apply_control_latency(self, action: float) -> float:
        """
        제어 명령 지연 시뮬레이션.
        1~3 스텝 랜덤 지연 (통신 지터 모델).

        ROS2 적용:
          action = AckermannDriveStamped.drive.steering_angle
        """
        self.command_history.append(action)
        actual_lag = np.random.randint(1, 4)

        if len(self.command_history) < actual_lag:
            return 0.0   # 초기화 지연

        return self.command_history.pop(0)

    def apply_to_rl_action(
        self,
        throttle: float,
        steering: float,
        speed:    float,
    ) -> tuple[float, float]:
        """
        RL 정책 출력 (throttle, steering) → 현실적 액추에이터 출력.
        core_pipeline.py 에서 MPPI 출력에 적용.
        """
        delayed_thr = self.apply_control_latency(throttle)
        delayed_ste = self.apply_control_latency(steering)

        real_thr = self.compute_statistical_force(
            delayed_thr * self.ideal_max_effort, speed
        ) / self.ideal_max_effort

        real_ste = float(np.clip(
            delayed_ste + np.random.normal(0, 0.02), -1, 1
        ))
        return float(np.clip(real_thr, -1, 1)), real_ste


# ══════════════════════════════════════════════════════════════════
# 3. Physics Disruption Env (Gymnasium Wrapper)
# ══════════════════════════════════════════════════════════════════

class PhysicsDisruptionEnv(gym.Wrapper):
    """
    매 에피소드마다 물리 법칙을 변조하는 Gymnasium 래퍼.

    자율주행 RL 훈련에 적용:
      - 중력 변동: 적재 중량 변화 시뮬
      - 바람 노이즈: 횡풍 / 돌풍 효과
      - 마찰 변동: BoltzmannTerrainWrapper 연동

    Isaac Sim 적용:
      env = PhysicsDisruptionEnv(isaac_env)
      obs, _ = env.reset()   ← 매 에피소드 물리 재설정
    """

    def __init__(
        self,
        env: gym.Env,
        gravity_range:     tuple = (-6.0, -14.0),   # 표준 -9.81
        wind_noise_range:  tuple = (0.0, 2.5),
        friction_range:    tuple = (0.2, 1.0),
    ):
        super().__init__(env)
        self.gravity_range    = gravity_range
        self.wind_noise_range = wind_noise_range
        self.friction_range   = friction_range
        self.terrain          = BoltzmannTerrainWrapper()

        self._current_gravity   = -9.81
        self._current_wind      = 0.0
        self._current_friction  = 0.7

    def reset(self, **kwargs):
        # 매 에피소드: 물리 파라미터 랜덤화
        self._current_gravity  = random.uniform(*self.gravity_range)
        self._current_wind     = random.uniform(*self.wind_noise_range)
        self._current_friction = random.uniform(*self.friction_range)
        self.terrain.base_friction = self._current_friction

        print(
            f"  [Physics] gravity={self._current_gravity:.2f} "
            f"wind={self._current_wind:.2f} "
            f"friction={self._current_friction:.2f}"
        )

        # Isaac Sim 환경이면 실제 파라미터 주입
        self._apply_to_sim()

        return self.env.reset(**kwargs)

    def step(self, action):
        # 바람 → 횡방향 외란 추가
        if hasattr(action, '__len__') and len(action) >= 2:
            wind_perturbation = np.random.normal(0, self._current_wind * 0.05)
            action = np.array(action)
            action[1] = float(np.clip(
                action[1] + wind_perturbation, -1, 1
            ))
        return self.env.step(action)

    def _apply_to_sim(self):
        """Isaac Sim 물리 씬에 파라미터 적용 (있을 때만)."""
        try:
            from omni.isaac.core import SimulationContext
            # PhysX 장면 중력 업데이트
            ctx = SimulationContext.instance()
            if ctx:
                ctx.get_physics_context().set_gravity(
                    self._current_gravity
                )
        except ImportError:
            pass   # Isaac Sim 없으면 스킵

    @property
    def current_physics(self) -> dict:
        return {
            "gravity":   self._current_gravity,
            "wind":      self._current_wind,
            "friction":  self._current_friction,
        }


# ══════════════════════════════════════════════════════════════════
# 테스트 실행
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  Boltzmann Physics Domain Randomization Test")
    print("=" * 56)

    # ── Test 1: 타이어 마찰 ────────────────────────────────────
    print("\n[1] BoltzmannTerrainWrapper — 타이어 마찰")
    terrain = BoltzmannTerrainWrapper(base_friction=0.7)
    speeds  = np.array([10.0, 30.0, 60.0])   # rad/s
    frictions = terrain.calculate_boltzmann_drag(speeds)
    for v, f in zip(speeds, frictions):
        print(f"  wheel={v:.0f} rad/s → friction={f:.3f}")

    # ── Test 2: FR3 / Taycan 액추에이터 지연 ──────────────────
    print("\n[2] BoltzmannDynamicsWrapper — 토크 감쇠 + 지연")
    fr3 = BoltzmannDynamicsWrapper(base_mass=3.06, max_effort=87.0)
    for step in range(5):
        cmd    = 50.0
        delayed = fr3.apply_control_latency(cmd)
        actual  = fr3.compute_statistical_force(delayed, velocity=2.5)
        print(f"  step {step}: sent={cmd:.0f}Nm → executed={actual:.2f}Nm")

    # ── Test 3: Gymnasium 물리 랜덤화 ─────────────────────────
    print("\n[3] PhysicsDisruptionEnv — 물리 법칙 랜덤화")
    try:
        base_env = gym.make("LunarLander-v3")
        phys_env = PhysicsDisruptionEnv(base_env)
        for ep in range(3):
            print(f"\n  Episode {ep+1}:", end="")
            phys_env.reset()
            print(f"  physics={phys_env.current_physics}")
        phys_env.close()
    except Exception as e:
        print(f"  LunarLander 없음: {e} — mock 테스트")
        w = PhysicsDisruptionEnv.__new__(PhysicsDisruptionEnv)
        w.gravity_range=(-6,-14); w.wind_noise_range=(0,2.5)
        w.friction_range=(0.2,1.0)
        w.terrain=BoltzmannTerrainWrapper()
        w._current_gravity=-9.81; w._current_wind=0; w._current_friction=0.7
        for ep in range(3):
            w._current_gravity  = random.uniform(*w.gravity_range)
            w._current_wind     = random.uniform(*w.wind_noise_range)
            w._current_friction = random.uniform(*w.friction_range)
            print(f"  Episode {ep+1}: gravity={w._current_gravity:.2f} "
                  f"wind={w._current_wind:.2f} friction={w._current_friction:.2f}")

    print("\n✅ All Boltzmann physics tests complete")
