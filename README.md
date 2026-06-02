# 🚗 AutoDrive RayRL — Porsche Taycan Sim2Real

**Ray RLlib PPO 기반 자율주행 Sim2Real 프로젝트**  
Isaac Sim 디지털 트윈 → 포르쉐 Taycan 실차 배포

---

## 아키텍처

```
LiDAR + Garmin GPS + Camera+IMU + HD Map
         ↓ EKF Sensor Fusion → state[N×13]
Planning Layer (Hybrid-A* / VLM / Garmin→Nav2)
         ↓ waypoints[W×3]
Low-level Layer (Ray RLlib PPO)
         ↓ raw cmd [throttle, steer]
Safety Layer rule-based (절대 우선) ← C99 구현
  e-stop | speed cap+TL | TTC | geofence
         ↓
safe cmd → AckermannDriveStamped → 포르쉐 Taycan
```

---

## 프로젝트 구조

```
autonomousDriving_rayrl/
├── isaac_physics_visualizer.py   # Isaac Sim 메인 루프
├── dynamics_wrapper.py           # Boltzmann 물리 도메인 랜덤화
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── sensor_fusion.py          # EKF 멀티센서 퓨전
│   ├── planning_layer.py         # Hybrid-A* + VLM + Garmin
│   ├── garmin_bridge.py          # GPX → UTM → PoseStamped
│   └── core_pipeline.py          # ElasticBand → MPPI → C Safety
│
├── recovery/
│   └── recovery_control_kernel.py  # 자가 복구 매니저
│
├── autonomy/
│   └── three_layer_stack_v2.py   # 3계층 스택 (10개 버그 수정)
│
├── rllib/
│   ├── smoke_test.py             # PerceptionEnv + PPO 5iter
│   └── shadow_validation.py      # KL divergence 배포 게이트
│
├── mlops/
│   └── mlops_pipeline.py         # TensorBoard→스트레스→TRT→Orin→Shadow
│
├── world_builder/
│   └── garmin_to_usd.py          # GPX → Isaac Sim USD
│
├── safety_cpp/
│   ├── include/safety/
│   │   ├── types.h               # 좌표 프레임 태그, 타임스탬프
│   │   └── sync.h                # RingBuffer, ApproxTimeSync
│   ├── src/safety_layer.c        # safety_apply() WCET 보장
│   ├── test/test_safety_layer.c  # 단위 테스트 (11개)
│   └── CMakeLists.txt
│
└── ros2_ws/
    ├── launch/autodrive.launch.py
    └── config/nav2_params.yaml
```

---

## 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/todd-jang/autonomousDriving_rayrl.git
cd autonomousDriving_rayrl
cp .env.example .env   # 토큰 입력
pip install -r requirements.txt
```

### 2. C Safety Layer 빌드

```bash
cd safety_cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest -V
```

### 3. Isaac Sim 실행 (mock 모드)

```bash
# Isaac Sim 없이 파이프라인 테스트
python isaac_physics_visualizer.py --mock

# Isaac Sim 설치된 환경
./python.sh isaac_physics_visualizer.py
```

### 4. RLlib Smoke Test

```bash
python -m rllib.smoke_test
```

### 5. MLOps 파이프라인

```bash
python -m mlops.mlops_pipeline \
    --orin-host 192.168.1.200 \
    --slack-id  U09HNFL0B9S
```

### 6. Boltzmann 물리 도메인 랜덤화 테스트

```bash
python dynamics_wrapper.py
```

---

## 10개 핵심 버그 수정 (three_layer_stack_v2.py)

| # | 수정 내용 |
|---|-----------|
| 1 | `quat_to_yaw()` — 완전한 quaternion yaw (pitch/roll 무관) |
| 2 | `world_to_ego()` — 좌표 변환 강제 (FrameTag 컴파일 시간 체크) |
| 3 | LaneConstrainedAvoider — 차선 경계 내 회피만 허용 |
| 4 | 복합 reasons 리스트 — 동시 규칙 로그 |
| 5 | KLHysteresis — ENTER=0.12 / EXIT=0.06 이중 임계치 |
| 6 | 5-waypoint sequence obs — 곡률 사전 인식 |
| 7 | 4채널 BEV — occupancy/height/intensity/dynamic |
| 8 | C Safety Layer — Python→C 구현 (WCET 보장) |
| 9 | ApproxTimeSync — NsTimestamp 강제, 50ms 슬라이딩 윈도우 |
| 10 | TrajectoryMemory — LSTM 히스토리 버퍼 [H×OBS_DIM] |

---

## MLOps 5단계

```
1. TensorBoard Plateau 감지 → 모델 풀
2. 가혹 조건 스트레스 테스트 (마찰↓/중량↑)
3. KL 이탈 최소 모델 스크리닝
4. TensorRT 변환 + Orin 60Hz 검증
5. Shadow Driving → 배포 승인
```

---

## 환경 변수

`.env.example` 참조. 필수 항목:

```
SLACK_BOT_TOKEN      Slack 봇 토큰
ANTHROPIC_API_KEY    VLM reasoning (Claude Vision)
GITHUB_TOKEN         CI/CD 자동화
VEHICLE_HOST         실차 SSH IP
ORIN_HOST            Jetson Orin IP
```

---

## 라이선스

MIT License — 연구·교육 목적 자유 사용
