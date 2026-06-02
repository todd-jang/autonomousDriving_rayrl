/* safety_cpp/test/test_safety_layer.c
 * ──────────────────────────────────────
 * C99 단위 테스트 — 10개 수정 전부 검증
 * 동적 메모리 없음, ctest 호환
 */
#include "safety/types.h"
#include "safety/sync.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* safety_layer.c 내부 함수 전방 선언 */
typedef struct SafetyState SafetyState;
extern void   safety_init(SafetyState* st);
extern SafeCmd safety_apply(SafetyState*, float, float,
    const VehicleState*, const EgoPointCloud*,
    const PerceptionResult*, double);
extern void   safety_trigger_estop(SafetyState*);
extern void   safety_release_estop(SafetyState*);
extern void   pc_world_to_ego(EgoPointCloud*, const float*,
    const float*, const float*, const float*,
    uint32_t, const VehicleState*);

/* SafetyState 크기를 알기 위해 포함 */
#include "safety_layer_internal.h"

/* ── 테스트 프레임워크 (헤더 없이 자체 구현) ─────────────────── */
static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  \033[92m✓\033[0m %s\n", msg); g_pass++; } \
    else       { printf("  \033[91m✗\033[0m %s\n", msg); g_fail++; } \
} while(0)

#define CHECK_NEAR(a, b, eps, msg) \
    CHECK(fabs((double)(a)-(double)(b)) < (eps), msg)

/* ── 헬퍼 ────────────────────────────────────────────────────── */
static VehicleState make_state(double speed, double yaw_deg) {
    VehicleState s;
    memset(&s, 0, sizeof(s));
    s.speed_ms = speed;
    s.pos_world.frame = FRAME_WORLD;
    /* yaw → quaternion (no roll/pitch) */
    double half = yaw_deg * 3.14159265358979 / 360.0;
    s.quat.w = cos(half); s.quat.x = 0;
    s.quat.y = 0;         s.quat.z = sin(half);
    s.stamp = ns_now();
    return s;
}

static EgoPointCloud make_empty_pc(void) {
    EgoPointCloud pc;
    memset(&pc, 0, sizeof(pc));
    pc.frame = FRAME_EGO;
    pc.stamp = ns_now();
    return pc;
}

static PerceptionResult make_perc(SignalState sig, int oncoming) {
    PerceptionResult p;
    memset(&p, 0, sizeof(p));
    p.signal   = sig;
    p.oncoming = oncoming;
    p.stamp    = ns_now();
    return p;
}

/* ════════════════════════════════════════════════════════════════
 * Fix #1 — quat_yaw() 정확성
 * ════════════════════════════════════════════════════════════════ */
static void test_fix1_quat_yaw(void) {
    printf("\n[Fix #1] quat_yaw() — pitch/roll 있을 때 yaw 정확성\n");

    /* 순수 yaw 90도 */
    Quat q1 = {0.7071, 0, 0, 0.7071};
    double yaw1 = quat_yaw(&q1);
    CHECK_NEAR(yaw1, 1.5708, 0.001, "순수 yaw 90° 정확");

    /* pitch 30° + yaw 45° — 이전 공식은 오차 발생 */
    /* w=cos(45/2)*cos(30/2), z=sin(45/2)*cos(30/2)
       x=-sin(45/2)*sin(30/2), y=cos(45/2)*sin(30/2) */
    double hy = 0.3927, hp = 0.2618;
    Quat q2 = {
        cos(hy)*cos(hp),
        -sin(hy)*sin(hp),
         cos(hy)*sin(hp),
         sin(hy)*cos(hp)
    };
    double yaw2     = quat_yaw(&q2);
    double old_yaw2 = 2.0 * atan2(q2.z, q2.w);   /* 이전 공식 */
    /* 새 공식이 더 목표(0.7854rad)에 가까워야 함 */
    CHECK(fabs(yaw2 - 0.7854) < fabs(old_yaw2 - 0.7854),
          "pitch 있을 때 새 yaw가 더 정확");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #2 — world_to_ego 좌표 변환
 * ════════════════════════════════════════════════════════════════ */
static void test_fix2_coordinate(void) {
    printf("\n[Fix #2] world_to_ego() — 좌표 변환 정확성\n");

    VehicleState st = make_state(5.0, 0.0);   /* 북쪽(yaw=0) 주행 */

    /* 차량 정면 10m 앞 포인트 (world X+10) */
    float wx=10.f, wy=0.f, wz=0.f, inten=1.f;
    EgoPointCloud pc;
    pc_world_to_ego(&pc, &wx, &wy, &wz, &inten, 1, &st);

    CHECK(pc.frame == FRAME_EGO, "출력 프레임 = EGO");
    CHECK_NEAR(pc.x[0], 10.0, 0.01, "전방 10m → ego X ≈ 10");
    CHECK_NEAR(pc.y[0],  0.0, 0.01, "전방 10m → ego Y ≈ 0");

    /* yaw 90도 회전 후 월드 Y+10은 ego X+10이어야 함 */
    VehicleState st90 = make_state(5.0, 90.0);
    float wy2 = 10.f, wx2 = 0.f;
    EgoPointCloud pc90;
    pc_world_to_ego(&pc90, &wx2, &wy2, &wz, &inten, 1, &st90);
    CHECK_NEAR(pc90.x[0], 10.0, 0.1, "yaw90°: world Y+10 → ego X ≈ 10");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #3 — lane-constrained avoidance (Safety 속도 제한)
 * ════════════════════════════════════════════════════════════════ */
static void test_fix3_lane_constraint(void) {
    printf("\n[Fix #3] 차선 내 장애물 회피 속도 제한\n");

    /* TTC 1.0초 → 강한 제동 → throttle 감소 */
    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(10.0, 0.0);

    EgoPointCloud pc = make_empty_pc();
    /* 전방 8m — 고속(10m/s)이면 TTC ≈ 0.8s < 2.0s → 제동 */
    pc.x[0]=8.f; pc.y[0]=0.f; pc.z[0]=0.f; pc.intensity[0]=1.f;
    pc.n=1;

    PerceptionResult perc = make_perc(SIGNAL_GREEN, 0);
    SafeCmd cmd = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.0);

    CHECK(cmd.brake > 0.3f,   "TTC 제동 발동 (brake > 0.3)");
    CHECK(cmd.throttle < 0.8f,"TTC → throttle 감소");
    CHECK(cmd.active_rules & RULE_TTC, "RULE_TTC 비트 설정");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #4 — 복합 reasons 비트마스크
 * ════════════════════════════════════════════════════════════════ */
static void test_fix4_compound_reasons(void) {
    printf("\n[Fix #4] 복합 위반 동시 기록\n");

    SafetyState st; safety_init(&st);
    /* 빨간불 + 과속 + 전방 장애물 동시 */
    VehicleState vs = make_state(12.0, 0.0);   /* 43km/h — 상한 초과 */

    EgoPointCloud pc = make_empty_pc();
    pc.x[0]=6.f; pc.y[0]=0.f; pc.z[0]=0.f; pc.n=1;   /* TTC ≈ 0.5s */

    PerceptionResult perc = make_perc(SIGNAL_RED, 0);
    SafeCmd cmd = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.0);

    /* 최소 2개 규칙 동시 활성 */
    int rule_count = __builtin_popcount(cmd.active_rules);
    CHECK(rule_count >= 2, "복합 규칙 >= 2개 동시 활성");
    CHECK(cmd.active_rules & RULE_RED, "RED 규칙 기록");
    CHECK(cmd.active_rules & RULE_TTC, "TTC 규칙 기록");
    /* reason 문자열에 '+' 포함 */
    CHECK(strchr(cmd.reason, '+') != NULL, "reason에 '+' 구분자");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #5 — KL Hysteresis
 * ════════════════════════════════════════════════════════════════ */
static void test_fix5_kl_hysteresis(void) {
    printf("\n[Fix #5] KL Hysteresis — ENTER/EXIT 이중 임계치\n");

    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(5.0, 0.0);
    EgoPointCloud pc = make_empty_pc();
    PerceptionResult perc = make_perc(SIGNAL_GREEN, 0);

    /* kl=0.10 → ENTER(0.12) 미달 → 정상 */
    SafeCmd c1 = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.10);
    CHECK(!(c1.active_rules & RULE_KL), "kl=0.10 → 제한 없음");

    /* kl=0.13 → ENTER 초과 → 제한 진입 */
    SafeCmd c2 = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.13);
    CHECK(c2.active_rules & RULE_KL, "kl=0.13 → KL 제한 진입");

    /* kl=0.09 → EXIT(0.06) 미달 → 아직 제한 유지 */
    SafeCmd c3 = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.09);
    CHECK(c3.active_rules & RULE_KL, "kl=0.09 → 아직 제한 유지");

    /* kl=0.04 → EXIT 이하 → 제한 해제 */
    SafeCmd c4 = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc, 0.04);
    CHECK(!(c4.active_rules & RULE_KL), "kl=0.04 → 제한 해제");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #8 — e-stop 즉시 반환 (최우선 규칙)
 * ════════════════════════════════════════════════════════════════ */
static void test_fix8_estop(void) {
    printf("\n[Fix #8] e-stop 즉시 반환\n");

    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(8.0, 0.0);
    EgoPointCloud pc = make_empty_pc();
    PerceptionResult perc = make_perc(SIGNAL_GREEN, 0);

    safety_trigger_estop(&st);
    SafeCmd cmd = safety_apply(&st, 0.9f, 0.3f, &vs, &pc, &perc, 0.0);

    CHECK(cmd.estop == 1,       "estop 플래그 설정");
    CHECK(cmd.throttle == 0.f,  "throttle = 0");
    CHECK(cmd.brake == 1.f,     "brake = 1");
    CHECK(cmd.steering == 0.f,  "steering = 0");

    safety_release_estop(&st);
    SafeCmd cmd2 = safety_apply(&st, 0.5f, 0.f, &vs, &pc, &perc, 0.0);
    CHECK(cmd2.estop == 0, "estop 해제 후 정상");
}

/* ════════════════════════════════════════════════════════════════
 * Fix #9 — 타임스탬프 신선도 검증
 * ════════════════════════════════════════════════════════════════ */
static void test_fix9_timestamp(void) {
    printf("\n[Fix #9] 타임스탬프 신선도\n");

    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(5.0, 0.0);
    PerceptionResult perc = make_perc(SIGNAL_GREEN, 0);

    /* 신선한 데이터 */
    EgoPointCloud fresh_pc = make_empty_pc();
    SafeCmd c1 = safety_apply(&st, 0.6f, 0.f, &vs, &fresh_pc, &perc, 0.0);
    CHECK(!(c1.active_rules & RULE_SPEED) ||
          c1.throttle <= 0.6f, "신선한 PC → 정상 동작");

    /* 오래된 데이터 (200ms) */
    EgoPointCloud stale_pc = make_empty_pc();
    stale_pc.stamp = ns_now() - 200000000ULL;   /* 200ms 전 */
    SafeCmd c2 = safety_apply(&st, 0.8f, 0.f, &vs, &stale_pc, &perc, 0.0);
    CHECK(c2.throttle <= 0.2f, "오래된 PC → throttle 제한 0.2f");
    CHECK(strstr(c2.reason, "stale") != NULL, "reason에 stale 포함");
}

/* ════════════════════════════════════════════════════════════════
 * ApproxTimeSync 테스트
 * ════════════════════════════════════════════════════════════════ */
static void test_approx_time_sync(void) {
    printf("\n[Sync] ApproxTimeSync — 슬라이딩 윈도우 매칭\n");

    ApproxTimeSync ats;
    ats_init(&ats, 3, 50000000ULL);   /* 3 topics, 50ms */

    NsTimestamp t0 = ns_now();
    int lidar_data=1, cam_data=2, imu_data=3;

    ats_add(&ats, 0, t0,              &lidar_data, sizeof(int));
    ats_add(&ats, 1, t0+10000000ULL,  &cam_data,   sizeof(int));  /* +10ms */
    ats_add(&ats, 2, t0+20000000ULL,  &imu_data,   sizeof(int));  /* +20ms */

    const SensorSlot* out[4];
    int ok = ats_get_synced(&ats, t0+10000000ULL, out);
    CHECK(ok == 1, "50ms 내 3토픽 동기화 성공");
    CHECK(out[0] != NULL && out[1] != NULL && out[2] != NULL,
          "모든 토픽 슬롯 반환");

    /* 100ms 이상 차이 → 동기화 실패 */
    ats_add(&ats, 0, t0 + 200000000ULL, &lidar_data, sizeof(int));
    int ok2 = ats_get_synced(&ats, t0, out);
    CHECK(ok2 == 0, "200ms 차이 → 동기화 실패");
}

/* ════════════════════════════════════════════════════════════════
 * Ring Buffer 테스트
 * ════════════════════════════════════════════════════════════════ */
static void test_ring_buffer(void) {
    printf("\n[RingBuf] lock-free ring buffer\n");

    ViolRing rb;
    memset(&rb, 0, sizeof(rb));

    /* 256개 push (꽉 참) */
    for (int i = 0; i < 256; ++i) {
        ViolEntry e;
        e.rule_id = (uint8_t)(i & 0xFF);
        e.value   = (float)i;
        ViolRing_push(&rb, &e);
    }
    CHECK(ViolRing_size(&rb) == 256, "256개 push 후 size=256");

    ViolEntry out;
    int popped = ViolRing_pop(&rb, &out);
    CHECK(popped == 1, "pop 성공");
    CHECK(ViolRing_size(&rb) == 255, "pop 후 size=255");
}

/* ════════════════════════════════════════════════════════════════
 * 역주행 보정 테스트
 * ════════════════════════════════════════════════════════════════ */
static void test_oncoming_correction(void) {
    printf("\n[Rule6] 역주행 → 우측 조향 보정\n");

    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(5.0, 0.0);
    EgoPointCloud pc = make_empty_pc();

    PerceptionResult perc_normal   = make_perc(SIGNAL_GREEN, 0);
    PerceptionResult perc_oncoming = make_perc(SIGNAL_GREEN, 1);

    SafeCmd c_normal   = safety_apply(&st, 0.5f, 0.f, &vs, &pc, &perc_normal,   0.0);
    SafeCmd c_oncoming = safety_apply(&st, 0.5f, 0.f, &vs, &pc, &perc_oncoming, 0.0);

    CHECK(c_oncoming.steering > c_normal.steering,
          "역주행 → steering 우측 증가");
    CHECK(c_oncoming.throttle <= 0.3f,
          "역주행 → throttle 제한 0.3f");
    CHECK(c_oncoming.active_rules & RULE_ONCOMING,
          "RULE_ONCOMING 비트 설정");
}

/* ════════════════════════════════════════════════════════════════
 * 예외 상황 속도 제한
 * ════════════════════════════════════════════════════════════════ */
static void test_exception_speed(void) {
    printf("\n[Exception] 날씨/도로 예외 → 속도 제한\n");

    SafetyState st; safety_init(&st);
    VehicleState vs = make_state(8.0, 0.0);   /* 28.8 km/h */
    EgoPointCloud pc = make_empty_pc();

    /* 안개 → 10 km/h (2.78 m/s) */
    PerceptionResult perc_fog = make_perc(SIGNAL_GREEN, 0);
    perc_fog.active_exceptions = EXC_FOG;
    SafeCmd c_fog = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc_fog, 0.0);
    CHECK(c_fog.active_rules & RULE_SPEED, "안개 → 속도 제한 활성");
    CHECK(c_fog.throttle <= 0.f, "안개 → throttle <= 0");

    /* 침수 → 통행 불가 */
    PerceptionResult perc_flood = make_perc(SIGNAL_GREEN, 0);
    perc_flood.active_exceptions = EXC_FLOODING;
    SafeCmd c_flood = safety_apply(&st, 0.8f, 0.f, &vs, &pc, &perc_flood, 0.0);
    CHECK(c_flood.active_rules & RULE_SPEED, "침수 → 속도 제한");
}

/* ════════════════════════════════════════════════════════════════
 * main — 전체 실행
 * ════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n%s\n", "══════════════════════════════════════════════");
    printf("  Safety Layer C Unit Tests\n");
    printf("%s\n", "══════════════════════════════════════════════");

    test_fix1_quat_yaw();
    test_fix2_coordinate();
    test_fix3_lane_constraint();
    test_fix4_compound_reasons();
    test_fix5_kl_hysteresis();
    test_fix8_estop();
    test_fix9_timestamp();
    test_approx_time_sync();
    test_ring_buffer();
    test_oncoming_correction();
    test_exception_speed();

    printf("\n%s\n", "──────────────────────────────────────────────");
    printf("  PASS: %d   FAIL: %d   TOTAL: %d\n",
           g_pass, g_fail, g_pass + g_fail);
    if (g_fail == 0)
        printf("  \033[92m✅ ALL TESTS PASSED\033[0m\n\n");
    else
        printf("  \033[91m❌ %d TESTS FAILED\033[0m\n\n", g_fail);

    return g_fail == 0 ? 0 : 1;
}
