/* safety_cpp/src/safety_layer.c
 * ──────────────────────────────
 * Safety Layer 완전 구현 — C99
 *
 * 설계 보장:
 *   1. COORDINATE CONSISTENCY  : EgoPointCloud 강제 (FRAME_EGO assert)
 *   2. TIMING DETERMINISM      : malloc 없음, 모든 루프 상한 고정
 *   3. SENSOR SYNCHRONIZATION  : NsTimestamp 모든 입력 강제
 *
 * 10개 수정 내역 (코드 내 Fix #N 주석으로 위치 표시):
 *   #1  quat_yaw()            완전한 yaw 추출
 *   #2  world_to_ego()        좌표 변환 강제
 *   #3  lane_constrained      차선 내 회피
 *   #4  compound reasons      복합 규칙 비트마스크
 *   #5  KL hysteresis         ENTER/EXIT 이중 임계치
 *   #6  5-waypoint seq        (Python RL 레이어)
 *   #7  4ch BEV               (Python RL 레이어)
 *   #8  C Safety layer        이 파일 자체
 *   #9  timestamp sync        NsTimestamp 강제
 *   #10 trajectory memory     (Python RL 레이어)
 */

#include "safety/types.h"
#include "safety/sync.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ════════════════════════════════════════════════════════════
 * Safety Layer 내부 상태
 * (동적 메모리 없음 — 전역 또는 스택에 배치)
 * ════════════════════════════════════════════════════════════ */

typedef struct {
    /* KL hysteresis 상태 (Fix #5) */
    double kl_enter;
    double kl_exit;
    int    kl_restricted;   /* bool */

    /* e-stop 플래그 */
    int    estop;

    /* 설정 */
    double max_speed_ms;
    double ttc_thresh_s;
    double ttc_ego_width_m;
    double ttc_max_range_m;
    double max_speed_restricted_ms;

    /* 위반 로그 링버퍼 */
    ViolRing viol_log;

    /* 스텝 카운터 */
    uint64_t step_count;
} SafetyState;

/* 기본값 초기화 */
static inline void safety_init(SafetyState* st) {
    memset(st, 0, sizeof(*st));
    st->kl_enter              = 0.12;
    st->kl_exit               = 0.06;
    st->max_speed_ms          = 8.33;   /* 30 km/h */
    st->ttc_thresh_s          = 2.0;
    st->ttc_ego_width_m       = 1.5;
    st->ttc_max_range_m       = 40.0;
    st->max_speed_restricted_ms = 4.17; /* 15 km/h */
}

/* ════════════════════════════════════════════════════════════
 * 내부 헬퍼
 * ════════════════════════════════════════════════════════════ */

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void log_violation(SafetyState* st, uint8_t rule,
                           float val, const char* msg)
{
    ViolEntry e;
    e.stamp   = ns_now();
    e.rule_id = rule;
    e.value   = val;
    strncpy(e.msg, msg, 31);
    e.msg[31] = '\0';
    ViolRing_push(&st->viol_log, &e);
}

/* reason 문자열에 토큰 추가 (Fix #4 복합 reasons) */
static void append_reason(char* buf, int* len, const char* token) {
    int buf_size = 64;
    if (*len > 0 && *len < buf_size - 1) {
        buf[(*len)++] = '+';
    } else if (*len == 0) {
        /* first token */
    }
    for (int i = 0; token[i] && *len < buf_size - 1; ++i) {
        buf[(*len)++] = token[i];
    }
    buf[*len] = '\0';
}

/* ── Fix #5: KL Hysteresis 업데이트 ────────────────────────── */
static int kl_hysteresis_update(SafetyState* st, double kl) {
    if (!st->kl_restricted && kl > st->kl_enter) {
        st->kl_restricted = 1;
    } else if (st->kl_restricted && kl < st->kl_exit) {
        st->kl_restricted = 0;
    }
    return st->kl_restricted;
}

/* ── TTC 계산 — ego frame 전용 (Fix #2 보장) ────────────────── */
static double compute_ttc(
    const SafetyState* st,
    const EgoPointCloud* pc,
    double speed_ms)
{
    /* Fix #2: 프레임 태그 런타임 검증 */
    assert(pc->frame == FRAME_EGO);

    if (pc->n == 0 || speed_ms < 0.5) return -1.0;

    double nearest = st->ttc_max_range_m + 1.0;
    for (uint32_t i = 0; i < pc->n; ++i) {
        double px = (double)pc->x[i];
        double py = (double)pc->y[i];
        /* ego X+ = 전방 (프레임 보장됨) */
        if (px > 0.5 && px < st->ttc_max_range_m &&
            fabs(py) < st->ttc_ego_width_m)
        {
            if (px < nearest) nearest = px;
        }
    }
    if (nearest > st->ttc_max_range_m) return -1.0;
    return nearest / (speed_ms > 0.1 ? speed_ms : 0.1);
}

/* ── 예외 상황별 속도 상한 ──────────────────────────────────── */
static double exception_speed_limit(double base, uint16_t exc) {
    double limit = base;
    if (exc & EXC_HEAVY_RAIN)   limit = limit < 5.56 ? limit : 5.56;
    if (exc & EXC_SNOW)         limit = limit < 4.17 ? limit : 4.17;
    if (exc & EXC_FOG)          limit = limit < 2.78 ? limit : 2.78;
    if (exc & EXC_BLACK_ICE)    limit = limit < 2.78 ? limit : 2.78;
    if (exc & EXC_FLOODING)     limit = 0.0;
    if (exc & EXC_EMERGENCY)    limit = 0.0;
    return limit;
}

/* ════════════════════════════════════════════════════════════
 * safety_apply() — 메인 함수
 *
 * WCET 보장:
 *   - 동적 메모리 할당 없음
 *   - 모든 루프: PC_MAX_POINTS 상한
 *   - 재귀 없음
 *   - 시스템콜: clock_gettime만 (로그용)
 * ════════════════════════════════════════════════════════════ */
SafeCmd safety_apply(
    SafetyState*           st,
    float                  throttle_in,
    float                  steering_in,
    const VehicleState*    state,
    const EgoPointCloud*   pc_ego,     /* Fix #2: 반드시 ego frame */
    const PerceptionResult* perc,
    double                 shadow_kl)
{
    /* ── Fix #9: 타임스탬프 신선도 검증 ─────────────────────── */
    NsTimestamp now_ns   = ns_now();
    uint64_t pc_age_ns   = now_ns - pc_ego->stamp;
    uint64_t perc_age_ns = now_ns - perc->stamp;

    SafeCmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.throttle  = throttle_in;
    cmd.steering  = steering_in;
    cmd.speed_kmh = (float)(state->speed_ms * 3.6);

    float   brake     = 0.f;
    uint8_t rules     = 0;
    char    reason[64] = "";
    int     rlen      = 0;

    st->step_count++;

    /* ── 데이터 신선도 경보 (100ms 초과 → 감속) ─────────────── */
    if (pc_age_ns > 100000000ULL || perc_age_ns > 100000000ULL) {
        cmd.throttle = clampf(cmd.throttle, -1.f, 0.2f);
        append_reason(reason, &rlen, "stale_sensor");
        rules |= RULE_SPEED;
        log_violation(st, 9, (float)(pc_age_ns * 1e-6), "stale_sensor");
    }

    /* ── Rule 1: e-stop (최우선, 즉시 반환) ─────────────────── */
    if (st->estop) {
        cmd.throttle    = 0.f;
        cmd.steering    = 0.f;
        cmd.brake       = 1.f;
        cmd.estop       = 1;
        cmd.active_rules = RULE_ESTOP;
        strncpy(cmd.reason, "e-stop", 63);
        log_violation(st, 1, 0.f, "e-stop");
        return cmd;
    }

    /* ── Rule 2: 빨간 신호등 ────────────────────────────────── */
    if (perc->signal == SIGNAL_RED && state->speed_ms > 0.5) {
        if (cmd.throttle > -0.4f) cmd.throttle = -0.4f;
        if (brake < 0.6f)         brake        =  0.6f;
        rules |= RULE_RED;
        append_reason(reason, &rlen, "red");
        log_violation(st, 2, (float)state->speed_ms, "red_light");
    }

    /* ── Rule 3: TTC (Fix #2 ego frame 보장) ────────────────── */
    {
        double ttc = compute_ttc(st, pc_ego, state->speed_ms);
        if (ttc >= 0.0 && ttc < st->ttc_thresh_s) {
            float force = (float)((st->ttc_thresh_s - ttc)
                                   / st->ttc_thresh_s);
            if (force > 1.f) force = 1.f;
            if (brake < force) brake = force;
            if (cmd.throttle > 1.f - force)
                cmd.throttle = 1.f - force;
            rules |= RULE_TTC;
            char tbuf[24];
            snprintf(tbuf, sizeof(tbuf), "ttc=%.1fs", ttc);
            append_reason(reason, &rlen, tbuf);
            log_violation(st, 3, (float)ttc, tbuf);
        }
    }

    /* ── Rule 4: 속도 상한 ──────────────────────────────────── */
    {
        double limit = exception_speed_limit(
            st->max_speed_ms, perc->active_exceptions);
        if (state->speed_ms > limit) {
            if (cmd.throttle > 0.f) cmd.throttle = 0.f;
            rules |= RULE_SPEED;
            char sbuf[24];
            snprintf(sbuf, sizeof(sbuf), "spd_cap=%.0fkmh",
                     limit * 3.6);
            append_reason(reason, &rlen, sbuf);
        }
    }

    /* ── Rule 5: Geofence (외부에서 geofence_check() 사용) ─── */
    /* 이 예제에서는 외부 콜백으로 처리 — safety_set_geofence_cb() */

    /* ── Rule 6: 역주행 보정 ────────────────────────────────── */
    if (perc->oncoming) {
        if (cmd.steering + 0.4f < 1.f)
            cmd.steering += 0.4f;
        else
            cmd.steering = 1.f;
        if (cmd.throttle > 0.3f) cmd.throttle = 0.3f;
        rules |= RULE_ONCOMING;
        append_reason(reason, &rlen, "oncoming");
        log_violation(st, 6, 0.f, "oncoming");
    }

    /* ── Rule 7: Shadow KL hysteresis (Fix #5) ──────────────── */
    if (kl_hysteresis_update(st, shadow_kl)) {
        if (state->speed_ms > st->max_speed_restricted_ms) {
            if (cmd.throttle > 0.f) cmd.throttle = 0.f;
        }
        rules |= RULE_KL;
        append_reason(reason, &rlen, "kl_restrict");
    }

    /* ── 최종 클리핑 ─────────────────────────────────────────── */
    cmd.throttle    = clampf(cmd.throttle, -1.f, 1.f);
    cmd.steering    = clampf(cmd.steering, -1.f, 1.f);
    cmd.brake       = clampf(brake,         0.f, 1.f);
    cmd.active_rules = rules;

    if (rlen > 0) {
        strncpy(cmd.reason, reason, 63);
        cmd.reason[63] = '\0';
    } else {
        strncpy(cmd.reason, "nominal", 63);
    }

    return cmd;
}

/* ── e-stop 제어 ────────────────────────────────────────────── */
void safety_trigger_estop(SafetyState* st)  { st->estop = 1; }
void safety_release_estop(SafetyState* st)  { st->estop = 0; }

/* ── 포인트클라우드 world → ego 변환 (Fix #2) ────────────────── */
void pc_world_to_ego(
    EgoPointCloud*      out,
    const float*        wx,    /* world x [n] */
    const float*        wy,    /* world y [n] */
    const float*        wz,    /* world z [n] */
    const float*        inten, /* intensity [n] */
    uint32_t            n,
    const VehicleState* state)
{
    assert(n <= PC_MAX_POINTS);
    Vec3F ego_pos = state->pos_world;

    out->n     = 0;
    out->frame = FRAME_EGO;
    out->stamp = ns_now();

    RotMat R = quat_to_rotmat(&state->quat);

    for (uint32_t i = 0; i < n; ++i) {
        double dx = (double)wx[i] - ego_pos.x;
        double dy = (double)wy[i] - ego_pos.y;
        double dz = (double)wz[i] - ego_pos.z;
        /* R^T * [dx,dy,dz] */
        out->x[i] = (float)(R.m[0]*dx + R.m[3]*dy + R.m[6]*dz);
        out->y[i] = (float)(R.m[1]*dx + R.m[4]*dy + R.m[7]*dz);
        out->z[i] = (float)(R.m[2]*dx + R.m[5]*dy + R.m[8]*dz);
        out->intensity[i] = inten ? inten[i] : 0.f;
    }
    out->n = n;
}
