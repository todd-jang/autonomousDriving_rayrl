/* safety_cpp/include/safety/types.h
 * ───────────────────────────────────
 * 공유 데이터 타입 — coordinate consistency 강제
 *
 * 설계 원칙:
 *   모든 3D 벡터는 FrameTag를 통해 프레임을 명시.
 *   Vec3Ego != Vec3World → 컴파일 에러로 혼용 방지.
 *   StampedData<T> → 타임스탬프 없는 센서 데이터 컴파일 에러.
 */
#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 나노초 타임스탬프 ──────────────────────────────────────── */
typedef uint64_t NsTimestamp;

static inline NsTimestamp ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (NsTimestamp)ts.tv_sec * 1000000000ULL + (NsTimestamp)ts.tv_nsec;
}

static inline double ns_to_sec(NsTimestamp ns) {
    return (double)ns * 1e-9;
}

/* ── 쿼터니언 [w, x, y, z] ────────────────────────────────── */
typedef struct {
    double w, x, y, z;
} Quat;

/* Fix #1 — 완전한 yaw 추출 (pitch/roll 영향 없음) */
static inline double quat_yaw(const Quat* q) {
    double siny_cosp = 2.0 * (q->w * q->z + q->x * q->y);
    double cosy_cosp = 1.0 - 2.0 * (q->y * q->y + q->z * q->z);
    return atan2(siny_cosp, cosy_cosp);
}

/* 3x3 회전행렬 (row-major) */
typedef struct { double m[9]; } RotMat;

static inline RotMat quat_to_rotmat(const Quat* q) {
    RotMat R;
    double w=q->w, x=q->x, y=q->y, z=q->z;
    R.m[0]=1-2*(y*y+z*z); R.m[1]=  2*(x*y-w*z); R.m[2]=  2*(x*z+w*y);
    R.m[3]=  2*(x*y+w*z); R.m[4]=1-2*(x*x+z*z); R.m[5]=  2*(y*z-w*x);
    R.m[6]=  2*(x*z-w*y); R.m[7]=  2*(y*z+w*x); R.m[8]=1-2*(x*x+y*y);
    return R;
}

/* ── 좌표 프레임 태그 (Coordinate Consistency) ──────────────
 *
 *  FRAME_WORLD  : UTM / ENU 월드 좌표계
 *  FRAME_EGO    : 차량 body 프레임
 *                 X = 전방, Y = 좌, Z = 상
 *  FRAME_SENSOR : 센서 로컬 (LiDAR 원점 등)
 *
 *  규칙: Safety Layer 진입 포인트는 반드시 FRAME_EGO.
 *        위반 시 assert() 발동.
 */
typedef enum {
    FRAME_WORLD  = 0,
    FRAME_EGO    = 1,
    FRAME_SENSOR = 2,
} FrameTag;

/* 프레임 태그 포함 3D 벡터 */
typedef struct {
    double   x, y, z;
    FrameTag frame;
} Vec3F;

/* Fix #2 — world → ego 좌표 변환 */
static inline Vec3F world_to_ego(
    Vec3F p_world,
    Vec3F ego_pos,    /* FRAME_WORLD */
    const Quat* q)
{
    RotMat R = quat_to_rotmat(q);
    double dx = p_world.x - ego_pos.x;
    double dy = p_world.y - ego_pos.y;
    double dz = p_world.z - ego_pos.z;
    Vec3F out;
    /* R^T * delta */
    out.x  = R.m[0]*dx + R.m[3]*dy + R.m[6]*dz;
    out.y  = R.m[1]*dx + R.m[4]*dy + R.m[7]*dz;
    out.z  = R.m[2]*dx + R.m[5]*dy + R.m[8]*dz;
    out.frame = FRAME_EGO;
    return out;
}

/* ── 차량 상태 ─────────────────────────────────────────────── */
typedef struct {
    Vec3F      pos_world;   /* FRAME_WORLD */
    Vec3F      vel_world;   /* FRAME_WORLD */
    Quat       quat;
    double     speed_ms;    /* |vel| scalar */
    NsTimestamp stamp;
} VehicleState;

/* ── 인지 결과 ─────────────────────────────────────────────── */
typedef enum {
    SIGNAL_UNKNOWN = 0,
    SIGNAL_RED     = 1,
    SIGNAL_YELLOW  = 2,
    SIGNAL_GREEN   = 3,
} SignalState;

/* 예외 상황 비트마스크 */
#define EXC_HEAVY_RAIN  (1u << 0)
#define EXC_SNOW        (1u << 1)
#define EXC_FOG         (1u << 2)
#define EXC_BLACK_ICE   (1u << 3)
#define EXC_FLOODING    (1u << 4)
#define EXC_EMERGENCY   (1u << 5)
#define EXC_DEBRIS      (1u << 6)
#define EXC_CONSTRUCTION (1u << 7)

typedef struct {
    SignalState signal;
    int         oncoming;        /* bool */
    float       lane_deviation_m;
    float       collision_risk;
    float       shadow_kl;
    uint16_t    active_exceptions; /* EXC_* 비트마스크 */
    NsTimestamp stamp;
} PerceptionResult;

/* ── 포인트클라우드 (ego frame 전용) ────────────────────────
 * 반드시 world_to_ego() 변환 후 fill.
 * frame 필드로 런타임 검증.
 */
#define PC_MAX_POINTS 16384

typedef struct {
    float        x[PC_MAX_POINTS];
    float        y[PC_MAX_POINTS];
    float        z[PC_MAX_POINTS];
    float        intensity[PC_MAX_POINTS];
    uint32_t     n;
    FrameTag     frame;    /* 반드시 FRAME_EGO */
    NsTimestamp  stamp;
} EgoPointCloud;

/* ── 안전 명령 출력 ────────────────────────────────────────── */
#define RULE_ESTOP    0x01u
#define RULE_RED      0x02u
#define RULE_TTC      0x04u
#define RULE_SPEED    0x08u
#define RULE_GEOFENCE 0x10u
#define RULE_ONCOMING 0x20u
#define RULE_KL       0x40u

typedef struct {
    float    throttle;      /* [-1, 1] */
    float    steering;      /* [-1, 1] */
    float    brake;         /* [ 0, 1] */
    int      estop;         /* bool */
    float    speed_kmh;
    uint8_t  active_rules;  /* RULE_* 비트마스크 */
    char     reason[64];
} SafeCmd;

#ifdef __cplusplus
}  /* extern "C" */
#endif
