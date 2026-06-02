/* safety_cpp/include/safety/sync.h
 * ──────────────────────────────────
 * Sensor Synchronization — timing determinism
 *
 * 구성:
 *   RingBuffer<T,N>   : lock-free, O(1), 동적 할당 없음
 *   SensorBuffer<T,N> : 타임스탬프 기반 nearest-match
 *   ApproxTimeSync    : 다중 센서 슬라이딩 윈도우 동기화
 */
#pragma once
#include "types.h"
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════
 * Lock-free Ring Buffer (WCET O(1), no malloc)
 * N must be power-of-2
 * ════════════════════════════════════════════════════════════ */

#define RINGBUF_DEFINE(T, N, NAME)                                  \
    typedef struct {                                                  \
        T        buf[N];                                              \
        volatile uint32_t head;                                       \
        volatile uint32_t tail;                                       \
    } NAME;                                                           \
                                                                      \
    static inline void NAME##_push(NAME* rb, const T* val) {         \
        uint32_t h = rb->head & (N-1);                               \
        memcpy(&rb->buf[h], val, sizeof(T));                         \
        rb->head++;                                                   \
    }                                                                 \
                                                                      \
    static inline int NAME##_pop(NAME* rb, T* out) {                 \
        if (rb->head == rb->tail) return 0;                          \
        uint32_t t = rb->tail & (N-1);                               \
        memcpy(out, &rb->buf[t], sizeof(T));                         \
        rb->tail++;                                                   \
        return 1;                                                     \
    }                                                                 \
                                                                      \
    static inline uint32_t NAME##_size(const NAME* rb) {             \
        return rb->head - rb->tail;                                   \
    }

/* ── 위반 로그 엔트리 ──────────────────────────────────────── */
typedef struct {
    NsTimestamp stamp;
    uint8_t     rule_id;
    float       value;
    char        msg[32];
} ViolEntry;

RINGBUF_DEFINE(ViolEntry, 256, ViolRing)

/* ════════════════════════════════════════════════════════════
 * SensorBuffer — 타임스탬프 기반 nearest-match
 * N = 링 크기 (power-of-2)
 * ════════════════════════════════════════════════════════════ */

/* 제네릭 센서 슬롯 */
typedef struct {
    NsTimestamp stamp;
    uint64_t    seq;
    void*       data_ptr;   /* 외부 버퍼 포인터 */
    uint32_t    data_size;
} SensorSlot;

#define SENSOR_BUF_SIZE 16   /* power-of-2 */
#define SENSOR_BUF_MASK (SENSOR_BUF_SIZE - 1)

typedef struct {
    SensorSlot slots[SENSOR_BUF_SIZE];
    uint32_t   head;
    uint64_t   seq_counter;
} SensorBuffer;

static inline void sbuf_init(SensorBuffer* sb) {
    memset(sb, 0, sizeof(*sb));
}

static inline void sbuf_add(
    SensorBuffer* sb,
    NsTimestamp   stamp,
    void*         data_ptr,
    uint32_t      data_size)
{
    uint32_t idx = sb->head & SENSOR_BUF_MASK;
    sb->slots[idx].stamp     = stamp;
    sb->slots[idx].seq       = ++sb->seq_counter;
    sb->slots[idx].data_ptr  = data_ptr;
    sb->slots[idx].data_size = data_size;
    sb->head++;
}

/* max_dt_ns 이내 ref_stamp와 가장 가까운 슬롯 반환 */
static inline const SensorSlot* sbuf_nearest(
    const SensorBuffer* sb,
    NsTimestamp         ref_stamp,
    uint64_t            max_dt_ns)
{
    uint32_t count = sb->head < SENSOR_BUF_SIZE
                   ? sb->head : SENSOR_BUF_SIZE;
    if (count == 0) return NULL;

    const SensorSlot* best = NULL;
    uint64_t best_dt = max_dt_ns + 1;

    for (uint32_t k = 0; k < count; ++k) {
        const SensorSlot* s =
            &sb->slots[(sb->head - 1 - k) & SENSOR_BUF_MASK];
        uint64_t dt = (s->stamp > ref_stamp)
                    ? s->stamp - ref_stamp
                    : ref_stamp - s->stamp;
        if (dt < best_dt) {
            best_dt = dt;
            best    = s;
        }
    }
    return (best_dt <= max_dt_ns) ? best : NULL;
}

/* ════════════════════════════════════════════════════════════
 * ApproxTimeSync
 * Fix #9 — 다중 센서 타임스탬프 동기화
 *
 * 3개 토픽 (LiDAR 10Hz / Camera 30Hz / IMU 200Hz) 을
 * 슬라이딩 윈도우로 매칭.
 * max_dt_ns 이내 샘플이 모두 있어야 synced = 1.
 * ════════════════════════════════════════════════════════════ */

#define SYNC_MAX_TOPICS 4

typedef struct {
    SensorBuffer bufs[SYNC_MAX_TOPICS];
    uint32_t     n_topics;
    uint64_t     max_dt_ns;   /* 기본: 50ms = 50,000,000 ns */
} ApproxTimeSync;

static inline void ats_init(ApproxTimeSync* ats, uint32_t n_topics,
                             uint64_t max_dt_ns)
{
    assert(n_topics <= SYNC_MAX_TOPICS);
    memset(ats, 0, sizeof(*ats));
    ats->n_topics  = n_topics;
    ats->max_dt_ns = max_dt_ns;
}

static inline void ats_add(ApproxTimeSync* ats, uint32_t topic_id,
                            NsTimestamp stamp, void* data, uint32_t sz)
{
    assert(topic_id < ats->n_topics);
    sbuf_add(&ats->bufs[topic_id], stamp, data, sz);
}

/* 모든 토픽이 동기화되면 1, ref_stamp에 가장 가까운 셋 반환 */
static inline int ats_get_synced(
    const ApproxTimeSync* ats,
    NsTimestamp ref_stamp,
    const SensorSlot* out[SYNC_MAX_TOPICS])
{
    for (uint32_t i = 0; i < ats->n_topics; ++i) {
        out[i] = sbuf_nearest(&ats->bufs[i], ref_stamp, ats->max_dt_ns);
        if (!out[i]) return 0;
    }
    return 1;
}

#ifdef __cplusplus
}
#endif
