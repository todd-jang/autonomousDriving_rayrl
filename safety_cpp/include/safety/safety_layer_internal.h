/* safety_layer_internal.h — test용 내부 구조체 노출 */
#pragma once
#include "safety/types.h"
#include "safety/sync.h"
typedef struct SafetyState {
    double kl_enter, kl_exit;
    int    kl_restricted;
    int    estop;
    double max_speed_ms, ttc_thresh_s, ttc_ego_width_m, ttc_max_range_m;
    double max_speed_restricted_ms;
    ViolRing viol_log;
    uint64_t step_count;
} SafetyState;
void safety_init(SafetyState* st);
SafeCmd safety_apply(SafetyState*, float, float, const VehicleState*,
    const EgoPointCloud*, const PerceptionResult*, double);
void safety_trigger_estop(SafetyState*);
void safety_release_estop(SafetyState*);
void pc_world_to_ego(EgoPointCloud*, const float*, const float*,
    const float*, const float*, uint32_t, const VehicleState*);
