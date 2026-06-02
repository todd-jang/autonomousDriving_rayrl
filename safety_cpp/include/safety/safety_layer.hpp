// safety_cpp/include/safety/safety_layer.hpp
// ────────────────────────────────────────────
// Hard-realtime Safety Layer — C++17
//
// 세 가지 핵심 설계 원칙:
//
// 1. COORDINATE CONSISTENCY
//    모든 포인트는 함수 진입 전 ego-frame으로 변환 완료.
//    프레임 태그(FrameId enum)로 컴파일 타임 강제.
//
// 2. TIMING DETERMINISM
//    apply()는 최악 실행 시간(WCET) 보장.
//    동적 메모리 할당 없음 — 모든 버퍼 스택/고정 배열.
//    로그는 lock-free ring buffer (Non-blocking).
//
// 3. SENSOR SYNCHRONIZATION
//    StampedData<T> 래퍼: 모든 데이터에 타임스탬프 강제.
//    ApproxTimeSync: O(1) 슬라이딩 윈도우 매칭.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>
#include <atomic>
#include <chrono>

namespace safety {

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Seconds   = std::chrono::duration<double>;

// ════════════════════════════════════════════════════════════
// 1. COORDINATE CONSISTENCY — 프레임 태그 시스템
// ════════════════════════════════════════════════════════════

enum class FrameId : uint8_t {
    WORLD = 0,   // UTM or ENU world frame
    EGO   = 1,   // vehicle body frame (x=forward, y=left, z=up)
    SENSOR= 2,   // sensor-specific (LiDAR origin)
};

// 컴파일 타임 프레임 안전성:
// Vec3<FrameId::EGO> 와 Vec3<FrameId::WORLD>는 다른 타입.
// 잘못된 프레임 혼용 → 컴파일 에러.
template <FrameId F>
struct Vec3 {
    double x{0}, y{0}, z{0};
    static constexpr FrameId frame = F;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    Vec3 operator-(const Vec3& o) const {
        return {x-o.x, y-o.y, z-o.z};
    }
    Vec3 operator*(double s) const {
        return {x*s, y*s, z*s};
    }
};

using Vec3World = Vec3<FrameId::WORLD>;
using Vec3Ego   = Vec3<FrameId::EGO>;

// 쿼터니언 — 항상 [w, x, y, z]
struct Quaternion {
    double w{1}, x{0}, y{0}, z{0};

    // Fix #1: 올바른 yaw 추출 (pitch/roll 있어도 정확)
    double to_yaw() const noexcept {
        const double siny_cosp = 2.0 * (w*z + x*y);
        const double cosy_cosp = 1.0 - 2.0 * (y*y + z*z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // 3×3 회전행렬 (행 우선)
    std::array<double, 9> to_rotation_matrix() const noexcept {
        return {
            1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
              2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
              2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
        };
    }
};

// Fix #2: World → Ego 변환 (컴파일 타임 프레임 체크)
inline Vec3Ego world_to_ego(
    const Vec3World& p_world,
    const Vec3World& ego_pos,
    const Quaternion& ego_quat) noexcept
{
    const auto R = ego_quat.to_rotation_matrix();
    const double dx = p_world.x - ego_pos.x;
    const double dy = p_world.y - ego_pos.y;
    const double dz = p_world.z - ego_pos.z;
    // R^T * delta  (R^T = R^-1 for rotation matrix)
    return {
        R[0]*dx + R[3]*dy + R[6]*dz,
        R[1]*dx + R[4]*dy + R[7]*dz,
        R[2]*dx + R[5]*dy + R[8]*dz,
    };
}

// ════════════════════════════════════════════════════════════
// 2. TIMING DETERMINISM — StampedData + WCET 보장
// ════════════════════════════════════════════════════════════

// 모든 센서 데이터에 타임스탬프 강제
template <typename T>
struct StampedData {
    T         data;
    TimePoint stamp;
    uint64_t  seq{0};   // 시퀀스 번호 (드롭 감지용)

    bool is_fresh(double max_age_s) const noexcept {
        const auto age = Seconds(Clock::now() - stamp).count();
        return age <= max_age_s;
    }
    double age_s() const noexcept {
        return Seconds(Clock::now() - stamp).count();
    }
};

// Lock-free ring buffer (WCET: O(1), 동적 할당 없음)
template <typename T, std::size_t N>
class RingBuffer {
    static_assert((N & (N-1)) == 0, "N must be power of 2");
public:
    void push(const T& v) noexcept {
        buf_[head_.fetch_add(1, std::memory_order_relaxed) & MASK_] = v;
    }
    bool try_pop(T& out) noexcept {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto t = tail_.load(std::memory_order_relaxed);
        if (h == t) return false;
        out = buf_[t & MASK_];
        tail_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    std::size_t size() const noexcept {
        return head_.load() - tail_.load();
    }
private:
    static constexpr std::size_t MASK_ = N - 1;
    std::array<T, N>     buf_{};
    std::atomic<uint64_t> head_{0}, tail_{0};
};

// Safety 위반 로그 엔트리 (고정 크기, 동적 메모리 없음)
struct ViolationEntry {
    TimePoint  stamp;
    uint8_t    rule_id;      // 1=estop 2=red 3=ttc 4=speed 5=geo 6=oncoming 7=kl
    float      value;        // 관련 수치
    char       reason[32];   // 고정 문자열 버퍼
};

// ════════════════════════════════════════════════════════════
// 3. SENSOR SYNCHRONIZATION — ApproxTimeSync
// ════════════════════════════════════════════════════════════

// 고정 크기 센서 버퍼 (동적 메모리 없음)
template <typename T, std::size_t BUF = 8>
class SensorBuffer {
public:
    void add(StampedData<T> d) noexcept {
        buf_[idx_ & MASK_] = std::move(d);
        ++idx_;
    }
    // max_dt 이내 최신 샘플 반환
    std::optional<StampedData<T>> nearest(
        TimePoint ref, double max_dt_s) const noexcept
    {
        const uint64_t count = std::min(idx_, static_cast<uint64_t>(BUF));
        if (count == 0) return std::nullopt;
        double best_dt = max_dt_s + 1.0;
        int    best_i  = -1;
        for (uint64_t k = 0; k < count; ++k) {
            const auto& s = buf_[(idx_ - 1 - k) & MASK_];
            const double dt = std::abs(
                Seconds(s.stamp - ref).count()
            );
            if (dt < best_dt) { best_dt = dt; best_i = (int)k; }
        }
        if (best_i < 0 || best_dt > max_dt_s)
            return std::nullopt;
        return buf_[(idx_ - 1 - best_i) & MASK_];
    }
    bool has_data() const noexcept { return idx_ > 0; }
private:
    static constexpr uint64_t MASK_ = BUF - 1;
    std::array<StampedData<T>, BUF> buf_{};
    uint64_t idx_{0};
};

// ════════════════════════════════════════════════════════════
// KL Hysteresis (Fix #5)
// ════════════════════════════════════════════════════════════

struct KLHysteresis {
    double enter_thresh{0.12};
    double exit_thresh{0.06};

    bool update(double kl) noexcept {
        if (!restricted_ && kl > enter_thresh) restricted_ = true;
        else if (restricted_ && kl < exit_thresh)  restricted_ = false;
        return restricted_;
    }
    bool restricted() const noexcept { return restricted_; }
private:
    bool restricted_{false};
};

// ════════════════════════════════════════════════════════════
// 차량 상태 (ego frame 기준)
// ════════════════════════════════════════════════════════════

struct VehicleState {
    Vec3World  pos;
    Vec3World  vel_world;
    Quaternion quat;
    Vec3World  ang_vel;
    double     speed_ms{0};    // |vel| scalar

    double yaw()   const noexcept { return quat.to_yaw(); }
    double speed() const noexcept { return speed_ms; }
};

// ════════════════════════════════════════════════════════════
// 안전 명령 출력
// ════════════════════════════════════════════════════════════

struct SafeCmd {
    float    throttle{0};       // [-1, 1]
    float    steering{0};       // [-1, 1]
    float    brake{0};          // [ 0, 1]
    bool     estop{false};
    float    speed_kmh{0};
    uint8_t  active_rules{0};   // 비트마스크: bit0=estop ... bit6=kl
    char     reason[64]{"nominal"};
};

// ════════════════════════════════════════════════════════════
// PointCloud (ego frame 전용)
// 반드시 world_to_ego() 변환 후 사용
// ════════════════════════════════════════════════════════════

struct EgoPointCloud {
    // 고정 배열 (동적 메모리 없음, 최대 16384 포인트)
    static constexpr std::size_t MAX_PTS = 16384;
    std::array<Vec3Ego, MAX_PTS> pts{};
    std::size_t n{0};
    float       intensity[MAX_PTS]{};
    TimePoint   stamp;

    void add(Vec3Ego p, float inten) noexcept {
        if (n >= MAX_PTS) return;
        pts[n]       = p;
        intensity[n] = inten;
        ++n;
    }
};

// ════════════════════════════════════════════════════════════
// 인지 결과 (신호등 / 역주행 / 예외)
// ════════════════════════════════════════════════════════════

enum class SignalState : uint8_t {
    UNKNOWN = 0, RED = 1, YELLOW = 2, GREEN = 3
};

struct PerceptionResult {
    SignalState signal{SignalState::UNKNOWN};
    bool        oncoming{false};
    float       lane_deviation_m{0};
    float       collision_risk{0};
    float       shadow_kl{0};
    uint16_t    active_exceptions{0};   // 비트마스크
};

// ════════════════════════════════════════════════════════════
// SafetyLayer — 메인 클래스
// ════════════════════════════════════════════════════════════

struct SafetyConfig {
    double max_speed_ms{8.33};    // 30 km/h
    double ttc_thresh_s{2.0};
    double ttc_ego_width_m{1.5};  // 전방 장애물 감지 폭
    double ttc_max_range_m{40.0};
    double kl_enter{0.12};
    double kl_exit{0.06};
    double max_speed_restricted{4.17};  // KL 제한 시 15 km/h
};

class SafetyLayer {
public:
    explicit SafetyLayer(SafetyConfig cfg = {}) noexcept
        : cfg_(cfg)
    {
        kl_hyst_.enter_thresh = cfg.kl_enter;
        kl_hyst_.exit_thresh  = cfg.kl_exit;
    }

    // ── 메인 apply() — WCET 보장, 동적 할당 없음 ──────────────
    SafeCmd apply(
        float               throttle_in,
        float               steering_in,
        const VehicleState& state,
        const EgoPointCloud& pc_ego,    // 반드시 ego frame
        const PerceptionResult& perc,
        double              shadow_kl) noexcept
    {
        SafeCmd cmd;
        cmd.throttle  = throttle_in;
        cmd.steering  = steering_in;
        cmd.speed_kmh = static_cast<float>(state.speed() * 3.6);

        float brake = 0.f;
        uint8_t rules = 0;
        char reason_buf[64] = "nominal";
        int reason_len = 7;

        auto append_reason = [&](std::string_view s) {
            if (reason_len > 7) {
                reason_buf[reason_len++] = '+';
            } else {
                reason_len = 0;
            }
            for (char c : s) {
                if (reason_len < 63)
                    reason_buf[reason_len++] = c;
            }
            reason_buf[reason_len] = '\0';
        };

        // Rule 1: e-stop (최우선)
        if (estop_) {
            cmd.throttle = 0.f;
            cmd.steering = 0.f;
            cmd.brake    = 1.f;
            cmd.estop    = true;
            cmd.active_rules = 0x01;
            constexpr std::string_view r = "e-stop";
            for (char c : r) cmd.reason[(&c - r.data())] = c;
            cmd.reason[r.size()] = '\0';
            log_violation(1, 0.f, "e-stop");
            return cmd;
        }

        // Rule 2: 빨간 신호
        if (perc.signal == SignalState::RED && state.speed() > 0.5) {
            cmd.throttle = std::min(cmd.throttle, -0.4f);
            brake        = std::max(brake, 0.6f);
            rules       |= 0x02;
            append_reason("red");
            log_violation(2, static_cast<float>(state.speed()), "red_light");
        }

        // Rule 3: TTC (ego frame, Fix #2)
        const auto ttc = compute_ttc_ego(state, pc_ego);
        if (ttc.has_value() && *ttc < cfg_.ttc_thresh_s) {
            const float force = static_cast<float>(
                std::max(0.0, (cfg_.ttc_thresh_s - *ttc) / cfg_.ttc_thresh_s)
            );
            brake        = std::max(brake, force);
            cmd.throttle = std::min(cmd.throttle, 1.f - force);
            rules       |= 0x04;
            char ttc_str[16];
            // sprintf 대신 간단한 정수 변환 (동적 메모리 없음)
            const int ttc_int = static_cast<int>(*ttc * 10);
            ttc_str[0] = 't'; ttc_str[1] = 't'; ttc_str[2] = 'c';
            ttc_str[3] = '='; ttc_str[4] = '0' + ttc_int/10;
            ttc_str[5] = '.'; ttc_str[6] = '0' + ttc_int%10;
            ttc_str[7] = 's'; ttc_str[8] = '\0';
            append_reason(ttc_str);
            log_violation(3, static_cast<float>(*ttc), ttc_str);
        }

        // Rule 4: 속도 상한
        const double limit = speed_limit(perc.active_exceptions);
        if (state.speed() > limit) {
            cmd.throttle = std::min(cmd.throttle, 0.f);
            rules       |= 0x08;
            append_reason("spd_cap");
        }

        // Rule 5: Geofence (있을 때만)
        // 구현은 geofence_check() 참조

        // Rule 6: 역주행 보정
        if (perc.oncoming) {
            cmd.steering = std::min(cmd.steering + 0.4f, 1.f);
            cmd.throttle = std::min(cmd.throttle, 0.3f);
            rules       |= 0x20;
            append_reason("oncoming");
            log_violation(6, 0.f, "oncoming");
        }

        // Rule 7: KL hysteresis (Fix #5)
        if (kl_hyst_.update(shadow_kl)) {
            if (state.speed() > cfg_.max_speed_restricted) {
                cmd.throttle = std::min(cmd.throttle, 0.f);
            }
            rules |= 0x40;
            append_reason("kl_restrict");
        }

        // 최종 클리핑
        cmd.throttle    = clamp(cmd.throttle, -1.f, 1.f);
        cmd.steering    = clamp(cmd.steering, -1.f, 1.f);
        cmd.brake       = clamp(brake,         0.f, 1.f);
        cmd.active_rules = rules;
        if (reason_len > 0) {
            for (int i = 0; i < reason_len && i < 63; ++i)
                cmd.reason[i] = reason_buf[i];
            cmd.reason[reason_len] = '\0';
        }

        return cmd;
    }

    void trigger_estop()  noexcept { estop_ = true;  }
    void release_estop()  noexcept { estop_ = false; }
    bool is_estop()       const noexcept { return estop_; }

    // 위반 로그 접근
    std::size_t violation_count() const noexcept {
        return violation_buf_.size();
    }

private:
    SafetyConfig cfg_;
    KLHysteresis kl_hyst_;
    bool         estop_{false};
    RingBuffer<ViolationEntry, 256> violation_buf_;

    // TTC 계산 — ego frame 전용 (Fix #2 보장)
    std::optional<double> compute_ttc_ego(
        const VehicleState& state,
        const EgoPointCloud& pc) const noexcept
    {
        if (pc.n == 0 || state.speed() < 0.5) return std::nullopt;

        double nearest = cfg_.ttc_max_range_m + 1.0;
        for (std::size_t i = 0; i < pc.n; ++i) {
            const auto& p = pc.pts[i];
            // ego X+ = 전방 — 프레임 보장됨
            if (p.x > 0.5 && p.x < cfg_.ttc_max_range_m &&
                std::abs(p.y) < cfg_.ttc_ego_width_m)
            {
                nearest = std::min(nearest, p.x);
            }
        }
        if (nearest > cfg_.ttc_max_range_m) return std::nullopt;
        return nearest / std::max(state.speed(), 0.1);
    }

    double speed_limit(uint16_t exceptions) const noexcept {
        double limit = cfg_.max_speed_ms;
        if (exceptions & 0x0001) limit = std::min(limit, 5.56);  // rain
        if (exceptions & 0x0002) limit = std::min(limit, 4.17);  // snow
        if (exceptions & 0x0004) limit = std::min(limit, 2.78);  // fog
        if (exceptions & 0x0008) limit = std::min(limit, 2.78);  // ice
        if (exceptions & 0x0010) limit = std::min(limit, 0.0);   // flood
        if (exceptions & 0x0020) limit = std::min(limit, 0.0);   // emergency
        return limit;
    }

    void log_violation(uint8_t rule, float val, const char* reason) noexcept {
        ViolationEntry e;
        e.stamp   = Clock::now();
        e.rule_id = rule;
        e.value   = val;
        // 고정 크기 복사 (strncpy 없이)
        for (int i = 0; i < 31 && reason[i]; ++i)
            e.reason[i] = reason[i];
        e.reason[31] = '\0';
        violation_buf_.push(e);
    }

    static float clamp(float v, float lo, float hi) noexcept {
        return v < lo ? lo : v > hi ? hi : v;
    }
};

}  // namespace safety
