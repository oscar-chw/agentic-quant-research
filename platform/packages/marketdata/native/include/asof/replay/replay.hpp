// Point-in-time book reconstruction: the C++ port of replay.book.reconstruct
// (scalar reference) and replay.book.reconstruct_many (one-sweep batch replay).
//
// The contract is the archive's, not this file's: answer only where coverage
// says we were collecting, and only from a keyframe inside that coverage epoch.
// Anything else REFUSES with NotCovered rather than returning a plausible book,
// because a book built across a disconnect looks complete and is wrong.
#pragma once

#include "asof/replay/book.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace asof::replay {

// A stored server snapshot. Keyframes include all state AT their timestamp,
// so deltas with the same timestamp are never re-applied on top of one.
struct Keyframe {
    std::int64_t ts_ms{};
    std::vector<Level> levels;
};

struct Delta {
    std::int64_t ts_ms{};
    Side side{};
    std::int32_t tick{};
    std::int64_t size_e2{};
};

// An interval during which data was actually arriving. Both endpoints are
// inclusive; an empty end_ms means the interval is still open.
struct Interval {
    std::int64_t start_ms{};
    std::optional<std::int64_t> end_ms;
};

// Thrown, never returned as an empty book. The message text matches the
// Python reference byte for byte so refusals can be compared in parity tests.
class NotCovered : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Scalar reference: rebuild one book by scanning the whole history. Like the
// Python original it does not require sorted deltas (it filters every row).
[[nodiscard]] Book reconstruct(std::span<const Keyframe> keyframes,
                               std::span<const Delta> deltas,
                               std::int64_t at_ms,
                               std::span<const Interval> coverage);

// Batch replay: one forward sweep over the deltas answers every query.
// Histories must be sorted by nondecreasing timestamp (std::invalid_argument
// otherwise); queries may repeat and arrive in any order. Results come back in
// query order as independent books, and every query is preflighted in caller
// order before any replay, so the first refusal reported is the one the scalar
// loop would report and no partial output escapes.
[[nodiscard]] std::vector<Book> reconstruct_many(std::span<const Keyframe> keyframes,
                                                 std::span<const Delta> deltas,
                                                 std::span<const std::int64_t> at_times,
                                                 std::span<const Interval> coverage);

}  // namespace asof::replay
