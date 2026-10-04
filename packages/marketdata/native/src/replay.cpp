#include "asof/replay/replay.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>

namespace asof::replay {
namespace {

// Start of the coverage epoch that contains at_ms, or nothing if we were not
// collecting. Overlapping records do not establish continuity across a
// reconnect, so the LATEST containing start wins: a keyframe from before the
// reconnect must not seed a book after it.
std::optional<std::int64_t> epoch_start(std::span<const Interval> coverage, std::int64_t at_ms) {
    std::optional<std::int64_t> start;
    for (const Interval& iv : coverage) {
        const bool contains = iv.start_ms <= at_ms && (!iv.end_ms || at_ms <= *iv.end_ms);
        if (contains && (!start || iv.start_ms > *start)) start = iv.start_ms;
    }
    return start;
}

[[noreturn]] void refuse_uncovered(std::int64_t at_ms) {
    throw NotCovered("no coverage interval contains " + std::to_string(at_ms));
}

[[noreturn]] void refuse_no_base(std::int64_t at_ms, std::int64_t start_ms) {
    throw NotCovered("covered at " + std::to_string(at_ms) +
                     " but no keyframe in coverage epoch starting " + std::to_string(start_ms));
}

template <class Row>
void require_sorted(std::span<const Row> rows, const char* name) {
    const auto descending = [](const Row& left, const Row& right) { return right.ts_ms < left.ts_ms; };
    if (std::adjacent_find(rows.begin(), rows.end(), descending) != rows.end()) {
        throw std::invalid_argument(std::string(name) + " must be sorted by nondecreasing timestamp");
    }
}

}  // namespace

Book reconstruct(std::span<const Keyframe> keyframes, std::span<const Delta> deltas,
                 std::int64_t at_ms, std::span<const Interval> coverage) {
    const auto start_ms = epoch_start(coverage, at_ms);
    if (!start_ms) refuse_uncovered(at_ms);

    // Same loop shape as the Python reference, including its early break, so
    // the scalar path stays a faithful oracle even on unsorted input.
    const Keyframe* base = nullptr;
    for (const Keyframe& frame : keyframes) {
        if (frame.ts_ms > at_ms) break;
        if (frame.ts_ms >= *start_ms) base = &frame;
    }
    // Covered but no baseline: replaying deltas onto an empty book would give a
    // partial book that looks complete.
    if (base == nullptr) refuse_no_base(at_ms, *start_ms);

    Book book;
    book.replace_from(base->levels);
    for (const Delta& d : deltas) {
        if (base->ts_ms < d.ts_ms && d.ts_ms <= at_ms) book.apply(d.side, d.tick, d.size_e2);
    }
    return book;
}

std::vector<Book> reconstruct_many(std::span<const Keyframe> keyframes, std::span<const Delta> deltas,
                                   std::span<const std::int64_t> at_times,
                                   std::span<const Interval> coverage) {
    // An empty request must not inspect (or reject) the history at all.
    if (at_times.empty()) return {};

    // The sweep below is only correct on sorted histories; the scalar path
    // tolerates disorder by rescanning, so the batch path must refuse it loudly
    // instead of returning a different answer.
    require_sorted(keyframes, "keyframes");
    require_sorted(deltas, "deltas");

    std::vector<std::int64_t> frame_times;
    frame_times.reserve(keyframes.size());
    for (const Keyframe& frame : keyframes) frame_times.push_back(frame.ts_ms);

    // Preflight in CALLER order, before any replay: sorting queries must not
    // change which refusal is reported, and a refusal must not leave behind
    // half a result.
    constexpr std::size_t no_base = static_cast<std::size_t>(-1);
    std::vector<std::size_t> bases;
    bases.reserve(at_times.size());
    for (const std::int64_t at_ms : at_times) {
        const auto start_ms = epoch_start(coverage, at_ms);
        if (!start_ms) refuse_uncovered(at_ms);
        // Last keyframe at or before at_ms. Equal timestamps resolve to the
        // last one, exactly as the scalar scan's "keep overwriting" does.
        const auto after = std::upper_bound(frame_times.begin(), frame_times.end(), at_ms);
        if (after == frame_times.begin() || *std::prev(after) < *start_ms) refuse_no_base(at_ms, *start_ms);
        bases.push_back(static_cast<std::size_t>(std::prev(after) - frame_times.begin()));
    }

    // Visit queries chronologically. The chosen keyframe index is monotone in
    // query time, so the delta cursor only ever moves forward: O(D) total
    // instead of O(Q*D) for the scalar loop.
    std::vector<std::size_t> order(at_times.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return at_times[a] < at_times[b]; });

    std::vector<Book> results(at_times.size());
    Book current;
    std::size_t base_index = no_base;
    std::size_t cursor = 0;
    for (const std::size_t result_index : order) {
        const std::int64_t at_ms = at_times[result_index];
        const std::size_t index = bases[result_index];
        if (index != base_index) {
            current.replace_from(keyframes[index].levels);
            base_index = index;
            // A keyframe already contains every change at its own timestamp;
            // equal-time deltas are skipped, exactly as in the scalar reader.
            const std::int64_t base_ts = keyframes[index].ts_ms;
            while (cursor < deltas.size() && deltas[cursor].ts_ms <= base_ts) ++cursor;
        }
        while (cursor < deltas.size() && deltas[cursor].ts_ms <= at_ms) {
            const Delta& d = deltas[cursor++];
            current.apply(d.side, d.tick, d.size_e2);
        }
        // A value copy: callers get independent books, so mutating one answer
        // can never leak into another (the Python test pins the same property).
        results[result_index] = current;
    }
    return results;
}

}  // namespace asof::replay
