#include "asof/replay/book.hpp"

#include <algorithm>
#include <iterator>

namespace asof::replay {
namespace {

// Levels are keyed by (side, tick); size is the payload, never part of the key.
constexpr bool key_less(const Level& a, const Level& b) noexcept {
    return a.side != b.side ? a.side < b.side : a.tick < b.tick;
}

constexpr bool same_key(const Level& a, const Level& b) noexcept {
    return a.side == b.side && a.tick == b.tick;
}

}  // namespace

void Book::apply(Side side, std::int32_t tick, std::int64_t size_e2) {
    const Level probe{side, tick, size_e2};
    const auto it = std::lower_bound(levels_.begin(), levels_.end(), probe, key_less);
    const bool present = it != levels_.end() && same_key(*it, probe);
    if (size_e2 == 0) {
        if (present) levels_.erase(it);
    } else if (present) {
        it->size_e2 = size_e2;
    } else {
        levels_.insert(it, probe);
    }
}

void Book::replace_from(std::span<const Level> snapshot) {
    levels_.assign(snapshot.begin(), snapshot.end());
    // stable_sort keeps duplicates in input order, so keeping the last of each
    // run reproduces Python's "later dict assignment wins".
    std::stable_sort(levels_.begin(), levels_.end(), key_less);
    auto out = levels_.begin();
    for (auto it = levels_.begin(); it != levels_.end(); ++it) {
        const auto next = std::next(it);
        if (next != levels_.end() && same_key(*it, *next)) continue;
        *out++ = *it;
    }
    levels_.erase(out, levels_.end());
}

}  // namespace asof::replay
