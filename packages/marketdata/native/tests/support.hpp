// Test-only helpers: readable failure output and short level literals.
#pragma once

#include "asof/replay/replay.hpp"

#include <ostream>
#include <vector>

namespace asof::replay {

inline std::ostream& operator<<(std::ostream& out, const Level& lv) {
    return out << '(' << static_cast<int>(lv.side) << ',' << lv.tick << ',' << lv.size_e2 << ')';
}

inline std::ostream& operator<<(std::ostream& out, const std::vector<Level>& levels) {
    out << '{';
    for (const Level& lv : levels) out << lv;
    return out << '}';
}

inline std::vector<Level> levels_of(const Book& book) {
    return {book.levels().begin(), book.levels().end()};
}

constexpr Level bid(std::int32_t tick, std::int64_t size) { return {Side::Bid, tick, size}; }
constexpr Level ask(std::int32_t tick, std::int64_t size) { return {Side::Ask, tick, size}; }

}  // namespace asof::replay
