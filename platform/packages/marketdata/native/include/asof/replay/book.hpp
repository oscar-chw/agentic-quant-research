// Price-level order book: the C++ port of replay.book.Book.
//
// The venue's price_change messages carry the AGGREGATE size resting at a price
// level, not a delta. Applying one is therefore a replace, and size 0 is a delete.
// Treating it as an add would accumulate error silently over hours of replay.
#pragma once

#include <compare>
#include <cstdint>
#include <span>
#include <vector>

namespace asof::replay {

// The Python reference stores sides as the ints 0 and 1; the enum keeps those
// values so exported workloads and golden files are byte-compatible.
enum class Side : std::uint8_t { Bid = 0, Ask = 1 };

// One aggregated level. Prices are integer ticks on a 1/10,000 grid and sizes
// are integer hundredths, so every comparison here is exact: there is no float
// anywhere in the book and parity with Python needs no tolerance.
struct Level {
    Side side{};
    std::int32_t tick{};
    std::int64_t size_e2{};

    // Member order makes this (side, tick, size), the order Python's
    // sorted(book.multiset()) produces.
    friend constexpr auto operator<=>(const Level&, const Level&) = default;
};

class Book {
public:
    Book() = default;

    // Replace the aggregate at (side, tick); size 0 removes the level.
    void apply(Side side, std::int32_t tick, std::int64_t size_e2);

    // Adopt a keyframe snapshot wholesale. Mirrors the Python dict build: on a
    // duplicate (side, tick) the LAST row wins, and a size-0 row is kept as a
    // level, because a snapshot is stored data, not a delete instruction.
    void replace_from(std::span<const Level> snapshot);

    // Levels sorted by (side, tick), at most one per key.
    [[nodiscard]] std::span<const Level> levels() const noexcept { return levels_; }

    friend bool operator==(const Book&, const Book&) = default;

private:
    // A sorted contiguous vector instead of a hash map: real books hold tens of
    // levels, so a binary search plus a short memmove beats hashing, a snapshot
    // copy is one allocation, and the sorted output the queries return is free.
    std::vector<Level> levels_;
};

}  // namespace asof::replay
