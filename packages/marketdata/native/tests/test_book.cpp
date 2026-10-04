// Book semantics: replay.book.Book's level rules plus the snapshot rules that
// reconstruct_many relies on.
#include "check.hpp"
#include "support.hpp"

using namespace asof::replay;
using Levels = std::vector<Level>;

TEST_CASE("price change replaces the aggregate, it does not add") {
    Book b;
    b.apply(Side::Bid, 5000, 1000);
    b.apply(Side::Bid, 5000, 300);
    CHECK_EQ(levels_of(b), (Levels{bid(5000, 300)}));
}

TEST_CASE("zero size deletes the level; deleting an absent level is a no-op") {
    Book b;
    b.apply(Side::Bid, 5000, 1000);
    b.apply(Side::Bid, 5000, 0);
    CHECK(b.levels().empty());
    b.apply(Side::Ask, 5100, 0);
    CHECK(b.levels().empty());
}

TEST_CASE("levels stay sorted by (side, tick) whatever the arrival order") {
    Book b;
    b.apply(Side::Ask, 5200, 1);
    b.apply(Side::Bid, 4800, 2);
    b.apply(Side::Ask, 5100, 3);
    b.apply(Side::Bid, 4900, 4);
    b.apply(Side::Bid, 4850, 5);
    CHECK_EQ(levels_of(b), (Levels{bid(4800, 2), bid(4850, 5), bid(4900, 4), ask(5100, 3), ask(5200, 1)}));
}

TEST_CASE("the same tick on opposite sides is two independent levels") {
    Book b;
    b.apply(Side::Bid, 5000, 7);
    b.apply(Side::Ask, 5000, 9);
    b.apply(Side::Bid, 5000, 0);
    CHECK_EQ(levels_of(b), (Levels{ask(5000, 9)}));
}

TEST_CASE("updating one level leaves its neighbours untouched") {
    Book b;
    for (std::int32_t tick = 100; tick < 110; ++tick) b.apply(Side::Bid, tick, tick);
    b.apply(Side::Bid, 105, 1);
    b.apply(Side::Bid, 103, 0);
    Levels want;
    for (std::int32_t tick = 100; tick < 110; ++tick) {
        if (tick == 103) continue;
        want.push_back(bid(tick, tick == 105 ? 1 : tick));
    }
    CHECK_EQ(levels_of(b), want);
}

TEST_CASE("snapshot replaces all prior state, last duplicate wins, size-0 rows are kept") {
    Book b;
    b.apply(Side::Ask, 9000, 999);
    const Levels snapshot{bid(100, 5), ask(110, 0), bid(100, 6), bid(90, 1)};
    b.replace_from(snapshot);
    CHECK_EQ(levels_of(b), (Levels{bid(90, 1), bid(100, 6), ask(110, 0)}));
    b.replace_from({});
    CHECK(b.levels().empty());
}
