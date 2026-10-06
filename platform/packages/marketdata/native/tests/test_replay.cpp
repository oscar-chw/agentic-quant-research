// Reconstruction contract, case by case: scalar reconstruction, replay across a
// reconnect, and the batch fixture (the same fixtures replay.book is tested on).
// Every case runs through BOTH the scalar and the batch path.
#include "check.hpp"
#include "support.hpp"

#include <optional>
#include <random>
#include <stdexcept>
#include <string>

using namespace asof::replay;
using Levels = std::vector<Level>;
using Frames = std::vector<Keyframe>;
using Deltas = std::vector<Delta>;
using Coverage = std::vector<Interval>;
using Queries = std::vector<std::int64_t>;

namespace {

constexpr Delta d(std::int64_t ts, Side side, std::int32_t tick, std::int64_t size) { return {ts, side, tick, size}; }
constexpr auto B = Side::Bid;
constexpr auto A = Side::Ask;
constexpr std::nullopt_t open_end = std::nullopt;

std::vector<Levels> answers(const std::vector<Book>& books) {
    std::vector<Levels> out;
    for (const Book& b : books) out.push_back(levels_of(b));
    return out;
}

std::vector<Book> scalar_all(const Frames& f, const Deltas& ds, const Queries& qs, const Coverage& c) {
    std::vector<Book> out;
    for (const auto q : qs) out.push_back(reconstruct(f, ds, q, c));
    return out;
}

// Both paths, one answer: the batch result must equal the scalar loop.
std::vector<Levels> both(const Frames& f, const Deltas& ds, const Queries& qs, const Coverage& c) {
    const auto batch = answers(reconstruct_many(f, ds, qs, c));
    CHECK_EQ(batch.size(), qs.size());
    const auto scalar = answers(scalar_all(f, ds, qs, c));
    for (std::size_t i = 0; i < qs.size() && i < batch.size(); ++i) CHECK_EQ(batch[i], scalar[i]);
    return batch;
}

void refused_by_both(const Frames& f, const Deltas& ds, const Queries& qs, const Coverage& c,
                     const std::string& message) {
    CHECK_THROWS(scalar_all(f, ds, qs, c), NotCovered, message);
    CHECK_THROWS(reconstruct_many(f, ds, qs, c), NotCovered, message);
}

// The batch fixture: two keyframes share ts 30 (the later one
// wins), deltas tie with keyframes, delete and re-add at one timestamp, and a
// far-future row that no query may see.
struct Fixture {
    Frames frames{{10, {bid(100, 10), ask(110, 20)}}, {30, {bid(101, 12)}}, {30, {bid(102, 13), ask(112, 14)}}};
    Deltas deltas{d(10, B, 100, 999), d(15, B, 100, 0), d(15, B, 100, 11), d(20, A, 110, 0),
                  d(30, B, 102, 999), d(35, A, 113, 7), d(100, B, 999, 999)};
    Coverage coverage{{10, open_end}};
};

}  // namespace

// --- scalar reconstruction ----------------------------------------------
namespace {
const Frames kf{{1000, {bid(5000, 100)}}};
const Deltas kd{d(1500, B, 5000, 200), d(2000, A, 5100, 300)};
const Coverage kc{{0, 5000}};
}  // namespace

TEST_CASE("rebuilds the book at a point in time") {
    CHECK_EQ(both(kf, kd, {2000}, kc)[0], (Levels{bid(5000, 200), ask(5100, 300)}));
}

TEST_CASE("deltas after the asked time are not applied") {
    CHECK_EQ(both(kf, kd, {1600}, kc)[0], (Levels{bid(5000, 200)}));
}

TEST_CASE("refuses inside a coverage gap, answers on its inclusive edge") {
    refused_by_both(kf, kd, {9000}, kc, "no coverage interval contains 9000");
    CHECK_EQ(both(kf, kd, {5000}, kc)[0], (Levels{bid(5000, 200), ask(5100, 300)}));
}

TEST_CASE("refuses when covered but no keyframe precedes the query") {
    refused_by_both({{3000, {}}}, kd, {2000}, kc, "covered at 2000 but no keyframe in coverage epoch starting 0");
}

TEST_CASE("an open interval covers the present") {
    CHECK_EQ(both(kf, kd, {4000}, {{0, open_end}})[0], (Levels{bid(5000, 200), ask(5100, 300)}));
}

TEST_CASE("a known-quiet empty book is distinguishable from no coverage") {
    CHECK_EQ(both({{10, {}}}, {}, {2000}, {{0, 5000}})[0], Levels{});
    refused_by_both({{10, {}}}, {}, {2000}, {}, "no coverage interval contains 2000");
}

// --- replay across a reconnect ------------------------------------------
TEST_CASE("a pre-gap base is refused at and after a reconnect") {
    for (const std::int64_t at : {3000, 3500, 4000}) {
        refused_by_both({{1000, {bid(4900, 100)}}}, {d(3200, A, 5100, 200)}, {at}, {{1000, 2000}, {3000, open_end}},
                        "covered at " + std::to_string(at) + " but no keyframe in coverage epoch starting 3000");
    }
}

TEST_CASE("a fresh base recovers without pre-gap or future levels") {
    const Frames frames{{1000, {bid(4000, 999)}}, {3100, {bid(4900, 100)}}, {5000, {ask(9000, 999)}}};
    const Deltas deltas{d(2900, B, 4500, 999), d(3100, B, 4900, 999), d(3200, A, 5100, 200),
                        d(3200, A, 5100, 0),   d(3200, A, 5100, 300), d(4000, B, 4600, 999)};
    CHECK_EQ(both(frames, deltas, {3500}, {{1000, 2000}, {3000, open_end}})[0],
             (Levels{bid(4900, 100), ask(5100, 300)}));
}

TEST_CASE("overlapping epochs use the latest start conservatively") {
    refused_by_both({{1000, {}}}, {}, {3500}, {{0, open_end}, {3000, 4000}},
                    "covered at 3500 but no keyframe in coverage epoch starting 3000");
}

TEST_CASE("exact epoch boundaries and an empty fresh snapshot") {
    const Coverage two{{1000, 2000}, {3000, 4000}};
    CHECK_EQ(both({{3000, {}}}, {}, {3000}, two)[0], Levels{});
    CHECK_EQ(both({{1000, {}}}, {}, {2000}, {{1000, 2000}, {3000, open_end}})[0], Levels{});
    refused_by_both({{1000, {}}}, {}, {2001}, {{1000, 2000}, {3000, open_end}}, "no coverage interval contains 2001");
}

// --- batch replay -------------------------------------------------------
TEST_CASE("unsorted and repeated queries answer in request order") {
    const Fixture fx;
    const auto got = both(fx.frames, fx.deltas, {35, 10, 15, 30, 20, 15, 10}, fx.coverage);
    CHECK_EQ(got[1], got[6]);
    CHECK_EQ(got[2], got[5]);
}

TEST_CASE("ties, deletions, future rows and the last keyframe at a timestamp") {
    const Fixture fx;
    const auto got = both(fx.frames, fx.deltas, {10, 15, 20, 30, 35}, fx.coverage);
    CHECK_EQ(got[0], (Levels{bid(100, 10), ask(110, 20)}));  // equal-time delta skipped
    CHECK_EQ(got[1], (Levels{bid(100, 11), ask(110, 20)}));  // delete then re-add
    CHECK_EQ(got[2], (Levels{bid(100, 11)}));
    CHECK_EQ(got[3], (Levels{bid(102, 13), ask(112, 14)}));  // second ts-30 keyframe wins
    CHECK_EQ(got[4], (Levels{bid(102, 13), ask(112, 14), ask(113, 7)}));  // no ts-100 row
}

TEST_CASE("results are independent books") {
    const Fixture fx;
    auto books = reconstruct_many(fx.frames, fx.deltas, Queries{15, 35, 15}, fx.coverage);
    books[0].apply(Side::Bid, 100, 999);
    books[1].replace_from({});
    CHECK_EQ(levels_of(books[2]), (Levels{bid(100, 11), ask(110, 20)}));
    CHECK_EQ(fx.frames[0].levels[0].size_e2, 10);
}

TEST_CASE("empty queries never inspect the history") {
    const Fixture fx;
    const Frames reversed(fx.frames.rbegin(), fx.frames.rend());
    CHECK(reconstruct_many(reversed, {}, Queries{}, {}).empty());
}

TEST_CASE("an empty valid snapshot and size-0 snapshot levels are preserved") {
    const auto got = both({{1, {}}}, {d(2, B, 100, 4)}, {1, 2, 1}, {{1, open_end}});
    CHECK_EQ(got, (std::vector<Levels>{{}, {bid(100, 4)}, {}}));
    CHECK_EQ(both({{1, {bid(100, 0)}}}, {}, {1}, {{1, 1}})[0], (Levels{bid(100, 0)}));
}

TEST_CASE("refusal is the first one in REQUESTED order, with the reference message") {
    const Fixture fx;
    refused_by_both({}, {}, {20}, {{10, open_end}}, "covered at 20 but no keyframe in coverage epoch starting 10");
    refused_by_both(fx.frames, fx.deltas, {5, 1}, fx.coverage, "no coverage interval contains 5");
    refused_by_both(fx.frames, fx.deltas, {20, 1000, 0}, {{10, 40}}, "no coverage interval contains 1000");
    refused_by_both(fx.frames, fx.deltas, {20}, {}, "no coverage interval contains 20");
    refused_by_both({{30, {}}}, {}, {20}, {{10, open_end}}, "covered at 20 but no keyframe in coverage epoch starting 10");
}

TEST_CASE("a reconnect needs a fresh base; endpoints are inclusive") {
    const Fixture fx;
    const Coverage gap{{10, 20}, {25, 40}};
    CHECK_THROWS(reconstruct_many(fx.frames, fx.deltas, Queries{25, 20}, gap), NotCovered,
                 "covered at 25 but no keyframe in coverage epoch starting 25");
    CHECK_THROWS(reconstruct_many(fx.frames, fx.deltas, Queries{40, 25}, gap), NotCovered,
                 "covered at 25 but no keyframe in coverage epoch starting 25");
    CHECK_THROWS(reconstruct_many(fx.frames, fx.deltas, Queries{21}, gap), NotCovered,
                 "no coverage interval contains 21");
    both(fx.frames, fx.deltas, {40, 20, 30, 10}, gap);
}

TEST_CASE("the latest overlapping epoch governs until it expires") {
    const Fixture fx;
    const Coverage overlap{{0, open_end}, {25, 35}};
    CHECK_THROWS(reconstruct_many(fx.frames, fx.deltas, Queries{25}, overlap), NotCovered,
                 "covered at 25 but no keyframe in coverage epoch starting 25");
    both(fx.frames, fx.deltas, {35, 20, 36, 30, 100}, overlap);
    CHECK_EQ(both({{10, {bid(1, 1)}}}, {}, {36, 20}, overlap), (std::vector<Levels>{{bid(1, 1)}, {bid(1, 1)}}));
}

TEST_CASE("the batch path rejects unsorted histories; the scalar path tolerates them") {
    const Fixture fx;
    const Frames frames_rev(fx.frames.rbegin(), fx.frames.rend());
    const Deltas deltas_rev(fx.deltas.rbegin(), fx.deltas.rend());
    CHECK_THROWS(reconstruct_many(frames_rev, fx.deltas, Queries{35}, fx.coverage), std::invalid_argument,
                 "keyframes must be sorted by nondecreasing timestamp");
    CHECK_THROWS(reconstruct_many(fx.frames, deltas_rev, Queries{35}, fx.coverage), std::invalid_argument,
                 "deltas must be sorted by nondecreasing timestamp");
    // The rescan filters every row but applies them in STORED order, so the
    // reversed ts-15 pair (set 11, then delete) now ends deleted; Python agrees.
    CHECK_EQ(levels_of(reconstruct(fx.frames, deltas_rev, 25, fx.coverage)), Levels{});
    CHECK_EQ(levels_of(reconstruct(fx.frames, fx.deltas, 25, fx.coverage)), (Levels{bid(100, 11)}));
}

TEST_CASE("random histories: batch equals the scalar loop (port of the Python property)") {
    for (unsigned seed = 0; seed < 12; ++seed) {
        std::mt19937_64 rng(seed);
        const auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi - 1)(rng); };
        Frames frames;
        Deltas deltas;
        for (int i = 0; i < 100; ++i) {
            if (i % 13 == 0) {
                Levels snap;
                for (int side = 0; side < 2; ++side)
                    for (int j = 0; j < 8; ++j) snap.push_back({static_cast<Side>(side), 100 + side * 20 + j, pick(0, 9)});
                frames.push_back({i, snap});
            }
            for (int k = 0; k < 3; ++k) deltas.push_back(d(i, static_cast<Side>(pick(0, 2)), pick(100, 130), pick(0, 9)));
        }
        Queries queries;
        for (int k = 0; k < 80; ++k) queries.push_back(pick(0, 100));
        queries.insert(queries.end(), {0, 99, 0});
        both(frames, deltas, queries, {{0, open_end}});
    }
}
