// Differential test against outcomes recorded from the Python reference
// (python/make_golden.py): fixed fixtures plus randomized histories with gaps,
// overlapping epochs, shared keyframe timestamps and unsorted input.
#include "check.hpp"
#include "support.hpp"

#include "asof/replay/workload.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#ifndef GOLDEN_FILE
#error "GOLDEN_FILE must name the recorded reference outcomes"
#endif

using namespace asof::replay;

namespace {

// Runs fn and reports what happened in the reference's vocabulary, so the
// comparison is one equality instead of a matrix of exception types.
template <class Fn>
Outcome observe(Fn&& fn) {
    Outcome outcome;
    try {
        outcome.books = fn();
    } catch (const NotCovered& error) {
        outcome.error_kind = "NotCovered";
        outcome.message = error.what();
    } catch (const std::invalid_argument& error) {
        outcome.error_kind = "ValueError";
        outcome.message = error.what();
    }
    return outcome;
}

std::vector<std::vector<Level>> answers(const std::vector<Book>& books) {
    std::vector<std::vector<Level>> out;
    for (const Book& b : books) out.push_back(levels_of(b));
    return out;
}

std::string describe(const Outcome& o) {
    if (!o.error_kind.empty()) return o.error_kind + ": " + o.message;
    std::string text = std::to_string(o.books.size()) + " books";
    for (const auto& book : o.books) text += " " + check::show(book);
    return text;
}

}  // namespace

TEST_CASE("golden: C++ matches every outcome recorded from the Python reference") {
    std::ifstream in(GOLDEN_FILE);
    CHECK(in.good());
    const auto cases = load_cases(in);
    // Absence policy: a missing or truncated golden file must fail, not pass
    // with nothing compared. 247 cases are recorded (7 fixed + 240 random).
    CHECK_EQ(cases.size(), std::size_t{247});

    std::size_t compared_books = 0;
    for (const Case& c : cases) {
        CHECK(c.batch.has_value());
        CHECK_EQ(c.scalar.size(), c.queries.size());
        if (!c.batch || c.scalar.size() != c.queries.size()) continue;

        const auto batch = observe([&] { return answers(reconstruct_many(c.keyframes, c.deltas, c.queries, c.coverage)); });
        if (describe(batch) != describe(*c.batch)) {
            check::fail(__FILE__, __LINE__, c.name + " batch: got " + describe(batch) + ", reference " + describe(*c.batch));
        }
        compared_books += batch.books.size();

        for (std::size_t i = 0; i < c.queries.size(); ++i) {
            const auto scalar = observe([&] {
                return std::vector<std::vector<Level>>{levels_of(reconstruct(c.keyframes, c.deltas, c.queries[i], c.coverage))};
            });
            if (describe(scalar) != describe(c.scalar[i])) {
                check::fail(__FILE__, __LINE__, c.name + " scalar q=" + std::to_string(c.queries[i]) + ": got " +
                                                    describe(scalar) + ", reference " + describe(c.scalar[i]));
            }
        }
    }
    // Guards against a generator change that turns every case into a refusal.
    CHECK(compared_books >= 500);
}
