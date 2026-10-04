// Plain-text interchange with the Python reference: workloads exported from
// the Python generator, golden expectations recorded from the Python replay,
// and canonical JSON output that is byte-identical to the Python reference's.
//
// Text, not a binary format, so a reviewer can open a golden file and read the
// case that failed. Line grammar (whitespace separated, '#' starts a comment):
//
//   case <name>
//   K <ts> <n> (<side> <tick> <size>){n}     keyframe
//   D <ts> <side> <tick> <size>              delta
//   C <start> <end | ->                      coverage interval ('-' = open)
//   Q <t> <t> ...                            query times (lines append)
//   batch books   then one   B <n> (<side> <tick> <size>){n}   per query
//   batch error <NotCovered|ValueError> <message...>
//   scalar book <n> (<side> <tick> <size>){n}                  one per query
//   scalar error NotCovered <message...>
//   end
#pragma once

#include "asof/replay/replay.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace asof::replay {

// What the Python reference did: either books (one level list per query) or
// the first error it raised, named by its Python exception class.
struct Outcome {
    std::string error_kind;  // empty when books holds the answer
    std::string message;
    std::vector<std::vector<Level>> books;
};

struct Case {
    std::string name;
    std::vector<Keyframe> keyframes;
    std::vector<Delta> deltas;
    std::vector<Interval> coverage;
    std::vector<std::int64_t> queries;
    std::optional<Outcome> batch;  // expected reconstruct_many result
    std::vector<Outcome> scalar;   // expected reconstruct result, per query
};

// Throws std::runtime_error naming the line on malformed input: a silently
// truncated golden file would make every test after it vacuous.
[[nodiscard]] std::vector<Case> load_cases(std::istream& in);

// {"books":[[[side,tick,size],...],...],"query_times_ms":[...]} with no
// spaces: the bytes json.dumps(sort_keys=True, separators=(",", ":")) writes.
void write_books_json(std::ostream& out, std::span<const std::int64_t> queries,
                      std::span<const Book> books);

}  // namespace asof::replay
