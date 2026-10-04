// Timing harness for the C++ port, on the same workload and timing scope as the
// Python harness (bench/py_bench.py):
//
//   timed:    API validation + preparation, the replay itself, 64 independent
//             Books, and every sorted output level copied out of them;
//   excluded: loading the workload, and checking the answer, for every
//             implementation alike.
//
//   replay_bench --mode batch|scalar --reps N <workload.txt> <expected.json>
//
// Prints one JSON object with the raw per-repetition nanoseconds, the unit the
// Python harness reads from time.perf_counter_ns.
#include "asof/replay/workload.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace asof::replay;
using Answer = std::vector<std::vector<Level>>;

namespace {

std::string slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Mirrors py_bench.py's replay(): build the Books, then materialize each
// one's sorted level list. The Books are already sorted, so this is a copy,
// but the copy is in the Python scope and stays in ours.
Answer replay(const Case& c, bool scalar, std::vector<Book>& books) {
    if (scalar) {
        books.clear();
        for (const std::int64_t q : c.queries) books.push_back(reconstruct(c.keyframes, c.deltas, q, c.coverage));
    } else {
        books = reconstruct_many(c.keyframes, c.deltas, c.queries, c.coverage);
    }
    Answer result;
    result.reserve(books.size());
    for (const Book& b : books) result.emplace_back(b.levels().begin(), b.levels().end());
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7 || std::string_view(argv[1]) != "--mode" || std::string_view(argv[3]) != "--reps") {
        std::cerr << "usage: replay_bench --mode batch|scalar --reps N <workload.txt> <expected.json>\n";
        return 2;
    }
    const std::string_view mode = argv[2];
    if (mode != "batch" && mode != "scalar") {
        std::cerr << "mode must be batch or scalar\n";
        return 2;
    }
    try {
        const int reps = std::stoi(argv[4]);
        std::istringstream workload(slurp(argv[5]));
        const auto cases = load_cases(workload);
        if (cases.size() != 1) throw std::runtime_error("expected exactly one case in the workload");
        const Case& c = cases.front();
        const std::string expected = slurp(argv[6]);

        std::vector<std::int64_t> nanoseconds;
        std::size_t output_levels = 0;
        // Declared outside the loop so each repetition frees the previous one's
        // books and level lists INSIDE the timed region, as py_bench.py's
        // rebinding `books, result = replay()` does.
        std::vector<Book> books;
        Answer result;
        for (int rep = 0; rep < reps; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            result = replay(c, mode == "scalar", books);
            const auto stop = std::chrono::steady_clock::now();
            nanoseconds.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

            // Every timed repetition must also be a correct one; a fast wrong
            // answer is not a benchmark result.
            std::ostringstream json;
            write_books_json(json, c.queries, books);
            if (json.str() != expected) throw std::runtime_error("output differs from the Python reference");
            output_levels = 0;
            for (const auto& levels : result) output_levels += levels.size();
        }

        std::cout << "{\"impl\":\"cpp\",\"mode\":\"" << mode << "\",\"queries\":" << c.queries.size()
                  << ",\"delta_rows\":" << c.deltas.size() << ",\"output_levels\":" << output_levels
                  << ",\"nanoseconds\":[";
        for (std::size_t i = 0; i < nanoseconds.size(); ++i) std::cout << (i ? "," : "") << nanoseconds[i];
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << "replay_bench: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
