// Parity driver: replay an exported workload and write the canonical JSON the
// Python reference writes, so the two outputs can be compared byte for byte.
//
//   replay_cli [--scalar] <workload.txt> <out.json>
#include "asof/replay/workload.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

using namespace asof::replay;

int main(int argc, char** argv) {
    const bool scalar = argc == 4 && std::string_view(argv[1]) == "--scalar";
    if (argc != 3 + static_cast<int>(scalar)) {
        std::cerr << "usage: replay_cli [--scalar] <workload.txt> <out.json>\n";
        return 2;
    }
    try {
        std::ifstream in(argv[1 + scalar]);
        if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1 + scalar]);
        const auto cases = load_cases(in);
        if (cases.size() != 1) throw std::runtime_error("expected exactly one case in the workload");
        const Case& c = cases.front();

        std::vector<Book> books;
        if (scalar) {
            for (const std::int64_t q : c.queries) books.push_back(reconstruct(c.keyframes, c.deltas, q, c.coverage));
        } else {
            books = reconstruct_many(c.keyframes, c.deltas, c.queries, c.coverage);
        }
        std::ofstream out(argv[2 + scalar], std::ios::binary);
        write_books_json(out, c.queries, books);
        if (!out) throw std::runtime_error("write failed");
    } catch (const std::exception& error) {
        std::cerr << "replay_cli: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
