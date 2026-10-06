#include "asof/replay/workload.hpp"

#include <charconv>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string_view>

namespace asof::replay {
namespace {

// Cursor over one line. Parsing is strict: a stray token is an error, not
// something to skip, so a hand-edited golden file cannot quietly lose a level.
class Tokens {
public:
    Tokens(std::string_view line, std::size_t line_no) : rest_(line), line_no_(line_no) {}

    std::string_view word() {
        skip_space();
        if (rest_.empty()) fail("unexpected end of line");
        const auto end = rest_.find(' ');
        const auto token = rest_.substr(0, end);
        rest_.remove_prefix(end == std::string_view::npos ? rest_.size() : end);
        return token;
    }

    template <class Int>
    Int integer() {
        const auto token = word();
        Int value{};
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || ptr != token.data() + token.size()) fail("bad integer '" + std::string(token) + "'");
        return value;
    }

    Side side() {
        const auto value = integer<int>();
        if (value != 0 && value != 1) fail("side must be 0 or 1");
        return static_cast<Side>(value);
    }

    std::vector<Level> levels() {
        const auto count = integer<std::size_t>();
        std::vector<Level> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const Side s = side();
            const auto tick = integer<std::int32_t>();
            out.push_back({s, tick, integer<std::int64_t>()});
        }
        return out;
    }

    // Error messages may contain spaces; they run to the end of the line.
    std::string remainder() {
        skip_space();
        std::string text(rest_);
        rest_ = {};
        return text;
    }

    bool at_end() {
        skip_space();
        return rest_.empty();
    }

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("workload line " + std::to_string(line_no_) + ": " + why);
    }

private:
    void skip_space() {
        while (!rest_.empty() && rest_.front() == ' ') rest_.remove_prefix(1);
    }

    std::string_view rest_;
    std::size_t line_no_;
};

Outcome parse_error(Tokens& t) {
    Outcome outcome;
    outcome.error_kind = std::string(t.word());
    outcome.message = t.remainder();
    return outcome;
}

}  // namespace

std::vector<Case> load_cases(std::istream& in) {
    std::vector<Case> cases;
    Case* current = nullptr;
    bool reading_batch_books = false;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        Tokens t(line, line_no);
        if (t.at_end() || line.front() == '#') continue;
        const auto tag = t.word();
        if (tag == "case") {
            if (current != nullptr) t.fail("case without end");
            current = &cases.emplace_back();
            current->name = t.remainder();
            reading_batch_books = false;
            continue;
        }
        if (current == nullptr) t.fail("record outside a case");
        if (tag == "K") {
            const auto ts = t.integer<std::int64_t>();
            current->keyframes.push_back({ts, t.levels()});
        } else if (tag == "D") {
            const auto ts = t.integer<std::int64_t>();
            const Side s = t.side();
            const auto tick = t.integer<std::int32_t>();
            current->deltas.push_back({ts, s, tick, t.integer<std::int64_t>()});
        } else if (tag == "C") {
            const auto start = t.integer<std::int64_t>();
            Interval iv{start, std::nullopt};
            if (t.at_end()) t.fail("coverage needs an end or '-'");
            if (Tokens peek = t; peek.word() == "-") {
                t = peek;
            } else {
                iv.end_ms = t.integer<std::int64_t>();
            }
            current->coverage.push_back(iv);
        } else if (tag == "Q") {
            while (!t.at_end()) current->queries.push_back(t.integer<std::int64_t>());
        } else if (tag == "batch") {
            const auto kind = t.word();
            if (kind == "books") {
                current->batch = Outcome{};
                reading_batch_books = true;
            } else if (kind == "error") {
                current->batch = parse_error(t);
            } else {
                t.fail("batch must be 'books' or 'error'");
            }
        } else if (tag == "B") {
            if (!reading_batch_books) t.fail("B line outside 'batch books'");
            current->batch->books.push_back(t.levels());
        } else if (tag == "scalar") {
            const auto kind = t.word();
            Outcome outcome;
            if (kind == "book") {
                outcome.books.push_back(t.levels());
            } else if (kind == "error") {
                outcome = parse_error(t);
            } else {
                t.fail("scalar must be 'book' or 'error'");
            }
            current->scalar.push_back(std::move(outcome));
        } else if (tag == "end") {
            current = nullptr;
            continue;
        } else {
            t.fail("unknown tag '" + std::string(tag) + "'");
        }
        if (!t.at_end()) t.fail("trailing tokens");
    }
    if (current != nullptr) throw std::runtime_error("workload: last case has no end");
    return cases;
}

void write_books_json(std::ostream& out, std::span<const std::int64_t> queries,
                      std::span<const Book> books) {
    out << "{\"books\":[";
    for (std::size_t b = 0; b < books.size(); ++b) {
        if (b) out << ',';
        out << '[';
        const auto levels = books[b].levels();
        for (std::size_t i = 0; i < levels.size(); ++i) {
            if (i) out << ',';
            out << '[' << static_cast<int>(levels[i].side) << ',' << levels[i].tick << ','
                << levels[i].size_e2 << ']';
        }
        out << ']';
    }
    out << "],\"query_times_ms\":[";
    for (std::size_t q = 0; q < queries.size(); ++q) {
        if (q) out << ',';
        out << queries[q];
    }
    out << "]}";
}

}  // namespace asof::replay
