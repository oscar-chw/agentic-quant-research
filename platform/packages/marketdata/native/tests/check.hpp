// A deliberately tiny test harness: no downloads allowed, and the suite needs
// only registration, a few assertions and a non-zero exit on failure.
#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct Test {
    const char* name;
    std::function<void()> body;
};

inline std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void fail(const char* file, int line, const std::string& what) {
    ++failures();
    std::cerr << file << ':' << line << ": FAIL " << what << '\n';
}

// Printing both sides of a failed comparison is the whole point of CHECK_EQ;
// a bare "false" sends the reader to a debugger.
template <class T>
std::string show(const T& value) {
    if constexpr (requires(std::ostream& os) { os << value; }) {
        std::ostringstream out;
        out << value;
        return out.str();
    } else {
        return "<unprintable>";
    }
}

}  // namespace check

#define CHECK_CONCAT_(a, b) a##b
#define CHECK_CONCAT(a, b) CHECK_CONCAT_(a, b)

#define TEST_CASE(name)                                                               \
    static void CHECK_CONCAT(test_body_, __LINE__)();                                 \
    static const check::Registrar CHECK_CONCAT(test_reg_, __LINE__){name,            \
                                                                    CHECK_CONCAT(test_body_, __LINE__)}; \
    static void CHECK_CONCAT(test_body_, __LINE__)()

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) check::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_EQ(actual, expected)                                                              \
    do {                                                                                        \
        /* By value: (actual) is often an element of a temporary container. */                \
        const auto check_a_ = (actual);                                                         \
        const auto check_e_ = (expected);                                                       \
        if (!(check_a_ == check_e_))                                                            \
            check::fail(__FILE__, __LINE__,                                                     \
                        "CHECK_EQ(" #actual ", " #expected "): " + check::show(check_a_) + " != " + \
                            check::show(check_e_));                                             \
    } while (0)

// Asserts the exception TYPE and, when non-empty, its exact message: refusals
// carry the same text as the Python reference, and that text is the contract.
#define CHECK_THROWS(expr, Type, message)                                                       \
    do {                                                                                        \
        try {                                                                                   \
            (void)(expr);                                                                       \
            check::fail(__FILE__, __LINE__, "expected " #Type " from " #expr);                  \
        } catch (const Type& check_error_) {                                                    \
            const std::string check_want_ = (message);                                          \
            if (!check_want_.empty() && check_error_.what() != check_want_)                     \
                check::fail(__FILE__, __LINE__,                                                 \
                            std::string("message '") + check_error_.what() + "' != '" + check_want_ + "'"); \
        }                                                                                       \
    } while (0)
