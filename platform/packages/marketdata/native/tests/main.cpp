#include "check.hpp"

#include <exception>
#include <iostream>

int main() {
    const auto& tests = check::registry();
    // Zero registered tests is a broken build (a source file dropped from the
    // Makefile), not a green run.
    if (tests.empty()) {
        std::cerr << "FAIL: no tests registered\n";
        return 1;
    }
    for (const auto& test : tests) {
        const int before = check::failures();
        try {
            test.body();
        } catch (const std::exception& error) {
            check::fail(test.name, 0, std::string("uncaught exception: ") + error.what());
        }
        std::cout << (check::failures() == before ? "  ok    " : "  FAIL  ") << test.name << '\n';
    }
    std::cout << tests.size() << " tests, " << check::failures() << " failed assertions\n";
    return check::failures() == 0 ? 0 : 1;
}
