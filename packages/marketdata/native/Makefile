# Plain Makefile: cmake is not assumed. Targets:
#   make test     C++ unit + golden differential tests (no Python needed)
#   make parity   C++ vs Python reference on the 100,000-message workload
#   make bench    Python scalar and batch, C++ scalar and batch, and C++ through the
#                 module, in one session -> a new bench/results-<date>[-N].json
#   make golden   re-record tests/golden from the Python reference
#   make ext      the pybind11 module -> build/python/ (needs pybind11 in PYTHON)
#   make pytest   the module vs the Python reference (fixtures + hypothesis)
#
# REF is the directory holding the Python reference package replay/ (book.py,
# feed.py). The default is packages/marketdata, this folder's parent.

CXX      = clang++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS = -Iinclude
PYTHON  ?= python3.11
REF     ?= ..

BUILD    = build
LIB_SRC  = src/book.cpp src/replay.cpp src/workload.cpp
LIB_OBJ  = $(LIB_SRC:src/%.cpp=$(BUILD)/obj/%.o)
TEST_SRC = tests/main.cpp tests/test_book.cpp tests/test_replay.cpp tests/test_golden.cpp
HEADERS  = $(wildcard include/asof/replay/*.hpp)

.PHONY: all test parity bench golden ext pytest clean

all: $(BUILD)/replay_tests $(BUILD)/replay_cli $(BUILD)/replay_bench

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/libreplay.a: $(LIB_OBJ)
	ar rcs $@ $^

$(BUILD)/replay_tests: $(TEST_SRC) tests/check.hpp tests/support.hpp $(BUILD)/libreplay.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DGOLDEN_FILE='"tests/golden/reference_cases.txt"' $(TEST_SRC) $(BUILD)/libreplay.a -o $@

$(BUILD)/replay_cli: tools/replay_cli.cpp $(BUILD)/libreplay.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libreplay.a -o $@

$(BUILD)/replay_bench: bench/replay_bench.cpp $(BUILD)/libreplay.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libreplay.a -o $@

test: $(BUILD)/replay_tests
	./$(BUILD)/replay_tests

parity: $(BUILD)/replay_cli
	$(PYTHON) python/parity.py --ref $(REF) --cli $(BUILD)/replay_cli

EXPORT = $(PYTHON) python/export_workload.py --ref $(REF) --workload $(BUILD)/workload.txt --expected $(BUILD)/expected.json

# The extension is compiled from the library sources with CXXFLAGS plus EXT_FLAGS; pybind11
# and Python headers are -isystem so -Werror applies to our code only. -DNDEBUG
# makes it a release build: without it pybind11 checks the GIL on every refcount
# change, a visible slowdown with 399,600 delta rows per call.
# The suffix carries the interpreter's ABI tag, so a mismatched Python cannot load it.
PYINCS := $(shell $(PYTHON) -c "import pybind11, sysconfig; print(pybind11.get_include(), sysconfig.get_paths()['include'])" 2>/dev/null)
EXT    := $(BUILD)/python/asof_replay_native$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
EXT_FLAGS = -DNDEBUG -fPIC -fvisibility=hidden -shared
EXT_LD := $(if $(filter Darwin,$(shell uname -s)),-undefined dynamic_lookup)

ext: $(EXT)

$(EXT): bindings/py_replay.cpp src/book.cpp src/replay.cpp $(HEADERS) Makefile
	@$(PYTHON) -c "import pybind11" || { echo "ext: $(PYTHON) has no pybind11" >&2; exit 1; }
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(addprefix -isystem ,$(PYINCS)) $(CXXFLAGS) $(EXT_FLAGS) $(EXT_LD) \
		bindings/py_replay.cpp src/book.cpp src/replay.cpp -o $@

pytest: $(EXT)
	ASOF_REQUIRE_NATIVE=1 PYTHONPATH=$(BUILD)/python:$(REF):python $(PYTHON) -m pytest -q -p no:cacheprovider -rs tests/test_binding.py

# After EXT is defined: a prerequisite is expanded when the rule is read.
bench: $(BUILD)/replay_bench $(EXT)
	$(EXPORT)
	$(PYTHON) bench/run_bench.py --ref $(REF) --python $(PYTHON) --cxx $(CXX) --cxxflags "$(CXXFLAGS)" \
		--ext-cxxflags "$(CXXFLAGS) $(EXT_FLAGS) $(EXT_LD)"

golden:
	$(PYTHON) python/make_golden.py --ref $(REF) --out tests/golden/reference_cases.txt

clean:
	rm -rf $(BUILD)
