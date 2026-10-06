# One install path for the monorepo. `make check` needs no install at all: it
# runs every package from this source tree (scripts/env.sh sets PYTHONPATH).
PYTHON ?= python3
PACKAGES = packages/marketdata packages/research packages/vault packages/factor packages/imc-sim apps/quantos

.PHONY: deps install check demo native-build native-test native-bench
deps:     ## test/demo dependencies from PyPI (network)
	$(PYTHON) -m pip install -r requirements.txt
install:  ## editable install of every package, offline, no dependency resolution
	$(PYTHON) -m pip install --no-index --no-build-isolation --no-deps $(foreach p,$(PACKAGES),-e $(p))
check:   ## interpreter: $$PYTHON, else $$PORTFOLIO_VENV, else ./.venv (scripts/env.sh)
	bash scripts/check.sh
demo:
	bash scripts/demo.sh
# Optional C++ hot path (packages/marketdata/native): clang++ and pybind11. The module
# is built for the interpreter scripts/env.sh picks ($PYTHON if given, else $PORTFOLIO_VENV, else .venv).
NATIVE = bash -c 'source scripts/env.sh && $(MAKE) -C packages/marketdata/native "$$@" \
         PYTHON="$$("$$PY" -c "import sys; print(sys.executable)")"' native
native-build:  ## the C++ library, tools and the pybind11 module
	$(NATIVE) all ext
native-test:   ## C++ tests, C++ vs Python on 100,000 messages, module vs Python (pytest)
	$(NATIVE) test parity pytest
native-bench:  ## Python scalar/batch, C++ scalar/batch, C++ via Python, one session -> a new native/bench/results-<date>[-N].json
	$(NATIVE) bench
