"""One shape for every plugin set: markets, strategies, model families and feature sets.

A registry is a named set with a source. `register` puts a value in it (as a decorator when
called with one argument), `get` takes one out, `all` returns them all, and `names` lists them.
`require` is the one place a set's membership is checked, so a set is never validated twice in
two files with two error messages.

The source is the module or package that fills the registry, imported on first use rather than
at import time. That is what makes "adding one is a file and a decorator" true: a new market is
a module under `pmlab.markets` carrying `@register`, and nothing else in the tree changes. It
also keeps this module's own import cheap and its import closure small — a registry that reaches
for `pmlab.afml.models` only when asked does not drag scikit-learn into every process that wants
the list of strategies.

Registered today:
- `STRATEGIES`: the families `pmlab.strategies.build` accepts. Their code is frozen by the
  confirmation registered on 2026-09-18 (scripts/confirm.py), so they are adapted here rather
  than decorated in place; `build` stays the one constructor, and tests/test_registry.py holds
  the registry to it family by family.
- `MODEL_FAMILIES`: filled by `pmlab.afml.models`, which declares each family beside its
  defaults and asks `require` in both places that used to test a tuple.
- `FEATURES`: the feature sets the pipeline can put in a table, each naming the symbol that
  produces it. Sets outside the registered package are resolved lazily, by dotted path.
- `RISK_POLICIES`: the risk overlays a strategy can be run under, each naming the suffix its
  live variant carries and the limits it applies. `pmlab.risk` is registered code, so the
  policies are named here and their limits are loaded on first use rather than at import.
- `pmlab.markets.REGISTRY`: the markets themselves (see that package).
"""
import importlib
import json
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Lazy:
    """A value named as "package.module:attribute" and imported when it is first asked for."""
    target: str

    def resolve(self):
        module, _, attr = self.target.partition(":")
        loaded = importlib.import_module(module)
        return getattr(loaded, attr) if attr else loaded


class Registry:
    """A named set of plugins of one kind. Not a global: every registry is a value someone holds."""

    def __init__(self, kind: str, source: str | None = None, fill: Callable[["Registry"], None] | None = None):
        self.kind = kind
        self.source = source            # a module, or a package whose submodules all register
        self.fill = fill                # or a callable that fills this set, for a set that comes from data
        self._entries: dict[str, object] = {}
        self._meta: dict[str, dict] = {}
        self._loaded = False

    # -- filling ----------------------------------------------------------------------------

    def register(self, name, value=None, **meta):
        """register(name, value, **meta), or @register(name, **meta) as a decorator."""
        if value is None:
            def decorate(v):
                self.register(name, v, **meta)
                return v
            return decorate
        if name in self._entries:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self._entries[name] = value
        self._meta[name] = dict(meta)
        return value

    def register_lazy(self, name, target: str, **meta):
        """Register a value by dotted path; it is imported the first time it is asked for."""
        return self.register(name, Lazy(target), **meta)

    def unregister(self, name) -> None:
        self._entries.pop(name, None)
        self._meta.pop(name, None)

    # -- saving and putting back ------------------------------------------------------------

    def snapshot(self) -> tuple:
        """Everything this registry holds, so a caller can put it back exactly as it was.

        `pmlab.hotswap` validates a candidate by importing it, and a plugin module registers itself on
        import (`pmlab.markets.polymarket_btc_5m`, every `pmlab.pricing.models.*`). Validation would
        therefore change the running set as a side effect of *checking* it. Taking a snapshot first and
        restoring it after is what makes "refused, and nothing changed" true rather than hoped for.

        The values are not copied: a plugin is whatever was registered, and restoring means the same
        objects are in the set again, which is what `tests/test_hot_swap.py` compares by identity.

        The source is loaded first, and the snapshot says so. A snapshot of a set that has not loaded
        yet would restore to an *empty* set that believes it has nothing to load — the source's modules
        are already in `sys.modules` by then, so importing them again registers nothing."""
        self.load()
        return (dict(self._entries), {k: dict(v) for k, v in self._meta.items()}, self._loaded)

    def restore(self, snap: tuple) -> None:
        """Put back a `snapshot()`. In place, so anyone already holding this registry sees the same set."""
        entries, meta, loaded = snap
        self._entries.clear()
        self._entries.update(entries)
        self._meta.clear()
        self._meta.update(meta)
        self._loaded = loaded

    # -- reading ----------------------------------------------------------------------------

    def load(self) -> None:
        """Fill the set, once: import the source, then run `fill` if there is one.

        A package source imports every submodule that is not private. `fill` is for a set whose members come
        from data rather than from code — `pmlab.bindings.BINDINGS` is filled from config/bindings.json — and
        it runs with `_loaded` already True, so registering into the set it is filling does not recurse.

        A source or a fill that raises leaves the set unloaded: `_loaded` goes back to False and the next
        caller gets the same exception, rather than a registry that quietly holds whatever registered before
        the failure."""
        if self._loaded or (self.source is None and self.fill is None):
            return
        self._loaded = True                      # set first: a source that imports us must not recurse
        try:
            if self.source is not None:
                module = importlib.import_module(self.source)
                for info in pkgutil.iter_modules(getattr(module, "__path__", [])):
                    if not info.name.startswith("_"):
                        importlib.import_module(f"{self.source}.{info.name}")
            if self.fill is not None:
                self.fill(self)
        except BaseException:                    # a half-filled set is not a set: every caller must see why
            self._loaded = False
            raise

    def names(self) -> tuple[str, ...]:
        self.load()
        return tuple(self._entries)

    def all(self) -> dict:
        return {name: self.get(name) for name in self.names()}

    def get(self, name):
        self.require(name)
        value = self._entries[name]
        if isinstance(value, Lazy):
            self._entries[name] = value = value.resolve()
        return value

    def meta(self, name) -> dict:
        self.require(name)
        return dict(self._meta[name])

    def require(self, name):
        """The one membership check. Raises with the same words the hand-written checks used."""
        self.load()
        if name not in self._entries:
            raise ValueError(f"unknown {self.kind} {name!r}")
        return name

    def __contains__(self, name) -> bool:
        self.load()
        return name in self._entries

    def __len__(self) -> int:
        return len(self.names())

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._entries)} registered)"


# ---------------------------------------------------------------------------------------- strategies

STRATEGIES = Registry("strategy")


@dataclass(frozen=True)
class StrategyFamily:
    """How a strategy family is built, and what it needs from a market's feature table."""
    name: str
    build: Callable[[dict], object]
    needs: tuple[str, ...] = ()              # feature columns its decide() reads
    note: str = ""

    def __call__(self, params: dict | None = None):
        return self.build(dict(params or {}))


def _register_existing_strategies() -> None:
    """The families pmlab.strategies.build accepts, registered by delegating to it.

    Their module is registered code (the confirmation hashes it), so the decorator cannot go on
    the classes themselves until the analysis on 2026-10-24; delegating keeps one constructor and
    one behaviour, which tests/test_registry.py checks family by family."""
    from pmlab import strategies as s
    from pmlab.bars import NINE

    quotes = ("ask", "bid", "ask_age", "bid_age")
    families = [(name, (column, *quotes), f"probability taker on the {column} column")
                for name, column in s.PROB_COLUMNS.items()]
    families += [("fair_maker", ("fair", "ask", "bid"), "rests bids either side of the Q fair value"),
                 ("open_fair", ("fair", *quotes), "buys the leaning side at the open"),
                 ("open_fair_x10", ("fair", *quotes), "open_fair with a smaller lean"),
                 ("endgame_maker", ("fair",), "rests a bid on the decided favourite"),
                 ("endgame_maker_cl", ("fair_cl",), "endgame_maker priced off the settlement feed")]
    families += [(f"flow_{kind}", (f"{kind}_imb", f"{kind}_age", "fair", *quotes), f"follows {kind} bar flow")
                 for kind in NINE]
    for name, needs, note in families:
        STRATEGIES.register(name, StrategyFamily(name, lambda p, n=name: s.build(n, p), needs, note))


# ---------------------------------------------------------------------------------------- model families

MODEL_FAMILIES = Registry("family", source="pmlab.afml.models")


@dataclass(frozen=True)
class ModelFamily:
    """A meta-label model family: its defaults, and anything it needs beyond the locked environment."""
    name: str
    defaults: dict
    needs: tuple[str, ...] = ()              # importable packages the registered environment must not have
    booster: bool = False                    # iterations come from the purged inner validation
    note: str = ""


# ---------------------------------------------------------------------------------------- feature sets

FEATURES = Registry("feature set")


@dataclass(frozen=True)
class FeatureSet:
    """A named group of columns, and the symbol that produces them."""
    name: str
    via: str                                 # the symbol a reader should go to
    columns: Callable[[], tuple[str, ...]]
    build: Callable | None = None
    chapter: int | None = None
    note: str = ""
    meta: dict = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        return tuple(self.columns())


def _register_existing_features() -> None:
    """The feature sets this project already computes. The two research sets live in the AFML
    library, which the registered environment never imports, so they resolve by dotted path."""
    from pmlab import features as f
    from pmlab import micro
    from pmlab.bars import NINE
    from pmlab.bars.base import BAR_COLUMNS

    lazy = lambda target: (lambda t=target: tuple(Lazy(t).resolve()))
    FEATURES.register("base", FeatureSet(
        "base", "pmlab.features.base_features", lambda: ("t", "fair", "sigma", "z_q", "btc_z10", "btc_z30"),
        f.base_features, chapter=2, note="per-second causal columns of one window"))
    # `builds` is what the builder actually puts in the table, per kind: a bar-shaped set *declares* the
    # bar's own columns or the estimators, and builds `<kind>_<suffix>`. Saying so here is what lets
    # `pmlab.hotswap.built_columns` tell whether a registered leg reads a set (review hot_swap 2);
    # tests/test_hot_swap.py holds each list against what the builder returns on a recorded window.
    FEATURES.register("bar", FeatureSet(
        "bar", "pmlab.features.bar_features", lambda: tuple(BAR_COLUMNS), f.bar_features, chapter=2,
        note="the last closed bar of each kind",
        meta={"kinds": NINE, "builds": ("imb", "ret", "age", "n30")}))
    FEATURES.register("micro", FeatureSet(
        "micro", "pmlab.micro.bar_micro_features", lambda: ("kyle", "amihud", "hasbrouck", "vpin", "roll"),
        micro.bar_micro_features, chapter=19, note="microstructural estimators on the bars",
        meta={"kinds": NINE, "builds": ("vpin", "kyle", "amihud", "cs")}))
    FEATURES.register("event", FeatureSet(
        "event", "pmlab.afml.events.FEATURE_NAMES", lazy("pmlab.afml.events:FEATURE_NAMES"),
        chapter=3, note="the event table's features, one row per sampled event"))
    FEATURES.register("microstructure", FeatureSet(
        "microstructure", "pmlab.afml.micro_features.FEATURE_NAMES",
        lazy("pmlab.afml.micro_features:FEATURE_NAMES"), chapter=19,
        note="chapter 19's tape, book and options features"))


# ---------------------------------------------------------------------------------------- risk policies

RISK_POLICIES = Registry("risk policy")

ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "research" / "strategy_params.json"       # what pmlab.live.app.load_engine reads


@dataclass(frozen=True)
class RiskPolicy:
    """A risk overlay a strategy can be run under: what the variant is called, and what limits it applies.

    A strategy and the overlay it runs under are two plugins, not one: the live engine runs `q_taker` bare,
    inside the managed limits, and inside the managed limits plus the microstructure gates, which is one
    strategy and three policies rather than three strategies. `suffix` is the name the variant carries
    (`pmlab.risk.live_variants` builds `<name><suffix>`), and `limits()` is the `pmlab.risk.RiskLimits` it
    applies — None for the bare rule, which is a policy too, and the one the raw legs run under.

    The limits are behind a callable because the live ones are read from research/strategy_params.json, and a
    registry that reads an artefact at import time cannot be imported by anything that does not have it."""
    name: str
    suffix: str
    limits: Callable[[], object] | None = None
    note: str = ""

    def resolve(self):
        """The RiskLimits this policy applies, or None for the bare rule."""
        return None if self.limits is None else self.limits()


def _frozen_live_limits(which: str) -> Callable[[], object]:
    """The frozen live limits `pmlab.live.app.load_engine` hands to `risk.live_variants`, read on first use.

    Frozen, so a later walk-forward run or recording cannot change what a registered leg is running under:
    scripts/fit_live_models.py wrote them into research/strategy_params.json and the engine uses them as they
    are (`pmlab.risk.live_variants`, `limits=`)."""
    def load():
        from pmlab.risk import RiskLimits
        return RiskLimits(**json.loads(PARAMS_FILE.read_text())["live_limits"][which])
    return load


def _register_existing_risk_policies() -> None:
    """The three overlays the live engine runs every strategy under, and the two fixed defaults beside them."""
    RISK_POLICIES.register("raw", RiskPolicy(
        "raw", "", None, "the strategy's own rule and nothing else; the engine runs every leg bare as well"))
    RISK_POLICIES.register("live_managed", RiskPolicy(
        "live_managed", "_managed", _frozen_live_limits("managed"),
        "research/strategy_params.json live_limits.managed: the frozen live limits, no microstructure gates"))
    RISK_POLICIES.register("live_managed_micro", RiskPolicy(
        "live_managed_micro", "_managed_micro", _frozen_live_limits("managed_micro"),
        "live_limits.managed_micro: the frozen live limits with every chapter-19 gate at its live quantile"))
    RISK_POLICIES.register("default", RiskPolicy(
        "default", "_managed", lambda: Lazy("pmlab.risk:DEFAULT").resolve(),
        "pmlab.risk.DEFAULT: the limits fixed before any managed result was computed"))
    RISK_POLICIES.register("micro_default", RiskPolicy(
        "micro_default", "_managed_micro", lambda: Lazy("pmlab.risk:MICRO_DEFAULT").resolve(),
        "pmlab.risk.MICRO_DEFAULT: DEFAULT with every microstructure gate at the 95% training quantile"))


_register_existing_strategies()
_register_existing_features()
_register_existing_risk_policies()
