"""The other three registries (plan/markets.md node 9): strategy, model family and feature set.

One shape for all of them, so adding one is a file and a decorator rather than edits in several
places. Two claims carry the node:

- **the existing sets register themselves**: the model families are no longer a tuple written in
  one file and tested in two others, and the strategies the app hand-assembles are reachable by
  name through the registry;
- **and behave identically**: family by family, the registry builds the object `pmlab.strategies.
  build` builds; family by family, `FAMILIES`, `BOOSTERS` and `DEFAULTS` are what they were, and
  both places that used to test the tuple now refuse the same names.

The registry itself is held to its own contract: register, get, all, names, require, a source
imported on first use and not before, and a lazy value resolved only when it is asked for.
"""
import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmlab import strategies as built_in                                        # noqa: E402
from pmlab.bars import NINE                                                     # noqa: E402
from pmlab.registry import (FEATURES, MODEL_FAMILIES, STRATEGIES, FeatureSet, Lazy,  # noqa: E402
                            ModelFamily, Registry, StrategyFamily)

OLD_FAMILIES = ("bagged_trees", "logistic", "hgb", "xgb")        # the tuple that used to live in afml/models.py
OLD_BOOSTERS = ("hgb", "xgb")
OLD_DEFAULTS_KEYS = {
    "bagged_trees": {"base", "n_estimators", "max_features", "min_weight_fraction_leaf", "max_depth", "balanced",
                     "update_trees"},
    "logistic": {"C", "class_weight", "max_iter"},
    "hgb": {"learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features",
            "class_weight", "val_splits"},
    "xgb": {"learning_rate", "n_estimators", "max_depth", "min_child_weight", "subsample", "colsample_bytree",
            "reg_lambda", "reg_alpha", "gamma", "val_splits"},
}


# ------------------------------------------------------------------ the shape

@pytest.mark.case
def test_a_registry_registers_gets_lists_and_requires():
    r = Registry("widget")
    assert r.names() == () and len(r) == 0 and "a" not in r
    r.register("a", 1, colour="red")
    assert r.get("a") == 1 and r.names() == ("a",) and r.all() == {"a": 1} and "a" in r
    assert r.meta("a") == {"colour": "red"}
    with pytest.raises(ValueError, match="widget 'a' is already registered"):
        r.register("a", 2)
    with pytest.raises(ValueError, match="unknown widget 'b'"):
        r.get("b")
    with pytest.raises(ValueError, match="unknown widget 'b'"):
        r.require("b")
    r.unregister("a")
    assert r.names() == ()


@pytest.mark.case
def test_a_registry_registers_by_decorator_so_a_plugin_is_a_file_and_a_decorator():
    r = Registry("widget")

    @r.register("spinner", chapter=2)
    def spin():
        return "spun"

    assert r.get("spinner") is spin and spin() == "spun" and r.meta("spinner") == {"chapter": 2}


@pytest.mark.case
def test_a_lazy_entry_is_imported_only_when_it_is_asked_for():
    r = Registry("widget")
    r.register_lazy("sections", "json:dumps")
    assert r._entries["sections"] == Lazy("json:dumps")        # still a path, not the value
    assert r.get("sections") is json.dumps
    assert r._entries["sections"] is json.dumps                # resolved once, then kept


@pytest.mark.case
def test_a_registrys_source_is_imported_on_first_use_and_not_at_import():
    """That is what keeps `import pmlab.registry` cheap and its import closure small: a registry
    that reached for pmlab.afml.models eagerly would pull scikit-learn into every process."""
    fresh = Registry("family", source="pmlab.afml.models")
    assert fresh._entries == {} and not fresh._loaded
    fresh.load()
    assert fresh._loaded
    # the source registers into MODEL_FAMILIES, the registry it names, so this one stays empty:
    # a source is where registrations come from, not a second copy of them
    assert set(MODEL_FAMILIES.names()) == set(OLD_FAMILIES)


@pytest.mark.case
def test_a_package_source_imports_every_submodule_that_is_not_private():
    from pmlab import markets

    assert markets.REGISTRY.source == "pmlab.markets"
    assert set(markets.names()) >= {"polymarket-btc-5m", "binance-btcusdt-spot"}


# ------------------------------------------------------------------ strategies

@pytest.mark.case
def test_every_registered_strategy_family_is_one_the_builder_accepts():
    """Both directions: nothing registered that `build` rejects, and none of the names `build`
    reads out of its own tables left out."""
    names = set(STRATEGIES.names())
    for name in names:
        assert built_in.build(name, {}) is not None
    with pytest.raises(ValueError, match="unknown strategy family"):
        built_in.build("not_a_family", {})
    assert set(built_in.PROB_COLUMNS) <= names                     # the probability takers, from build's own table
    assert {f"flow_{k}" for k in NINE} <= names                    # one per bar kind, from pmlab.bars
    assert {"fair_maker", "open_fair", "open_fair_x10", "endgame_maker", "endgame_maker_cl"} <= names


@pytest.mark.case
@pytest.mark.parametrize("name", sorted(set(built_in.PROB_COLUMNS) | {"fair_maker", "open_fair", "endgame_maker",
                                                                      "flow_dollar_imbalance"}))
def test_a_registered_strategy_is_the_one_the_builder_builds(name):
    """Identical behaviour, not merely an equivalent one: the same dataclass with the same fields."""
    family = STRATEGIES.get(name)
    assert isinstance(family, StrategyFamily) and family.name == name
    mine, theirs = family({}), built_in.build(name, {})
    assert type(mine) is type(theirs) and dataclasses.asdict(mine) == dataclasses.asdict(theirs)
    tuned = {"t_min": 11} if hasattr(theirs, "t_min") else {}
    assert dataclasses.asdict(family(tuned)) == dataclasses.asdict(built_in.build(name, tuned))


@pytest.mark.case
def test_the_live_strategies_the_app_assembles_are_all_reachable_by_name():
    """pmlab.live.app hand-assembles its strategies from research/strategy_params.json plus a few
    fixed rules; every one of those names has to be in the registry, or "add a strategy" would
    still mean editing the app."""
    params = json.loads((ROOT / "research" / "strategy_params.json").read_text())["strategies"]
    named = set(params) | {"q_taker_cl", "open_fair", "open_fair_x10", "endgame_maker", "endgame_maker_cl",
                           "drift_taker"}
    assert named <= set(STRATEGIES.names()), sorted(named - set(STRATEGIES.names()))


@pytest.mark.case
def test_a_registered_strategy_declares_the_columns_it_reads():
    for name in STRATEGIES.names():
        family = STRATEGIES.get(name)
        assert family.needs and family.note
    assert "fair" in STRATEGIES.get("q_taker").needs
    assert "dollar_imbalance_imb" in STRATEGIES.get("flow_dollar_imbalance").needs


# ------------------------------------------------------------------ model families

@pytest.mark.case
def test_the_model_families_register_themselves_and_are_what_they_were():
    from pmlab.afml import models

    assert models.FAMILIES == OLD_FAMILIES                 # same names, same order
    assert models.BOOSTERS == OLD_BOOSTERS
    assert set(models.DEFAULTS) == set(OLD_FAMILIES)
    for name, keys in OLD_DEFAULTS_KEYS.items():
        assert set(models.DEFAULTS[name]) == keys
        entry = MODEL_FAMILIES.get(name)
        assert isinstance(entry, ModelFamily) and entry.defaults is models.DEFAULTS[name]
    assert MODEL_FAMILIES.get("xgb").needs == ("xgboost",)
    assert MODEL_FAMILIES.get("logistic").needs == ()


@pytest.mark.case
def test_both_places_that_tested_the_tuple_now_ask_the_registry():
    from pmlab.afml import models

    with pytest.raises(ValueError, match="unknown family 'nope'"):
        models.MetaModel("nope")
    with pytest.raises(ValueError, match="unknown family 'nope'"):
        models.PurgedSearch("nope", {})
    assert models.MetaModel("logistic").family == "logistic"
    assert models.PurgedSearch("logistic", {"C": [1.0]}).family == "logistic"


@pytest.mark.case
def test_the_model_family_tuple_is_no_longer_written_anywhere():
    """The point of the node: one declaration, not a tuple in one file and a check in two others."""
    text = (ROOT / "src" / "pmlab" / "afml" / "models.py").read_text()
    assert 'FAMILIES = ("bagged_trees"' not in text
    assert text.count("MODEL_FAMILIES.require(family)") == 2


# ------------------------------------------------------------------ feature sets

@pytest.mark.case
def test_the_feature_sets_register_themselves_with_the_symbol_that_produces_them():
    assert set(FEATURES.names()) == {"base", "bar", "micro", "event", "microstructure"}
    for name in FEATURES.names():
        feature_set = FEATURES.get(name)
        assert isinstance(feature_set, FeatureSet) and feature_set.via and feature_set.names()
    from pmlab.afml import events, micro_features

    assert FEATURES.get("event").names() == tuple(events.FEATURE_NAMES)
    assert FEATURES.get("microstructure").names() == tuple(micro_features.FEATURE_NAMES)
    from pmlab import features as base_features

    assert FEATURES.get("base").build is base_features.base_features
    assert "close" in FEATURES.get("bar").names() and FEATURES.get("bar").meta["kinds"] == NINE


@pytest.mark.case
def test_a_source_that_fails_to_import_says_so_every_time_it_is_asked():
    """`load()` marks itself loaded before it imports, so the first caller saw the ImportError and every
    later one saw a silently short set. A registry that could not fill itself has to refuse, not shrink
    (review new_layers.md registry 1)."""
    reg = Registry("thing", source="pmlab._no_such_source_module")
    with pytest.raises(ModuleNotFoundError):
        reg.names()
    with pytest.raises(ModuleNotFoundError):
        reg.names()
    with pytest.raises(ModuleNotFoundError):
        reg.get("anything")
