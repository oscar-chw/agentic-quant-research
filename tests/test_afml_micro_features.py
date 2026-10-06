"""Hand cases and leak guards for chapter 19's microstructural features (pmlab.afml.micro_features, plan afml-pipeline
node 15).

Every test here was mutation-checked: `scripts/micro_features.py --mutations` applies each recorded mutation to a copy of
the library, runs this file against it and requires a failure (research/micro_features.json, "mutations").
"""
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab import store
from pmlab.afml import micro_features as mf
from pmlab.evaluation import CONFIRM_START

S = 1_788_000_000                         # a window start (a multiple of 300 and of 3600 / 12) before the confirmation start
EVENTS_T = np.array([40, 95, 120, 150, 200, 240, 262])


def ms(sec: float) -> int:
    return int(round((S + sec) * 1000))


def tape_frame(ts, size, price, sign, outcome=None, side=None, tx=None) -> pd.DataFrame:
    n = len(ts)
    sign = np.asarray(sign, np.int8)
    outcome = np.asarray(["Up"] * n if outcome is None else outcome, dtype=object)
    if side is None:
        side = np.where((sign > 0) == (outcome == "Up"), "BUY", "SELL")
    return pd.DataFrame({"ts": np.asarray(ts, np.int64), "outcome": outcome, "side": np.asarray(side, dtype=object),
                         "price": np.asarray(price, float), "size": np.asarray(size, float), "sign": sign,
                         "shares": np.asarray(size, float),
                         "tx": [f"0x{i:064x}" for i in range(n)] if tx is None else list(tx)})


def row_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    x, y = a[mf.FEATURE_NAMES].to_numpy(float), b[mf.FEATURE_NAMES].to_numpy(float)
    return bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]))


# ---------------------------------------------------------------------------------------------------- hand cases

def test_round_size_share_hand_example():
    size = [5, 10, 7, 12.5, 100, 4, 20, 3.3333]
    price = [0.5, 0.5, 0.5, 0.8, 0.53, 0.25, 0.37, 0.3]
    shares, usd = mf.round_flags(size, price)
    # multiples of 5 shares; $10.00 (12.5 x 0.80) and $1.00 (4 x 0.25) are whole dollars, $5.00 (10 x 0.5) and
    # $53.00 (100 x 0.53) are already round in shares, $0.99999 is not a whole cent
    assert shares.tolist() == [True, True, False, False, True, False, True, False]
    assert usd.tolist() == [False, False, False, True, False, True, False, False]

    ts = S + np.arange(21)                      # 20 trades known at e = S + 20, one stamped at e itself
    sizes = size + [7.0] * 12 + [5.0]
    prices = price + [0.5] * 12 + [0.5]
    tape = tape_frame(ts, sizes, prices, np.where(np.arange(21) % 2, 1, -1))
    f = mf.tape_features(mf.tape_arrays(tape), [S + 20])
    assert f["m19_round_shares_100"][0] == pytest.approx(4 / 20)
    assert f["m19_round_usd_100"][0] == pytest.approx(2 / 20)


def test_sign_autocorrelation_and_runs_hand_example():
    x = [1, 1, -1, -1, 1, 1, -1, -1]
    # pairs (x[i-1], x[i]): mean product 1/7, means 1/7 and -1/7, variances 48/49 -> (1/7 + 1/49) / (48/49) = 1/6
    assert mf.lag1_autocorr(x) == pytest.approx(1 / 6)
    z, mean_run, last = mf.runs(x)
    # n1 = n2 = 4, 4 runs: mu = 2*16/8 + 1 = 5, var = 2*16*(32 - 8) / (64 * 7) = 768 / 448
    assert z == pytest.approx((4 - 5) / math.sqrt(768 / 448))
    assert (mean_run, last) == (2.0, -2.0)
    assert mf.lag1_autocorr([1, 2, 3, 4]) == pytest.approx(1.0)

    signs = [1, 1, -1, -1] * 5
    tape = tape_frame(S + np.arange(20), [7.13] * 20, [0.41] * 20, signs)
    f = mf.tape_features(mf.tape_arrays(tape), [S + 20])
    # 19 pairs: mean product 1/19, means 1/19 and -1/19, variances 360/361 -> (1/19 + 1/361) / (360/361) = 1/18
    assert f["m19_sign_ac1_100"][0] == pytest.approx(1 / 18)
    assert f["m19_signed_vol_ac1_100"][0] == pytest.approx(1 / 18)
    # n1 = n2 = 10, 10 runs: mu = 11, var = 2*100*(200 - 20) / (400 * 19)
    assert f["m19_runs_z_100"][0] == pytest.approx((10 - 11) / math.sqrt(36000 / 7600))
    assert (f["m19_run_mean_100"][0], f["m19_run_last"][0]) == (2.0, -2.0)


def test_inter_trade_regularity_hand_example():
    assert mf.gap_cv([0, 2, 4, 6]) == 0.0
    assert mf.gap_cv([0, 1, 1, 4]) == pytest.approx(0.5)          # distinct seconds 0, 1, 4: gaps 1 and 3
    assert math.isnan(mf.gap_cv([3, 3, 5]))

    # a slicer: 5 x 21.62 shares every 3 s; a GUI trader: 5 x 10 shares (round, not a slicer); 10 one-off sizes
    slicer = [(10 + 3 * k, 21.62, 1) for k in range(5)]
    gui = [(s, 10.0, -1) for s in (11, 12, 14, 15, 17)]
    other = [(20 + k, 7.13 + 0.01 * k, 1 if k % 2 else -1) for k in range(10)]
    rows = sorted(slicer + gui + other)
    tape = tape_frame([S + r[0] for r in rows], [r[1] for r in rows], [0.4] * 20, [r[2] for r in rows])
    f = mf.tape_features(mf.tape_arrays(tape), [S + 30])
    assert f["m19_slicer_share_100"][0] == pytest.approx(5 / 20)
    assert f["m19_slicer_gap_cv_100"][0] == 0.0

    # minute clustering: 10 + 20 shares in opening seconds (0 and 2 of a minute), 60 shares at second 30
    e = S + 600
    tape = tape_frame([S + 299, S + 300, S + 330, S + 362, e], [1000, 10, 60, 20, 1000], [0.5] * 5, [1, 1, 1, -1, 1])
    f = mf.tape_features(mf.tape_arrays(tape), [e])
    assert f["m19_minute_open_ratio_300"][0] == pytest.approx((30 / 25) / (90 / 300))
    assert f["m19_minute_open_imb_300"][0] == pytest.approx((10 - 20) / 30)

    w = mf.wallet_stats(["a", "a", "a", "a", "b"], [10, 10, 10, 10, 60], [1, 1, 1, -1, 1], [0, 2, 4, 6, 5])
    assert w["top_share"] == pytest.approx(0.6) and w["hhi"] == pytest.approx(0.52) and w["top_flow"] == pytest.approx(1.0)
    assert w["top_gap_cv"] == 0.0                                  # wallet a trades every 2 s


def test_size_entropy_hand_example():
    """19.6.1: the order-size feature is Shannon entropy over floor(log2 size) bins clipped to [-3, 12], divided by
    ln(16) so a flat spread over all 16 bins reads 1 (docs/afml_sections/ch19.md F19.7: it had no hand case)."""
    assert mf.N_BINS == 16 and (mf.ENTROPY_LO, mf.ENTROPY_HI) == (-3, 12)
    assert mf.size_entropy([7, 5, 4, 6]) == 0.0                              # all in the floor(log2) = 2 bin
    assert mf.size_entropy([4, 4, 8, 8]) == pytest.approx(math.log(2) / math.log(16))     # two bins, 0.25
    assert mf.size_entropy([1, 2, 4, 8]) == pytest.approx(math.log(4) / math.log(16))     # four bins, 0.5
    assert mf.size_entropy([2, 2, 2, 4]) == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(16))
    assert mf.size_entropy([2.0 ** k for k in range(-3, 13)]) == pytest.approx(1.0)       # one trade in every bin
    # the clip: anything at or under 1/8 share shares the bottom bin, anything at or over 4096 shares the top one
    assert mf.size_entropy([0.001, 0.125]) == 0.0 and mf.size_entropy([4096, 10 ** 7]) == 0.0
    assert mf.size_entropy([0.125, 4096]) == pytest.approx(math.log(2) / math.log(16))
    assert math.isnan(mf.size_entropy([7])) and math.isnan(mf.size_entropy([0, -3, 7]))


def test_the_minute_open_ratio_reads_the_events_second_while_the_tape_is_shorter_than_its_lookback():
    """19.6.3 and docs/afml_sections/ch19.md F19.8: m19_minute_open_ratio_300 divides minute-open shares by a fixed
    25 s and all shares by a fixed 300 s. A token barely trades before its window opens, so early in a window the
    lookback holds only t seconds of flow, and under perfectly flat flow the ratio is 12 x (minute-open seconds
    since the open) / t rather than 1: 2.0 at t = 30, 1.82 at t = 66, 1.01 at t = 119. The feature encodes the
    event's second. Dividing by the seconds actually observed gives 1, which is what a no-clustering reading needs."""
    flat = tape_frame([S + k for k in range(300)], [10.0] * 300, [0.5] * 300, [1] * 300)
    arrays = mf.tape_arrays(flat)
    opens_by_t = lambda t: sum(1 for k in range(t) if k % 60 < mf.MINUTE_OPEN_S)
    for t, expected in ((30, 2.0), (66, 12 * 10 / 66), (119, 12 * 10 / 119)):
        got = mf.tape_features(arrays, [S + t])[f"m19_minute_open_ratio_{mf.MINUTE_LOOKBACK}"][0]
        assert got == pytest.approx(12 * opens_by_t(t) / t) == pytest.approx(expected)
        # the same shares against the seconds actually in the lookback read 1: flat flow, no minute clustering
        v_open, v_all = 10.0 * opens_by_t(t), 10.0 * t
        assert (v_open / opens_by_t(t)) / (v_all / t) == pytest.approx(1.0)
        assert got == pytest.approx((v_open / (mf.MINUTE_LOOKBACK // 60 * mf.MINUTE_OPEN_S)) / (v_all / mf.MINUTE_LOOKBACK))
    assert opens_by_t(66) == 10 and opens_by_t(119) == 10
    # once the lookback is full the fixed denominators are the right ones and flat flow reads 1
    full = mf.tape_features(arrays, [S + 300])[f"m19_minute_open_ratio_{mf.MINUTE_LOOKBACK}"][0]
    assert full == pytest.approx(1.0) and opens_by_t(300) == 25


def hand_book(snap_times=(10.0, 65.0)):
    snaps = {"ts_ms": np.array([ms(x) for x in snap_times]), "recv_q": np.array([ms(x) for x in snap_times]) << 10}
    levels = []
    for i, bids in enumerate(({5000: 100, 4900: 50, 4000: 70}, {5000: 95, 4900: 50, 4000: 70})):
        levels += [(i, 0, p, s) for p, s in bids.items()] + [(i, 1, p, s) for p, s in {5100: 80, 5200: 40}.items()]
    for k, col in enumerate(("snap", "side", "price", "size")):
        snaps[col] = np.array([lv[k] for lv in levels])
    changes = [(60, 0, 5000, 90, 5000, 5100),      # before the lookback
               (75, 0, 5000, 60, 5000, 5100),      # -35 against the later snapshot (95), 30 traded at 75.2
               (80, 0, 5000, 20, 5000, 5100),      # -40, no trade: cancelled
               (82, 1, 5100, 100, 5000, 5100),     # +20 at the ask touch
               (85, 0, 4000, 0, 5000, 5100),       # -70 ten cents from the touch: not near
               (88, 1, 5300, 25, 5000, 5100),      # +25 two cents behind the ask: near
               (90, 0, 4900, 30, 5000, 5100),      # -20, 20 traded at 90.5
               (99.5, 1, 5200, 0, 5000, 5100),     # -40; the trade that took it prints at 100.3, after the event
               (100.0, 0, 5000, 0, 4900, 5100)]    # at the event instant: not known
    ch = {"ts_ms": np.array([ms(c[0]) for c in changes]), "recv_q": np.array([ms(c[0]) for c in changes]) << 10,
          "side": np.array([c[1] for c in changes]), "price": np.array([c[2] for c in changes]),
          "size": np.array([c[3] for c in changes], float), "best_bid": np.array([c[4] for c in changes], float),
          "best_ask": np.array([c[5] for c in changes], float)}
    trades = [(60.1, 0, 5000, 10), (75.2, 0, 5000, 30), (90.5, 0, 4900, 20), (100.3, 1, 5200, 40)]
    tr = mf.Trades(np.array([ms(x[0]) for x in trades]), np.array([x[1] for x in trades]), np.array([x[2] for x in trades]),
                   np.array([x[3] for x in trades], float))
    return mf.book_deltas(ch, snaps), tr


def test_cancellation_counting_hand_example():
    book, trades = hand_book()
    e = S + 100
    f = {k: v[0] for k, v in mf.book_features(book, trades, np.zeros((0, 2)), [e]).items()}
    assert f["m19_bk_ok"] == 1.0
    # bid: level 50c decreased 35 + 40, traded 30 -> 45 cancelled; 49c decreased 20, traded 20 -> 0; 40c is far
    assert f["m19_bk_cancel_bid_30"] == pytest.approx(45 / 30)
    assert f["m19_bk_add_bid_30"] == 0.0 and math.isnan(f["m19_bk_cancel_add_bid_30"])
    assert f["m19_bk_cancel_n_bid_30"] == pytest.approx(1 / 30)        # only the 80 s decrease has no trade within 1 s
    # ask: 20 + 25 added; 40 decreased at 52c with no trade before the event
    assert f["m19_bk_add_ask_30"] == pytest.approx(45 / 30)
    assert f["m19_bk_cancel_ask_30"] == pytest.approx(40 / 30)
    assert f["m19_bk_cancel_add_ask_30"] == pytest.approx(40 / 45)
    assert f["m19_bk_cancel_n_ask_30"] == pytest.approx(1 / 30)
    assert f["m19_bk_cancel_trade_30"] == pytest.approx((45 + 40) / (30 + 20))

    # a silence of 6 s inside the lookback, known by the event: not recorded; a 7 s stall that began 3 s before the
    # event is not yet known at it; a lookback that starts before the first snapshot: not recorded
    stall = mf.book_features(book, trades, np.array([[S + 80, S + 86]]), [e])
    assert stall["m19_bk_ok"][0] == 0.0 and all(math.isnan(stall[k][0]) for k in mf.BOOK_NAMES if k != "m19_bk_ok")
    late = mf.book_features(book, trades, np.array([[S + 97, S + 104]]), [e])
    assert late["m19_bk_ok"][0] == 1.0 and late["m19_bk_cancel_bid_30"][0] == pytest.approx(45 / 30)
    early, _ = hand_book(snap_times=(72.0, 73.0))
    assert mf.book_features(early, trades, np.zeros((0, 2)), [e])["m19_bk_ok"][0] == 0.0


# ---------------------------------------------------------------------------------------------------- recorded files

def write_changes(path: Path, rows):
    """rows: (ts s, window, outcome, side, price, size, best_bid, best_ask) as the recorder sees them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    e4 = lambda v: pa.array([None if x is None else int(round(x * 1e4)) for x in v], pa.int16())
    pq.write_table(pa.table({
        "recv_q": pa.array([int((r[0] + 3) * (1 << 22)) for r in rows], pa.int64()),
        "ts_ms": pa.array([int(round(r[0] * 1000)) for r in rows], pa.int64()),
        "window": pa.array([r[1] for r in rows], pa.int64()), "outcome": [r[2] for r in rows],
        "asset_id": ["a" + r[2] for r in rows], "side": [r[3] for r in rows], "price_e4": e4([r[4] for r in rows]),
        "size": pa.array([float(r[5]) for r in rows]), "best_bid_e4": e4([r[6] for r in rows]),
        "best_ask_e4": e4([r[7] for r in rows]), "mirror": pa.array([None] * len(rows), pa.int8()),
        "mirror_asset": pa.array([None] * len(rows), pa.string())}), path)


def write_book(path: Path, rows):
    """rows: (ts s, window, outcome, bids [(price, size)], asks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lp = lambda lv: pa.array([[int(round(p * 1e4)) for p, _ in x] for x in lv], pa.list_(pa.int16()))
    ls = lambda lv: pa.array([[float(s) for _, s in x] for x in lv], pa.list_(pa.float64()))
    pq.write_table(pa.table({
        "recv_q": pa.array([int((r[0] + 3) * (1 << 22)) for r in rows], pa.int64()),
        "ts_ms": pa.array([int(round(r[0] * 1000)) for r in rows], pa.int64()),
        "window": pa.array([r[1] for r in rows], pa.int64()), "outcome": [r[2] for r in rows],
        "asset_id": ["a" + r[2] for r in rows], "bid_price_e4": lp([r[3] for r in rows]), "bid_size": ls([r[3] for r in rows]),
        "ask_price_e4": lp([r[4] for r in rows]), "ask_size": ls([r[4] for r in rows]),
        "hash_b20": pa.array([None] * len(rows), pa.binary(20)), "hash": pa.array(["h"] * len(rows), pa.string())}), path)


def write_trades(path: Path, rows):
    """rows: (ts s, window, outcome, side, price, size)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"recv": [r[0] + 3 for r in rows], "ts": [float(r[0]) for r in rows],
                             "window": pa.array([r[1] for r in rows], pa.int64()), "outcome": [r[2] for r in rows],
                             "asset_id": ["a" + r[2] for r in rows], "side": [r[3] for r in rows],
                             "price": [float(r[4]) for r in rows], "size": [float(r[5]) for r in rows],
                             "fee_rate_bps": [0.0] * len(rows), "tx": ["0x"] * len(rows)}), path)


def hour_dir(root: Path, kind: str, hour: int) -> Path:
    return root / kind / f"hour={mf.hour_name(hour)}"


def test_down_token_rows_enter_the_up_frame_mirrored(tmp_path):
    w = S
    hour = S - S % 3600
    write_changes(hour_dir(tmp_path, "price_change", hour) / "v2.parquet",
                  [(S + 1.0, w, "Down", "BUY", 0.55, 10, 0.54, 0.56),     # a Down bid at 55c is an Up ask at 45c
                   (S + 2.0, w, "Up", "SELL", 0.47, 7, 0.44, 0.46),
                   (S + 3.0, w, "Up", "BUY", 0.43, 5, 0.43, None)])
    write_trades(hour_dir(tmp_path, "trade", hour) / "data.parquet",
                 [(S + 1.5, w, "Down", "BUY", 0.55, 3),                   # buying Down takes the Up bid at 45c
                  (S + 2.5, w, "Up", "BUY", 0.46, 4)])
    ch, stamps = mf.read_changes(hour_dir(tmp_path, "price_change", hour) / "v2.parquet", [w])
    c = ch[w]
    assert c["side"].tolist() == [1, 1, 0] and c["price"].tolist() == [4500, 4700, 4300]
    assert c["best_bid"][:2].tolist() == [4400, 4400] and c["best_ask"][:2].tolist() == [4600, 4600]
    assert math.isnan(c["best_ask"][2]) and stamps.tolist() == [S + 1.0, S + 2.0, S + 3.0]
    tr = mf.read_trades(hour_dir(tmp_path, "trade", hour) / "data.parquet", [w])[w]
    assert tr["side"].tolist() == [0, 1] and tr["price"].tolist() == [4500, 4600]


def test_nothing_at_or_after_the_confirmation_start_is_read(tmp_path, monkeypatch):
    # windows: the last one allowed ends exactly at the confirmation start
    tape = mf.tape_arrays(tape_frame([], [], [], []))
    assert len(mf.window_features(CONFIRM_START - T, [100], tape)) == 1
    with pytest.raises(ValueError):
        mf.window_features(CONFIRM_START, [100], tape)

    # live hour files: listed only when the whole hour ends by the confirmation start; later files are never opened
    c = S - S % 3600 + 7200
    w = c - T
    before, last = c - 7200, c - 3600
    for kind, name in (("price_change", "v2.parquet"), ("book", "v2.parquet"), ("trade", "data.parquet"), ("tape", "data.parquet")):
        for h in (c, c + 3600):
            hour_dir(tmp_path, kind, h).mkdir(parents=True)
            (hour_dir(tmp_path, kind, h) / name).write_bytes(b"not parquet: must never be read")
    write_changes(hour_dir(tmp_path, "price_change", before) / "v2.parquet", [(before + 5.0, w - T, "Up", "BUY", 0.5, 1, 0.5, 0.51)])
    write_changes(hour_dir(tmp_path, "price_change", last) / "v2.parquet",
                  [(w - 200 + k, w, "Up", "BUY", 0.50, 10 + k, 0.50, 0.51) for k in range(400)])
    write_book(hour_dir(tmp_path, "book", last) / "v2.parquet", [(w - 250.0, w, "Up", [(0.50, 9)], [(0.51, 5)])])
    write_trades(hour_dir(tmp_path, "trade", last) / "data.parquet", [(w + 10.0, w, "Up", "SELL", 0.5, 1)])
    listed = [h for h, _ in mf.live_hours(tmp_path, "price_change", before, c + 7200, confirm_start=c)]
    assert listed == [before, last]
    opened = []
    real = pq.read_table
    monkeypatch.setattr(mf.pq, "read_table", lambda p, *a, **k: opened.append(Path(p)) or real(p, *a, **k))
    got = {x[0]: x for x in mf.book_windows([w], root=tmp_path, confirm_start=c)}
    assert got[w][1] is not None and len(got[w][1].ts_ms) == 400
    assert opened and all(p.parent.name < f"hour={mf.hour_name(c)}" for p in opened)
    assert mf.read_wallets(before, c + 7200, root=tmp_path, confirm_start=c) == ({}, [])

    # the round-size pool: trades of windows ending after the confirmation start, or stamped after it, are dropped
    rows = pd.DataFrame({"start": [c - 600, c - 300, c - 300, c], "ts": [c - 700, c - 10, c + 5, c + 20],
                         "size": [5.0, 5.0, 5.0, 5.0], "price": [0.5] * 4})
    asked = []
    pool = mf.pool_for_day(store.day_name(c - 1), confirm_start=c, read_day=lambda kind, day: asked.append(day) or rows)
    assert pool.ts.tolist() == [c - 700, c - 700, c - 10, c - 10] and all(store.day_start(d) < c for d in asked)

    # DVOL: a minute known after the confirmation start is never used, even for a later instant
    dvol = pd.DataFrame({"ts": [(c - 360) * 1000, c * 1000], "close": [50.0, 999.0]})
    f = mf.dvol_features([c + 200], dvol, confirm_start=c)
    assert f["m19_dvol"][0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------------------------------- leak test

@dataclass
class World:
    tape: pd.DataFrame
    wallets: dict
    pool: pd.DataFrame
    changes: dict
    snaps: dict
    trades: dict
    gaps: np.ndarray
    dvol: pd.DataFrame
    rv300: np.ndarray

    def features(self) -> pd.DataFrame:
        tp = mf.tape_arrays(self.tape, self.wallets)
        pool = mf.make_pool(self.pool["ts"], self.pool["size"], self.pool["price"])
        book = mf.book_deltas(self.changes, self.snaps)
        tr = mf.Trades(self.trades["ts_ms"], self.trades["side"], self.trades["price"], self.trades["size"])
        return mf.window_features(S, EVENTS_T, tp, pool, book, tr, self.gaps, self.dvol, self.rv300)


def make_world(seed: int = 11) -> World:
    rng = np.random.default_rng(seed)
    e = S + EVENTS_T
    ts = np.concatenate([rng.integers(S - 240, S + T + 10, 700), e, e - 1])
    n = len(ts)
    p_up = np.clip(np.round(0.5 + np.cumsum(rng.normal(0, 0.01, n)), 2), 0.03, 0.97)
    sign = rng.choice([1, -1], n)
    outcome = rng.choice(["Up", "Down"], n)
    price = np.where(outcome == "Up", p_up, 1 - p_up)
    kind = rng.integers(0, 3, n)
    size = np.where(kind == 0, rng.choice([5.0, 10.0, 20.0, 50.0], n),
                    np.where(kind == 1, np.round(rng.integers(1, 20, n) / price, 6), np.round(rng.exponential(15, n), 2) + 0.01))
    slicer = S - 200 + 5 * np.arange(95)                         # 13.37 shares of Up every 5 s
    ts, sign = np.r_[ts, slicer], np.r_[sign, np.ones(len(slicer), int)]
    outcome, price = np.r_[outcome, np.full(len(slicer), "Up")], np.r_[price, np.full(len(slicer), 0.41)]
    size = np.r_[size, np.full(len(slicer), 13.37)]
    o = np.argsort(ts, kind="stable")
    tape = tape_frame(ts[o], size[o], price[o], sign[o], outcome[o])
    keys = mf.wallet_key(tape["tx"], tape["outcome"], tape["side"], tape["size"])
    wallets = {k: f"w{rng.integers(0, 5)}" for k in keys if rng.random() < 0.97}
    pts = np.concatenate([rng.integers(S - 4000, S + T + 50, 4000), e, e - 1])
    pp = rng.uniform(0.05, 0.95, len(pts))
    pool = pd.DataFrame({"ts": pts, "price": pp, "size": np.where(rng.random(len(pts)) < 0.4, 10.0, rng.exponential(12, len(pts)) + 0.013)})

    m = 6000
    ch_ts = np.sort(rng.integers(ms(-250), ms(T), m))
    mid = 5000 + 100 * np.round(np.cumsum(rng.normal(0, 0.05, m))).astype(np.int64)
    side = rng.integers(0, 2, m)
    price = np.where(side == 0, mid - 100 * rng.integers(0, 4, m), mid + 100 + 100 * rng.integers(0, 4, m))
    size = np.where(rng.random(m) < 0.15, 0.0, np.round(rng.exponential(100, m), 2))
    recv = ch_ts + rng.integers(-900, 900, m)
    special = []
    for x in e * 1000:                        # per event: a change whose receive order crosses one after the event,
        lvl = 5000                            # one at the event instant, and a trade just after the event at that level
        special += [(x - 300, x + 900, 0, lvl, 60.0), (x + 200, x - 800, 0, lvl, 10.0), (x, x, 1, lvl + 100, 0.0)]
    sp = np.array(special, dtype=float)
    changes = {"ts_ms": np.r_[ch_ts, sp[:, 0]].astype(np.int64), "recv_q": np.r_[recv, sp[:, 1]].astype(np.int64) << 12,
               "side": np.r_[side, sp[:, 2]].astype(np.int64), "price": np.r_[price, sp[:, 3]].astype(np.int64),
               "size": np.r_[size, sp[:, 4]], "best_bid": np.r_[mid, np.full(len(sp), 5000)].astype(float),
               "best_ask": np.r_[mid + 100, np.full(len(sp), 5100)].astype(float)}
    snap_ts = np.r_[ms(-255), ms(100.5), ms(200.5), e[[2, 5]] * 1000]
    grid = np.arange(4000, 6100, 100)
    lv = [(i, s, p, float(np.round(rng.exponential(80), 2))) for i in range(len(snap_ts)) for s in (0, 1) for p in grid]
    snaps = {"ts_ms": snap_ts.astype(np.int64), "recv_q": snap_ts.astype(np.int64) << 12,
             "snap": np.array([x[0] for x in lv]), "side": np.array([x[1] for x in lv]),
             "price": np.array([x[2] for x in lv]), "size": np.array([x[3] for x in lv])}
    k = 800
    tr_ts = np.r_[rng.integers(ms(-250), ms(T), k), e * 1000 + 400]
    trades = {"ts_ms": tr_ts.astype(np.int64), "side": np.r_[rng.integers(0, 2, k), np.zeros(len(e))].astype(np.int64),
              "price": np.r_[5000 + 100 * rng.integers(-3, 4, k), np.full(len(e), 5000)].astype(np.int64),
              "size": np.r_[np.round(rng.exponential(20, k), 2), np.full(len(e), 25.0)]}
    gaps = np.array([(S + 130, S + 137)] + [(x - 3, x + 1) for x in e], dtype=float)
    minutes = np.arange(S - 7200, S + T + 600, 60)
    dvol = pd.DataFrame({"ts": minutes * 1000, "close": rng.uniform(40, 60, len(minutes))})
    return World(tape, wallets, pool, changes, snaps, trades, gaps, dvol, rng.uniform(1e-5, 5e-5, len(EVENTS_T)))


def cut(world: World, e: int, remove: bool, rng) -> World:
    """Every input at or after second e perturbed or removed; nothing before it touched."""
    x = e * 1000
    tape = world.tape.copy()
    late = tape["ts"].to_numpy() >= e
    pool = world.pool.copy()
    plate = pool["ts"].to_numpy() >= e
    ch = {k: v.copy() for k, v in world.changes.items()}
    cl = ch["ts_ms"] >= x
    sn = {k: v.copy() for k, v in world.snaps.items()}
    sl = sn["ts_ms"] >= x
    tr = {k: v.copy() for k, v in world.trades.items()}
    tl = tr["ts_ms"] >= x
    dvol = world.dvol.copy()
    dl = dvol["ts"].to_numpy() // 1000 + 60 >= e
    gaps = world.gaps.copy()
    if remove:
        tape, pool, dvol = tape[~late], pool[~plate], dvol[~dl]
        ch = {k: v[~cl] for k, v in ch.items()}
        tr = {k: v[~tl] for k, v in tr.items()}
        keep_lv = ~sl[sn["snap"]]
        remap = np.cumsum(~sl) - 1
        sn = {"ts_ms": sn["ts_ms"][~sl], "recv_q": sn["recv_q"][~sl], "snap": remap[sn["snap"][keep_lv]],
              "side": sn["side"][keep_lv], "price": sn["price"][keep_lv], "size": sn["size"][keep_lv]}
        gaps = gaps[gaps[:, 0] < e]
        gaps[:, 1] = np.where(gaps[:, 1] > e, np.inf, gaps[:, 1])     # a silence still open at e
    else:
        tape.loc[late, "size"] = tape.loc[late, "size"] * 3 + 0.37
        tape.loc[late, "shares"] = tape.loc[late, "size"]
        tape.loc[late, "sign"] = -tape.loc[late, "sign"]
        tape.loc[late, "tx"] = [f"0x{'f' * 60}{i:04x}" for i in range(int(late.sum()))]
        pool.loc[plate, "size"] = 5.0
        ch["size"][cl] = ch["size"][cl] + 17.5
        ch["side"][cl] = 1 - ch["side"][cl]
        sn["size"][sl[sn["snap"]]] += 33.0
        tr["size"][tl] *= 2.0
        tr["price"][tl] = 5000
        dvol.loc[dl, "close"] = dvol.loc[dl, "close"] * 3
        gaps[:, 1] = np.where(gaps[:, 1] >= e, gaps[:, 1] + 50, gaps[:, 1])
        gaps = np.vstack([gaps, [e + 1, e + 30]])
    wallets = dict(world.wallets)
    if not remove:
        wallets.update({k: "w_late" for k in mf.wallet_key(tape.loc[late, "tx"], tape.loc[late, "outcome"],
                                                           tape.loc[late, "side"], tape.loc[late, "size"])})
    return replace(world, tape=tape, wallets=wallets, pool=pool, changes=ch, snaps=sn, trades=tr, gaps=gaps, dvol=dvol)


@pytest.mark.parametrize("remove", [False, True], ids=["perturb", "remove"])
def test_perturbing_or_removing_anything_after_an_event_changes_nothing_known_at_it(remove):
    world = make_world()
    base = world.features()
    # the world exercises every group: otherwise equality would hold vacuously
    for col in mf.FEATURE_NAMES:
        assert np.isfinite(base[col]).sum() >= 3, col
    assert base["m19_bk_ok"].tolist().count(0.0) >= 1 and base["m19_bk_ok"].sum() >= 4
    rng = np.random.default_rng(5)
    for i, t in enumerate(EVENTS_T):
        e = S + int(t)
        after = cut(world, e, remove, rng).features()
        assert row_equal(base.iloc[[i]], after.iloc[[i]]), (t, remove)
        if i + 1 < len(EVENTS_T):
            assert not row_equal(base.iloc[i + 1:], after.iloc[i + 1:]), "the cut changes nothing at all"
