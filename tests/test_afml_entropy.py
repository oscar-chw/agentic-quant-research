import numpy as np
import pytest

from pmlab.afml import entropy as en

H01 = -(0.1 * np.log2(0.1) + 0.9 * np.log2(0.9))       # 0.469 bits


@pytest.mark.case
def test_plug_in_hand_cases():
    assert en.plug_in("0000", 1) == 0.0
    assert en.plug_in("01" * 500, 1) == pytest.approx(1.0)
    # words of two: "01" and "10" in (nearly) equal numbers -> 1 bit per 2 symbols
    assert en.plug_in("01" * 500, 2) == pytest.approx(0.5, abs=1e-6)
    assert en.plug_in("0012", 1) == pytest.approx(1.5)                  # p = 1/2, 1/4, 1/4


@pytest.mark.eval
def test_plug_in_converges_on_iid_sources():
    """Thresholds (100,000 symbols): fair bits within 0.005 of 1 for w = 1..3; Bernoulli(0.1) within 0.005
    of 0.469; four equiprobable symbols within 0.005 of 2."""
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 100_000)
    for w in (1, 2, 3):
        assert en.plug_in(b, w) == pytest.approx(1.0, abs=0.005)
    assert en.plug_in((rng.random(100_000) < 0.1).astype(int), 1) == pytest.approx(H01, abs=0.005)
    assert en.plug_in(rng.integers(0, 4, 100_000), 1) == pytest.approx(2.0, abs=0.005)


@pytest.mark.case
def test_lempel_ziv_library_hand_case():
    lib, c = en.lempel_ziv("11011010000011")
    assert lib == ["1", "10", "11", "0", "100", "00", "01"]
    assert c == 0.5


@pytest.mark.eval
def test_lempel_ziv_orders_sources_by_complexity():
    """Threshold (2000 symbols): complexity iid bits > 2x Bernoulli(0.1) > periodic."""
    rng = np.random.default_rng(1)
    iid = en.lempel_ziv(rng.integers(0, 2, 2000))[1]
    skew = en.lempel_ziv((rng.random(2000) < 0.1).astype(int))[1]
    per = en.lempel_ziv([0, 1] * 1000)[1]
    assert iid > 1.5 * skew > per


@pytest.mark.case
def test_match_length():
    assert en.match_length("101010", 2, 2) == (3, "10")
    assert en.match_length("abcxyz", 3, 3) == (1, "")
    # the match may run past i: "aaaa" from i = 1 matches from j = 0 up to the window length
    assert en.match_length("aaaaa", 1, 1) == (2, "a")
    assert en.match_length("aaaaaa", 3, 3) == (4, "aaa")


@pytest.mark.eval
def test_kontoyiannis_ranks_sources():
    """Thresholds (2000 symbols): fair bits h in [0.75, 1.1] (the estimator is biased low at this length),
    Bernoulli(0.1) h in [0.35, 0.7], a period-2 message h < 0.1; the sliding window gives the same
    ordering."""
    rng = np.random.default_rng(2)
    iid, skew, per = rng.integers(0, 2, 2000), (rng.random(2000) < 0.1).astype(int), [0, 1] * 1000
    h = [en.kontoyiannis(m)["h"] for m in (iid, skew, per)]
    assert 0.75 <= h[0] <= 1.1 and 0.35 <= h[1] <= 0.7 and h[2] < 0.1
    hw = [en.kontoyiannis(m, window=250)["h"] for m in (iid, skew, per)]
    assert hw[0] > hw[1] > hw[2]


@pytest.mark.case
def test_encodings():
    r = [0.3, -0.1, 0.0, 0.2, np.nan, -0.4]
    assert en.encode_binary(r).tolist() == [1, 0, 1, 0]
    assert en.encode_binary(r, zero="keep").tolist() == [2, 0, 1, 2, 0]
    edges = en.quantile_edges(np.linspace(0, 1, 101), 4)
    assert edges.tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert en.encode_quantile([0.1, 0.3, 0.6, 0.9, 0.25], edges).tolist() == [0, 1, 2, 3, 1]
    assert en.encode_sigma([-1.0, -0.45, 0.05, 0.9, 5.0], 0.5, -1.0, 1.0).tolist() == [0, 1, 2, 3, 3]
    assert en.as_message([0, 1, 10, 35]) == "01az"


@pytest.mark.case
def test_kontoyiannis_hand_case():
    # i = 1: "1" has no match in "0" -> L = 1; i = 2: "01" matches at 0 -> L = 3
    out = en.kontoyiannis("0101")
    h = (np.log2(2) / 1 + np.log2(3) / 3) / 2
    assert out["num"] == 2 and out["h"] == pytest.approx(h)
    assert out["r"] == pytest.approx(1 - h / np.log2(2))       # 18.2: against the alphabet, 2 letters here
    # sliding window of 1 on "00100": points 1..4, look-back of one symbol each
    w = en.kontoyiannis("00100", window=1)
    lengths = [2, 1, 1, 2]              # "0"|"0" match; "1"|"0" none; "0"|"1" none; "0"|"0" match
    assert w["num"] == 4 and w["h"] == pytest.approx(np.mean([1 / L for L in lengths]))


@pytest.mark.eval
def test_estimator_bias_on_short_messages():
    """Thresholds: plug-in (w = 1) on 40 i.i.d. symbols from 4 equiprobable letters, 3,000 messages, has a mean
    bias within 0.012 bits of the Miller-Madow term -(K - 1) / (2 n ln 2) = -0.054; Kontoyiannis on the same
    source is biased below -0.2 bits; a skewed source's true entropy is its Shannon entropy."""
    out = en.estimator_bias(lambda m: en.plug_in(m, 1), [0.25] * 4, 40, n_sim=3000, rng=0)
    assert out["true"] == pytest.approx(2.0) and out["length"] == 40 and out["n_sim"] == 3000
    assert out["bias"] == pytest.approx(-3 / (80 * np.log(2)), abs=0.012)
    assert out["mean"] == pytest.approx(out["true"] + out["bias"])
    k = en.estimator_bias(lambda m: en.kontoyiannis(m)["h"], [0.25] * 4, 40, n_sim=300, rng=1)
    assert k["bias"] < -0.2
    assert en.estimator_bias(lambda m: en.plug_in(m, 1), [0.1, 0.9], 2000, n_sim=50, rng=2)["true"] == pytest.approx(H01)


@pytest.mark.case
def test_empirical_cdf_against_a_reference_sample():
    ref = [4.0, 1.0, 3.0, 2.0]
    assert en.empirical_cdf(ref, [0.0, 2.0, 2.5, 5.0, np.nan]).tolist()[:4] == [0.0, 0.5, 0.5, 1.0]
    assert np.isnan(en.empirical_cdf(ref, [np.nan])[0])


@pytest.mark.case
def test_buy_share_codes_for_order_flow_entropy():
    # 18.8.4: the buy share of each volume bar, quantised by quantiles fitted on a reference sample
    share = en.buy_share([3.0, 0.0, 5.0, 0.0], [1.0, 2.0, 5.0, 0.0])
    assert share[:3].tolist() == [0.75, 0.0, 0.5] and np.isnan(share[3])
    edges = en.quantile_edges([0.1, 0.3, 0.6, 0.9], 2)
    assert en.encode_quantile(share, edges).tolist() == [1, 0, 1]


@pytest.mark.case
def test_kontoyiannis_reproduces_the_books_worked_messages():
    """18.4's own numbers (docs/afml_sections/ch18.md F18.2): the entropy rate of "11100001" is 0.96 and of
    "01100001" is 0.84, and "10000111" and "10000110" have the SAME rate because the symmetric matching window never
    reads the last bit after the unmatchable run before it. The existing hand case fixes the log2(n + 1) form from the
    snippet; these fix the estimator against the book's published values."""
    k = lambda m: en.kontoyiannis(m)["h"]
    assert k("11100001") == pytest.approx(0.96, abs=0.01)
    assert k("01100001") == pytest.approx(0.84, abs=0.01)
    assert k("10000111") == pytest.approx(k("10000110"))
    assert k("11100001") > k("01100001")          # the book's point: reverse the message and the last bits count


@pytest.mark.case
def test_redundancy_is_measured_against_the_alphabet_not_the_message_length():
    """18.2 defines R[X] = 1 - H[X] / log2(||A||) over the alphabet size (docs/afml_sections/ch18.md F18.1). On a
    random 4-letter message of 40 symbols the alphabet form reads about 0.33 where the message-length form read
    0.75."""
    rng = np.random.default_rng(0)
    msg = "".join("0123"[i] for i in rng.integers(0, 4, 40))
    out = en.kontoyiannis(msg)
    assert len(set(msg)) == 4
    assert out["r"] == pytest.approx(1 - out["h"] / np.log2(4))
    assert out["r"] == pytest.approx(0.33, abs=0.03)
    assert 1 - out["h"] / np.log2(len(msg)) == pytest.approx(0.75, abs=0.03)     # the superseded reading
    # an incompressible binary message has redundancy near 0; a nearly constant one, near 1
    flat = "".join("01"[i] for i in rng.integers(0, 2, 2000))
    assert en.kontoyiannis(flat)["r"] == pytest.approx(0.0, abs=0.12)
    assert en.kontoyiannis("0" * 2000 + "1")["r"] > 0.9
    assert np.isnan(en.kontoyiannis("0" * 50)["r"])                              # one symbol: no alphabet to beat


@pytest.mark.case
def test_gaussian_entropy_benchmarks_an_estimator_and_implies_a_volatility():
    """18.6: an IID Normal process has H = 1/2 log(2 pi e sigma^2) nats, 1.42 for the standard Normal, which is
    the known answer an estimator is judged against; sigma = exp(H - 1/2) / sqrt(2 pi) inverts it into an
    entropy-implied volatility. Entropy grows by log(k) when the scale is multiplied by k."""
    assert en.gaussian_entropy(1.0) == pytest.approx(1.4189, abs=5e-5)
    for sigma in (0.001, 0.25, 1.0, 40.0):
        assert en.entropy_implied_sigma(en.gaussian_entropy(sigma)) == pytest.approx(sigma)
    assert en.gaussian_entropy(6.0) - en.gaussian_entropy(2.0) == pytest.approx(np.log(3.0))
    # the benchmark in the sense the section means it: a quantile-coded Normal sample, read by the plug-in
    # estimator, must land on the entropy of its own (uniform) code distribution, not on the continuous 1.42
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    code = en.encode_quantile(x, en.quantile_edges(x, 8))
    assert en.plug_in(code, 1) == pytest.approx(3.0, abs=0.01)       # 8 equally likely letters


@pytest.mark.case
def test_mutual_information_is_symmetric_non_negative_and_matches_the_normal_form():
    """18.2: MI = H[X] + H[Y] - H[X, Y], always non-negative, symmetric, and zero exactly under independence.
    For jointly Normal variables it collapses to -1/2 log(1 - rho^2), which is what ties it to Pearson's rho;
    the check is against the differential entropies of the same pair, 2 h(X) - h(X, Y) with |Sigma| = 1 - rho^2."""
    margin_x, margin_y = np.array([0.3, 0.7]), np.array([0.2, 0.5, 0.3])
    assert en.mutual_information(np.outer(margin_x, margin_y)) == pytest.approx(0.0, abs=1e-12)
    joint = np.array([[0.30, 0.00], [0.00, 0.70]])                  # Y determines X
    assert en.mutual_information(joint) == pytest.approx(en.source_entropy([0.3, 0.7]))
    assert en.mutual_information(joint.T) == pytest.approx(en.mutual_information(joint))
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.random((3, 4))
        assert en.mutual_information(p) >= -1e-12
    for rho in (0.0, 0.3, -0.6, 0.95):
        direct = 2 * en.gaussian_entropy(1.0) - 0.5 * np.log((2 * np.pi * np.e) ** 2 * (1 - rho ** 2))
        assert en.gaussian_mutual_information(rho) == pytest.approx(direct)
    assert en.gaussian_mutual_information(0.0) == 0.0


@pytest.mark.case
def test_the_generalized_mean_gives_the_books_special_cases_and_never_falls_with_q():
    """18.7: M_q[x, p] = (sum p_i x_i^q)^(1/q) is the minimum at q -> -inf, the harmonic mean at q = -1, the
    geometric mean at q -> 0, the weighted mean at q = 1, the quadratic mean at q = 2 and the maximum at
    q -> +inf; Jensen's inequality makes it non-decreasing in q."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.full(4, 0.25)
    assert en.generalized_mean(x, w, -np.inf) == 1.0
    assert en.generalized_mean(x, w, -1) == pytest.approx(1 / np.mean(1 / x))
    assert en.generalized_mean(x, w, 0) == pytest.approx(float(np.exp(np.mean(np.log(x)))))
    assert en.generalized_mean(x, w, 1) == pytest.approx(float(x.mean()))
    assert en.generalized_mean(x, w, 2) == pytest.approx(float(np.sqrt((x ** 2).mean())))
    assert en.generalized_mean(x, w, np.inf) == 8.0
    p = np.array([0.5, 0.3, 0.15, 0.05])
    assert en.generalized_mean(x, p, 1) == pytest.approx(float(p @ x))
    qs = np.linspace(-8, 8, 65)
    vals = [en.generalized_mean(x, p, q) for q in qs]
    assert np.all(np.diff(vals) >= -1e-12)


@pytest.mark.case
def test_the_log_effective_number_is_shannons_entropy_in_the_limit_q_to_one():
    """18.7: N_q[p] = 1 / M_(q-1)[p, p] is the effective number of items in p; it is exactly k whatever q when
    the weight is spread evenly over k of them, it never rises with q, and log(N_q) -> H[p] as q -> 1, which is
    the sense in which entropy measures diversity."""
    for k in (1, 3, 10):
        p = np.zeros(12)
        p[:k] = 1.0 / k
        assert all(en.effective_number(p, q) == pytest.approx(float(k)) for q in (0.5, 1 + 1e-9, 2, 5))
    p = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.log2(en.effective_number(p, 1 + 1e-9)) == pytest.approx(en.source_entropy(p), abs=1e-6)
    # 18.7 writes the same entropy the other way round: H[p] = -log(M_q[p, p]) as q -> 0, the weighted
    # geometric mean of the probabilities by themselves
    assert -np.log2(en.generalized_mean(p, p, 0)) == pytest.approx(en.source_entropy(p), abs=1e-12)
    numbers = [en.effective_number(p, q) for q in np.linspace(0.05, 8, 40)]
    assert np.all(np.diff(numbers) <= 1e-12)
    assert en.effective_number(p, 1e-9) == pytest.approx(4.0, abs=1e-6)      # q -> 0: the count of non-zero p


@pytest.mark.case
def test_portfolio_concentration_is_zero_on_evenly_spread_principal_component_risk():
    """18.8.3: H = 1 - (1/N) exp(-sum theta log theta) over the risk shares theta of the N principal components.
    Evenly spread risk (theta_i = 1/N, the book's first exercise) gives 0; all the risk in one component gives
    1 - 1/N; and mixing the two, alpha uniform plus (1 - alpha) skewed, moves H monotonically between them."""
    n = 10
    assert en.portfolio_concentration(np.full(n, 1.0 / n)) == pytest.approx(0.0, abs=1e-12)
    assert en.portfolio_concentration([1.0] + [0.0] * (n - 1)) == pytest.approx(1 - 1.0 / n)
    skewed = np.arange(1, n + 1) / (n * (n + 1) / 2)             # theta_i = i / 55 for N = 10
    assert skewed.sum() == pytest.approx(1.0)
    mixed = [en.portfolio_concentration(a / n + (1 - a) * skewed) for a in np.linspace(0, 1, 21)]
    assert mixed[0] == pytest.approx(en.portfolio_concentration(skewed)) and mixed[0] > 0
    assert np.all(np.diff(mixed) < 0) and mixed[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.case
def test_binary_coding_discards_the_magnitude_a_wider_alphabet_keeps():
    """18.5.1: sign coding is natural where |r| is nearly constant, and discards information when |r| ranges
    widely. Two series with identical signs and very different magnitudes code to the SAME binary message, so
    every entropy read off it is the same, while a four-letter quantile coding separates them."""
    rng = np.random.default_rng(11)
    sign = rng.choice([-1.0, 1.0], 2000)
    calm = sign * 0.01                                       # |r| constant: the price-bar case
    wild = sign * rng.lognormal(-4.0, 1.5, 2000)             # |r| over two orders of magnitude
    assert en.encode_binary(calm).tolist() == en.encode_binary(wild).tolist()
    assert en.plug_in(en.encode_binary(calm), 1) == en.plug_in(en.encode_binary(wild), 1)
    edges = en.quantile_edges(wild, 4)
    assert en.plug_in(en.encode_quantile(wild, edges), 1) > en.plug_in(en.encode_binary(wild), 1) + 0.9
    # the magnitude the binary message threw away is recoverable from the four-letter one
    assert len(set(en.encode_quantile(wild, edges).tolist())) == 4
    assert len(set(en.encode_binary(wild).tolist())) == 2


@pytest.mark.eval
def test_a_subordinated_clock_regularises_the_absolute_return_a_time_clock_leaves_heteroscedastic():
    """18.5.1: sampling on a subordinated process (volume bars) regularises |r| and reduces the need for a large
    alphabet. Thresholds: on a tape whose arrival rate switches between regimes 200 times a second apart, the
    tick count per time bar has a coefficient of variation above 2 while a volume bar's is exactly 0, and the
    dispersion of |r| across time bars is more than 10% above the volume bars', which sit within 8% of the
    homoscedastic sqrt(pi/2 - 1) = 0.7555 that a constant-volatility Normal gives."""
    from pmlab.bars.standard import TimeBars, VolumeBars
    from pmlab.events import make_trade

    window, n = 1_000_000, 8000
    rng = np.random.default_rng(0)
    regime = np.repeat(rng.random(n // 200) < 0.5, 200)
    ts = np.cumsum(np.where(regime, rng.exponential(0.005, n), rng.exponential(1.0, n)))
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, n)), 0.02, 0.98)
    trades = [make_trade(window + float(t), window, float(p), 1, 1.0, float(p)) for t, p in zip(ts, px)]

    def bars(builder):
        builder.new_window(window)
        return [b for b in (builder.update(t) for t in trades) if b is not None]

    def dispersion(built):
        r = np.abs(np.diff(np.log([b.close for b in built])))
        ticks = np.array([b.n_ticks for b in built], dtype=float)
        return float(r.std() / r.mean()), float(ticks.std() / ticks.mean())

    time_cv, time_ticks = dispersion(bars(TimeBars(20.0)))
    vol_cv, vol_ticks = dispersion(bars(VolumeBars(40.0)))
    assert time_ticks > 2.0 and vol_ticks == 0.0             # the volume clock fixes the sample per bar
    assert vol_cv == pytest.approx(np.sqrt(np.pi / 2 - 1), rel=0.08)
    assert time_cv > 1.10 * vol_cv


@pytest.mark.case
def test_quantile_codes_hold_equal_counts_while_the_ranges_they_span_differ():
    """18.5.2: the same number of observations lands in each letter in sample, but a letter's share of the
    return RANGE differs from letter to letter -- the tails span many times the width of the middle bins."""
    rng = np.random.default_rng(3)
    r = rng.standard_normal(20_000)
    edges = en.quantile_edges(r, 8)
    counts = np.bincount(en.encode_quantile(r, edges), minlength=8)
    assert counts.tolist() == [2500] * 8                     # uniform in sample, exactly
    widths = np.diff(np.concatenate([[r.min()], edges, [r.max()]]))
    assert widths.max() > 5 * widths.min()                   # the outer letters cover far more of the range
    assert np.argmax(widths) in (0, 7)


@pytest.mark.case
def test_sigma_coding_reads_less_entropy_than_quantile_coding_of_the_same_returns():
    """18.5.3 against 18.5.2: equal-width letters are not equally likely, so entropy reads lower than under
    equal-frequency letters, which sit at the alphabet's ceiling. Same sample, same alphabet size."""
    rng = np.random.default_rng(4)
    r = rng.standard_normal(20_000)
    lo, hi = float(np.percentile(r, 1)), float(np.percentile(r, 99))
    for k in (4, 8):
        quantile = en.plug_in(en.encode_quantile(r, en.quantile_edges(r, k)), 1)
        sigma = en.plug_in(en.encode_sigma(r, (hi - lo) / k, lo, hi), 1)
        assert quantile == pytest.approx(np.log2(k), abs=0.01)       # equal frequencies: the ceiling
        assert sigma < quantile - 0.1                                 # equal widths: below it
        assert len(set(en.encode_sigma(r, (hi - lo) / k, lo, hi).tolist())) == k


@pytest.mark.eval
def test_a_small_alphabet_discards_information_and_reads_a_lower_entropy():
    """18.6 and Figure 18.1: quantile-code a standard Normal sample into 2, 5, 7 and 10 letters, estimate the
    entropy of 200 messages of length 100 with Kontoyiannis, and compare with the 1.4189 nats the Gaussian
    closed form gives. Thresholds: the estimate rises with every step up in alphabet size; at 2 letters it is
    below three quarters of the true value (information discarded); at 10 letters it is within 20% of it."""
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(100_000)
    true = en.gaussian_entropy(1.0)
    nats = []
    for k in (2, 5, 7, 10):
        edges = en.quantile_edges(reference, k)
        h = [en.kontoyiannis(en.encode_quantile(rng.standard_normal(100), edges))["h"] for _ in range(200)]
        nats.append(float(np.mean(h)) * np.log(2))
    assert np.all(np.diff(nats) > 0.1)
    assert nats[0] < 0.75 * true
    assert nats[-1] == pytest.approx(true, rel=0.2)


@pytest.mark.eval
def test_an_unpredictable_price_string_is_incompressible_and_a_patterned_one_is_not():
    """18.8.1: a martingale leaves no discernible pattern, so its sign message is incompressible -- entropy at
    the alphabet's ceiling and redundancy near 0. A price whose increments persist is redundant, hence
    compressible, hence low entropy: the 'compressed market' the section calls inefficient. Thresholds (4,000
    increments): the martingale reads h > 0.85 bits and R < 0.15, the persistent one h < 0.7 and R > 0.3."""
    rng = np.random.default_rng(5)
    martingale = rng.choice([-1.0, 1.0], 4000)
    persistent = [1.0]
    for _ in range(3999):
        persistent.append(persistent[-1] if rng.random() < 0.9 else -persistent[-1])
    efficient = en.kontoyiannis(en.encode_binary(martingale))
    inefficient = en.kontoyiannis(en.encode_binary(np.array(persistent)))
    assert efficient["h"] > 0.85 and efficient["r"] < 0.15
    assert inefficient["h"] < 0.7 and inefficient["r"] > 0.3
    assert en.lempel_ziv(en.encode_binary(martingale))[1] > en.lempel_ziv(en.encode_binary(np.array(persistent)))[1]
    # the plug-in estimator sees the same thing one word later: pairs are uniform under a martingale
    assert en.plug_in(en.encode_binary(martingale), 2) == pytest.approx(1.0, abs=0.02)
    assert en.plug_in(en.encode_binary(np.array(persistent)), 2) < 0.8


@pytest.mark.case
def test_principal_component_risk_shares_sum_to_one_and_reproduce_the_portfolio_variance():
    """18.8.3's first three steps on a covariance matrix and an allocation, with no portfolio required: the
    loadings are W' w, the contributions f_i^2 Lambda_i sum to the portfolio variance w' V w, and the shares
    theta sum to 1 with each in [0, 1]. On a diagonal covariance the eigenvectors are the assets themselves,
    so theta is the hand formula w_i^2 v_i over its sum."""
    v = np.diag([0.04, 0.01, 0.25])
    w = np.array([0.5, -0.3, 0.2])
    out = en.principal_component_risk(v, w)
    hand = (w ** 2 * np.diag(v)) / float(w ** 2 @ np.diag(v))
    assert np.allclose(np.sort(out["theta"]), np.sort(hand))
    assert out["contribution"].sum() == pytest.approx(out["variance"])
    assert out["variance"] == pytest.approx(float(w @ v @ w))
    rng = np.random.default_rng(6)
    for _ in range(20):
        a = rng.standard_normal((6, 6))
        cov, weights = a @ a.T / 6, rng.standard_normal(6)
        out = en.principal_component_risk(cov, weights)
        assert out["theta"].sum() == pytest.approx(1.0)
        assert (out["theta"] >= -1e-12).all() and (out["theta"] <= 1 + 1e-12).all()
        assert np.allclose(cov @ out["eigenvectors"], out["eigenvectors"] * out["eigenvalues"])
    # equal risk over N independent assets is the evenly spread case concentration reads as 0
    n = 10
    even = en.principal_component_risk(np.eye(n), np.full(n, 1.0 / n))
    assert np.allclose(even["theta"], 1.0 / n)
    assert en.portfolio_concentration(even["theta"]) == pytest.approx(0.0, abs=1e-12)
    concentrated = en.principal_component_risk(np.diag([1.0] + [1e-8] * (n - 1)), np.full(n, 1.0 / n))
    assert en.portfolio_concentration(concentrated["theta"]) > 0.85
