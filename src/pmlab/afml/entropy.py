"""Entropy features (AFML ch.18).

Estimators on a message (a sequence of symbols):
- plug_in: maximum-likelihood (plug-in) entropy rate of words of length w, in bits per symbol (18.3).
  Words are all overlapping substrings of length w; the book's snippet starts at the w-th symbol and
  stops one word short of the end, which only matters for very short messages.
- lempel_ziv: the library of the LZ decomposition, each new phrase being the shortest substring from the
  current position not yet in the library; complexity = library size / message length (18.4).
- match_length and kontoyiannis: the Kontoyiannis (1998) estimator from longest matches with the past,
  averaged over an expanding window (window=None) or a sliding window of fixed length (18.4). h is in
  bits per symbol; redundancy is 18.2's R[X] = 1 - H[X] / log2(||A||), against the log of the ALPHABET
  size (the distinct symbols the message uses), which is the largest entropy a message over that
  alphabet can reach. It read 1 - h / log2(message length) until docs/afml_sections/ch18.md F18.1; on a
  random 40-symbol, 4-letter message the two give 0.33 and 0.75. NaN below two distinct symbols.

Encodings of returns (18.6):
- encode_binary: 1 for a positive return, 0 for a negative one; zero returns are dropped (or kept as a
  third symbol with zero="keep").
- encode_quantile: the index of the quantile bin, with edges from a reference sample so the encoding
  is fitted before the data it encodes.
- encode_sigma: bins of fixed width sigma_step from a reference minimum, clipped to the reference range.

The chapter's closed forms, each with a value that is known before it is computed:
- gaussian_entropy and entropy_implied_sigma: 18.6's 1/2 log(2 pi e sigma^2) in nats (1.4189 for the standard
  Normal) and its inverse, the entropy-implied volatility.
- mutual_information and gaussian_mutual_information: 18.2's H[X] + H[Y] - H[X, Y] from a joint table, and the
  Normal case -1/2 log(1 - rho^2) that ties it to Pearson's correlation.
- generalized_mean, effective_number: 18.7's M_q[x, p] and N_q[p] = 1 / M_(q-1)[p, p]; Shannon's entropy is
  log(N_q) as q -> 1, which is why entropy reads as the log effective number of items in a distribution.
- principal_component_risk and portfolio_concentration: 18.8.3's eigenvalue decomposition, factor loadings and
  per-component risk shares theta, then Meucci's 1 - (1/N) exp(-sum theta log theta) over them.

Checks and features:
- estimator_bias: the mean estimate over messages drawn i.i.d. from known symbol probabilities, against
  their Shannon entropy: the benchmark of 18.6 (a source whose entropy is known) at the message length
  and alphabet actually used, so an estimator's small-sample bias is measured rather than assumed.
- buy_share and empirical_cdf: the order-flow imbalance feature of 18.8.4: the buy share of each
  volume bar, quantile-coded, its entropy, and that entropy's CDF against a reference sample.
"""
import numpy as np

SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def as_message(seq) -> str:
    if isinstance(seq, str):
        return seq
    seq = np.asarray(seq, dtype=int)
    if len(seq) and (seq.min() < 0 or seq.max() >= len(SYMBOLS)):
        raise ValueError(f"symbols must be in [0, {len(SYMBOLS)})")
    return "".join(SYMBOLS[i] for i in seq)


def plug_in(msg, w: int = 1) -> float:
    msg = as_message(msg)
    n = len(msg) - w + 1
    if n <= 0:
        return np.nan
    counts: dict[str, int] = {}
    for i in range(n):
        word = msg[i:i + w]
        counts[word] = counts.get(word, 0) + 1
    p = np.array(list(counts.values()), dtype=float) / n
    return float(-np.sum(p * np.log2(p)) / w)


def lempel_ziv(msg) -> tuple[list[str], float]:
    msg = as_message(msg)
    if not msg:
        return [], np.nan
    lib, seen = [msg[0]], {msg[0]}
    i = 1
    while i < len(msg):
        for j in range(i, len(msg)):
            phrase = msg[i:j + 1]
            if phrase not in seen:
                seen.add(phrase)
                lib.append(phrase)
                break
        i = j + 1
    return lib, len(lib) / len(msg)


def match_length(msg: str, i: int, n: int) -> tuple[int, str]:
    """1 + the length of the longest substring starting at i that also starts somewhere in [i - n, i)
    (matches may run past i)."""
    sub = ""
    for length in range(1, n + 1):
        if i + length > len(msg):
            break
        cand = msg[i:i + length]
        found = False
        for j in range(i - n, i):
            if msg[j:j + length] == cand:
                sub, found = cand, True
                break
        if not found:
            break
    return len(sub) + 1, sub


def kontoyiannis(msg, window: int | None = None) -> dict:
    msg = as_message(msg)
    if len(msg) < 2:
        return {"h": np.nan, "r": np.nan, "num": 0}
    total, num = 0.0, 0
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            length, _ = match_length(msg, i, i)
            total += np.log2(i + 1) / length
        else:
            length, _ = match_length(msg, i, window)
            total += np.log2(window + 1) / length
        num += 1
    h = total / num
    alphabet = len(set(msg))
    r = float(1 - h / np.log2(alphabet)) if alphabet > 1 else np.nan      # 18.2, against the alphabet size
    return {"h": float(h), "r": r, "num": num}


def source_entropy(probs) -> float:
    """Shannon entropy in bits of a distribution over symbols."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log2(p)))


def estimator_bias(estimate, probs, length: int, n_sim: int = 500, rng=None) -> dict:
    """estimate(message) on n_sim i.i.d. messages of `length` symbols drawn with `probs`: mean, sd and bias
    against the source entropy (bits per symbol)."""
    rng = np.random.default_rng(rng)
    p = np.asarray(probs, dtype=float)
    vals = np.array([estimate(rng.choice(len(p), length, p=p / p.sum())) for _ in range(n_sim)], dtype=float)
    true = source_entropy(p)
    mean = float(np.nanmean(vals))
    return {"true": true, "mean": mean, "sd": float(np.nanstd(vals, ddof=1)), "bias": mean - true,
            "relative_bias": (mean - true) / true if true > 0 else np.nan, "length": int(length), "n_sim": int(n_sim)}


def buy_share(buy, sell) -> np.ndarray:
    """Buy volume / total volume per bar; NaN for a bar with no volume."""
    buy, sell = np.asarray(buy, dtype=float), np.asarray(sell, dtype=float)
    total = buy + sell
    return np.divide(buy, total, out=np.full(len(total), np.nan), where=total > 0)


def empirical_cdf(reference, x) -> np.ndarray:
    """Share of the finite reference values <= x; NaN where x is not finite."""
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    x = np.asarray(x, dtype=float)
    out = np.searchsorted(ref, np.where(np.isfinite(x), x, 0.0), side="right") / len(ref)
    return np.where(np.isfinite(x), out, np.nan)


def encode_binary(returns, zero: str = "drop") -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if zero == "drop":
        r = r[r != 0]
        return (r > 0).astype(int)
    return np.where(r > 0, 2, np.where(r < 0, 0, 1))


def quantile_edges(reference, n_bins: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    return np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])


def encode_quantile(returns, edges) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return np.searchsorted(np.asarray(edges), r, side="right")


def encode_sigma(returns, sigma_step: float, lo: float, hi: float) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n_bins = int(np.ceil((hi - lo) / sigma_step))
    return np.clip(np.floor((r - lo) / sigma_step).astype(int), 0, max(n_bins - 1, 0))


def gaussian_entropy(sigma: float) -> float:
    """Differential entropy of an IID Normal process in NATS, 1/2 log(2 pi e sigma^2) (18.6); 1.4189 at sigma = 1,
    the book's benchmark value for an estimator (its estimators return bits, so divide by log(2) to compare)."""
    s = float(sigma)
    return float(0.5 * np.log(2 * np.pi * np.e * s * s)) if s > 0 else -np.inf


def entropy_implied_sigma(h: float) -> float:
    """The entropy-implied volatility of 18.6, sigma = exp(h - 1/2) / sqrt(2 pi), with h in nats: the inverse of
    gaussian_entropy, valid only where returns really are Normal."""
    return float(np.exp(float(h) - 0.5) / np.sqrt(2 * np.pi))


def mutual_information(joint) -> float:
    """MI[X, Y] = H[X] + H[Y] - H[X, Y] in bits from a joint probability table (18.2). Non-negative, symmetric,
    and zero exactly when the table is a product of its margins."""
    p = np.asarray(joint, dtype=float)
    p = p / p.sum()
    return source_entropy(p.sum(axis=1)) + source_entropy(p.sum(axis=0)) - source_entropy(p.ravel())


def gaussian_mutual_information(rho: float) -> float:
    """MI of two jointly Normal variables in NATS, -1/2 log(1 - rho^2) (18.2): the link that makes mutual
    information the nonlinear generalisation of Pearson's correlation."""
    r = float(rho)
    return float(-0.5 * np.log(1 - r * r)) if abs(r) < 1 else np.inf


def generalized_mean(x, p, q: float) -> float:
    """M_q[x, p] = (sum p_i x_i^q)^(1/q), the weighted power mean of 18.7, with the limits the section names:
    q -> -inf the minimum, q -> 0 the geometric mean, q -> +inf the maximum. x must be positive for q <= 0."""
    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    p = p / p.sum()
    keep = p > 0
    x, p = x[keep], p[keep]
    if q <= -np.inf:
        return float(x.min())
    if q >= np.inf:
        return float(x.max())
    if q == 0:
        return float(np.exp(np.sum(p * np.log(x))))
    return float(np.sum(p * x ** q) ** (1.0 / q))


def effective_number(p, q: float) -> float:
    """N_q[p] = 1 / M_(q-1)[p, p], 18.7's effective number (diversity) of a partition: k when the weight is
    spread evenly over k items, whatever q. Shannon's entropy is log(N_q) in the limit q -> 1."""
    return 1.0 / generalized_mean(p, p, q - 1)


def principal_component_risk(cov, weights) -> dict:
    """18.8.3's first three steps, on any covariance matrix and any allocation.

    The eigenvalue decomposition V W = W Lambda; the factor loadings f = W' w; and the share of the risk carried
    by each principal component, theta_i = f_i^2 Lambda_ii / sum_n f_n^2 Lambda_nn. The denominator is the
    portfolio variance w' V w, so the shares sum to 1 and, for a positive semi-definite V, each lies in [0, 1].
    These theta are the argument portfolio_concentration expects.
    """
    v = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    values, vectors = np.linalg.eigh(v)
    loadings = vectors.T @ w
    contribution = loadings ** 2 * values
    total = float(contribution.sum())
    theta = contribution / total if total > 0 else np.full(len(values), np.nan)
    return {"eigenvalues": values, "eigenvectors": vectors, "loadings": loadings,
            "contribution": contribution, "variance": float(w @ v @ w), "theta": theta}


def portfolio_concentration(theta) -> float:
    """Meucci's entropy-inspired concentration of 18.8.3, H = 1 - (1/N) exp(-sum theta_i log theta_i), over the
    shares theta of total risk carried by each of the N principal components. 0 when risk is spread evenly over
    all N, 1 - 1/N when one component carries it all."""
    t = np.asarray(theta, dtype=float)
    t = t / t.sum()
    n = len(t)
    return float(1 - np.exp(np.sum(-t[t > 0] * np.log(t[t > 0]))) / n) if n else np.nan
