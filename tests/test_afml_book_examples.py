"""The book's own examples and definitions applied to functions that live in modules owned by other plan nodes
(docs/afml_fidelity.md names these tests as evidence)."""
import numpy as np
import pytest

from pmlab.afml import event_importance as ei


@pytest.mark.case
def test_weighted_kendall_tau_of_the_books_example_is_0_8133():
    """Snippet 8.6: importances .55 .33 .07 .05 against PCA ranks 1 2 4 3; the book prints 0.8133."""
    tau = ei.weighted_tau(np.array([0.55, 0.33, 0.07, 0.05]), np.array([1, 2, 4, 3]))
    assert tau == pytest.approx(0.8133, abs=5e-5)


@pytest.mark.case
def test_orthogonal_features_follow_the_books_construction():
    """8.4.2 / Snippet 8.5: standardise, eigen-decompose Z'Z, sort eigenvalues descending, keep the fewest components
    whose cumulative share reaches 95%; P = ZW has P'P diagonal with the kept eigenvalues (up to the 1/n scale)."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(4000, 3))
    X = f[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]] * rng.uniform(0.5, 2.0, 9) + 0.1 * rng.normal(size=(4000, 9))  # 3 blocks
    fit = ei.pca_fit(X, 0.95)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)                  # the book standardises with pandas' std
    val = np.sort(np.linalg.eigvalsh(Z.T @ Z))[::-1]
    dim = int(np.searchsorted(np.cumsum(val) / val.sum(), 0.95)) + 1
    assert len(fit["eigenvalues"]) == dim == 3
    assert fit["eigenvalues"] / fit["eigenvalues"].sum() == pytest.approx(val[:dim] / val[:dim].sum(), rel=1e-6)
    P = ei.pca_transform(fit, X).astype(float)
    PtP = P.T @ P / len(P)
    assert np.allclose(PtP, np.diag(np.diag(PtP)), atol=1e-3 * PtP.max())
    assert np.diag(PtP) == pytest.approx(fit["eigenvalues"], rel=1e-3)
