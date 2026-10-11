"""
📐 LEVEL-1 МОДЕЛІ (sklearn) + СТЕКІНГ + ПОРОГИ QWK.

Усе рахується як Out-Of-Fold (OOF): кожна модель отримує чесні прогнози для train та
усереднений прогноз для test (сума по фолдах = навчання на ВСІХ даних, вимога п.12 ТЗ).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

logger = logging.getLogger("hw_fp.stacking")


# ------------------------------------------------------------------------------
# Метрика та оптимальні пороги
# ------------------------------------------------------------------------------
def qwk(y_true, y_pred, lo: int | None = None, hi: int | None = None) -> float:
    """Quadratic Weighted Kappa (швидка векторна реалізація, збігається з sklearn)."""
    yt, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    lo = int(min(yt.min(), yp.min())) if lo is None else lo
    hi = int(max(yt.max(), yp.max())) if hi is None else hi
    k = hi - lo + 1
    O = np.bincount((yt - lo) * k + (yp - lo), minlength=k * k).reshape(k, k).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / max(k - 1, 1) ** 2
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else 0.0


class OptimizedRounder:
    """Пошук порогів (hi-lo штук) під QWK: Nelder-Mead з кількома рестартами (QWK — ступінчаста функція)."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.thresholds_ = lo + 0.5 + np.arange(hi - lo)
        self.score_ = float("nan")

    def predict(self, x, thresholds=None) -> np.ndarray:
        thr = np.sort(self.thresholds_ if thresholds is None else thresholds)
        return np.digitize(np.asarray(x), thr) + self.lo

    def fit(self, x, y, restarts: int = 8, seed: int = 42) -> "OptimizedRounder":
        x, y = np.asarray(x, float), np.asarray(y)
        rng = np.random.default_rng(seed)
        base = self.lo + 0.5 + np.arange(self.hi - self.lo)

        def loss(t):
            return -qwk(y, self.predict(x, t), self.lo, self.hi)

        best_t, best_s = base, loss(base)
        for r in range(restarts):
            x0 = base + (rng.normal(0, 0.15, base.shape) if r else 0.0)
            res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 1500})
            if res.fun < best_s:
                best_t, best_s = np.sort(res.x), res.fun
        self.thresholds_, self.score_ = np.sort(best_t), -best_s
        return self


def honest_threshold_qwk(x, y, lo: int, hi: int, reps: int = 4, seed: int = 0) -> Tuple[float, float]:
    """Пороги підбираються на одній половині OOF, а QWK міряється на іншій → без оптимізму від 3 параметрів."""
    x, y = np.asarray(x), np.asarray(y)
    rng, sc = np.random.default_rng(seed), []
    for _ in range(reps):
        idx = rng.permutation(len(x))
        a, b = idx[: len(x) // 2], idx[len(x) // 2:]
        for tr, te in ((a, b), (b, a)):
            r = OptimizedRounder(lo, hi).fit(x[tr], y[tr], restarts=4)
            sc.append(qwk(y[te], r.predict(x[te]), lo, hi))
    return float(np.mean(sc)), float(np.std(sc))


# ------------------------------------------------------------------------------
# Каркас OOF
# ------------------------------------------------------------------------------
_FOLD_HOOK = None


def set_fold_hook(fn) -> None:
    """fn(k, K) викликається після кожного фолду будь-якої OOF-моделі (для прогрес-бару в ноутбуці)."""
    global _FOLD_HOOK
    _FOLD_HOOK = fn


def oof_loop(fn: Callable, folds: np.ndarray, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    K = int(folds.max()) + 1
    oof, test = np.zeros(len(folds)), np.zeros(n_test)
    for k in range(K):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        pv, pt = fn(tr, va)
        oof[va], test = pv, test + pt / K
        if _FOLD_HOOK:
            _FOLD_HOOK(k + 1, K)
    return oof, test


# ------------------------------------------------------------------------------
# TF-IDF (класичний текстовий сигнал)
# ------------------------------------------------------------------------------
def build_tfidf(comp) -> sparse.csr_matrix:
    text = comp.full_text
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000, sublinear_tf=True)
    return sparse.hstack([w.fit_transform(text), c.fit_transform(text)]).tocsr()


def tfidf_svd(X: sparse.csr_matrix, n: int = 128, seed: int = 42) -> np.ndarray:
    n = min(n, X.shape[1] - 1)
    return TruncatedSVD(n_components=n, random_state=seed).fit_transform(X).astype(np.float32)


def ridge_sparse_oof(X, y, folds, Xte, alphas=(0.5, 1.5, 4.0)):
    best = None
    for a in alphas:
        def fn(tr, va, a=a):
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            return m.predict(X[va]), m.predict(Xte)
        oof, te = oof_loop(fn, folds, Xte.shape[0])
        rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, a, oof, te)
    logger.info(f"TF-IDF Ridge: alpha={best[1]}, RMSE={best[0]:.4f}")
    return best[2], best[3]


# ------------------------------------------------------------------------------
# Dense-ембединги та дерева
# ------------------------------------------------------------------------------
def ridge_dense_oof(X, y, folds, Xte):
    def fn(tr, va):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 4.5, 8)))
        m.fit(X[tr], y[tr])
        return m.predict(X[va]), m.predict(Xte)
    return oof_loop(fn, folds, Xte.shape[0])


def pca_block(X: np.ndarray, n: int = 24, seed: int = 42) -> np.ndarray:
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=min(n, X.shape[1]), random_state=seed).fit_transform(Z).astype(np.float32)


def hgb_oof(X, y, folds, Xte, seed: int = 42):
    X, Xte = np.asarray(X, np.float32), np.asarray(Xte, np.float32)

    def fn(tr, va):
        m = HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=800, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=2.0, early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=50, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        return m.predict(X[va]), m.predict(Xte)
    return oof_loop(fn, folds, Xte.shape[0])


# ------------------------------------------------------------------------------
# Звіти та стекінг
# ------------------------------------------------------------------------------
def model_report(oofs: Dict[str, np.ndarray], y, lo: int, hi: int) -> pd.DataFrame:
    rows = []
    for n, o in oofs.items():
        o = np.clip(o, lo, hi)
        rows.append({
            "модель": n,
            "RMSE": round(float(np.sqrt(np.mean((o - y) ** 2))), 4),
            "QWK (округл.)": round(qwk(y, np.clip(np.rint(o), lo, hi), lo, hi), 4),
            "QWK (пороги)": round(OptimizedRounder(lo, hi).fit(o, y, restarts=3).score_, 4),
        })
    return pd.DataFrame(rows).sort_values("QWK (пороги)", ascending=False).reset_index(drop=True)


def stack(oofs: Dict[str, np.ndarray], tests: Dict[str, np.ndarray], y, lo: int, hi: int, seed: int = 123) -> dict:
    names = list(oofs)
    A = np.column_stack([oofs[n] for n in names])
    T = np.column_stack([tests[n] for n in names])
    meta = LinearRegression(positive=True)                       # невід'ємні ваги → стабільний бленд
    oof_meta = cross_val_predict(meta, A, y, cv=KFold(5, shuffle=True, random_state=seed))
    meta.fit(A, y)
    test_pred = np.clip(meta.predict(T), lo, hi)
    oof_meta = np.clip(oof_meta, lo, hi)
    rnd = OptimizedRounder(lo, hi).fit(oof_meta, y)
    h_mean, h_std = honest_threshold_qwk(oof_meta, y, lo, hi)
    return {
        "names": names,
        "weights": dict(zip(names, np.round(meta.coef_, 4))),
        "intercept": float(meta.intercept_),
        "oof": oof_meta,
        "test_raw": test_pred,
        "thresholds": rnd.thresholds_,
        "qwk_oof_thr": rnd.score_,
        "qwk_honest": (h_mean, h_std),
        "qwk_round_only": qwk(y, np.clip(np.rint(oof_meta), lo, hi), lo, hi),
        "test_labels": rnd.predict(test_pred),
    }


# ------------------------------------------------------------------------------
# kNN по ембедингах / TF-IDF: ловить шаблонні описи, ті самі притулки/фотосесії (near-duplicates)
# ------------------------------------------------------------------------------
def _topk(Q, R, k: int):
    S = Q @ R.T
    if sparse.issparse(S):
        S = S.toarray()
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, 1)
    order = np.argsort(-sims, axis=1)
    return np.take_along_axis(sims, order, 1), np.take_along_axis(idx, order, 1)


def _knn_vote(sims, labs, k: int, tau: float) -> np.ndarray:
    s, l = sims[:, :k], labs[:, :k]
    w = np.exp((s - s.max(1, keepdims=True)) / tau)
    return (w * l).sum(1) / w.sum(1)


def knn_oof(X, y, folds, Xte, ks=(10, 30), taus=(0.03, 0.1), kmax: int = 50):
    """OOF kNN-регресія за косинусною схожістю; k та температура вибираються за OOF-RMSE."""
    X, Xte = normalize(X), normalize(Xte)
    K = int(folds.max()) + 1
    cache = []
    for f in range(K):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        sv, iv = _topk(X[va], X[tr], kmax)
        st_, it = _topk(Xte, X[tr], kmax)
        cache.append((va, sv, y[tr][iv], st_, y[tr][it]))
        if _FOLD_HOOK:
            _FOLD_HOOK(f + 1, K)
    best = None
    for k in ks:
        for tau in taus:
            oof, test = np.zeros(len(y)), np.zeros(Xte.shape[0])
            for va, sv, lv, st_, lt in cache:
                oof[va] = _knn_vote(sv, lv, k, tau)
                test += _knn_vote(st_, lt, k, tau) / K
            rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, k, tau, oof, test)
    logger.info(f"kNN: k={best[1]}, tau={best[2]}, RMSE={best[0]:.4f}")
    return best[3], best[4]


def neighbor_diagnostics(blocks: Dict[str, object], y, n_train: int, thrs=(0.99, 0.95, 0.90), on_block=None) -> pd.DataFrame:
    """
    Чи є «майже дублікати»? Для кожного блоку: частка train-об'єктів із сусідом у train зі схожістю ≥ thr,
    як часто мітка сусіда збігається, і частка test-об'єктів, що мають такого сусіда в train.
    """
    y = np.asarray(y).astype(int)
    rows = []
    for i, (name, X) in enumerate(blocks.items(), 1):
        Xn = normalize(X)
        Xtr, Xte = Xn[:n_train], Xn[n_train:]
        S = Xtr @ Xtr.T
        S = S.toarray() if sparse.issparse(S) else np.asarray(S)
        np.fill_diagonal(S, -np.inf)
        nn = S.argmax(1)
        nsim = S[np.arange(n_train), nn]
        T = Xte @ Xtr.T
        T = T.toarray() if sparse.issparse(T) else np.asarray(T)
        tsim = T.max(1)
        for thr in thrs:
            m = nsim >= thr
            rows.append({
                "блок": name, "поріг cos": thr,
                "train із сусідом, %": round(100 * float(m.mean()), 1),
                "мітка сусіда збігається, %": round(100 * float((y[m] == y[nn[m]]).mean()), 1) if m.any() else np.nan,
                "test із сусідом у train, %": round(100 * float((tsim >= thr).mean()), 1),
            })
        if on_block:
            on_block(i, len(blocks), name)
    return pd.DataFrame(rows)
