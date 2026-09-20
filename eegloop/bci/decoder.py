"""Fit once, freeze, apply: a decoder as arrays, with its cross-validated number and its chance band.

``fit_frozen`` fits one of ``eegloop.bci.pipelines`` with folds **in cue order, never shuffled** --
adjacent trials share slow state (drift, arousal, an electrode drying), and a shuffled split lets a
classifier learn that state and call it the class (``pf-decoding-leakage``). It then fits on every
epoch and *freezes* the pipeline: the covariance estimator, the spatial filters, the tangent-space
reference, the scaler and the linear classifier become numpy arrays that replay the pipeline to
floating-point precision. The frozen decoder is what the loop runs and what ``save`` writes; it
needs no scikit-learn, and it cannot refit.

The chance band is ``helpers_l6.chance_band`` -- the binomial interval a single accuracy must
clear -- reproduced here and held to it by a parity test.

The leakage guard for calibrate-then-apply is **temporal**: :func:`temporal_split` gives fit and
apply index sets that are disjoint *and* non-adjacent, because no shippable four-channel set on the
site has two sessions and a second session is the only better guard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .pipelines import PIPELINES, make_pipeline

__all__ = ["FrozenDecoder", "fit_frozen", "freeze", "chance_interval", "temporal_split", "oas_covariance", "tangent_space"]


# --------------------------------------------------------------------------- #
# The arithmetic a frozen decoder replays -- each one checked against its library form by a test
# --------------------------------------------------------------------------- #
def chance_interval(n: int, p: float = 0.5, conf: float = 0.95) -> tuple[float, float]:
    """Binomial interval around chance for ``n`` trials: the band a single accuracy must clear.
    ``helpers_l6.chance_band``, reproduced."""
    from scipy import stats as sps

    lo, hi = sps.binom.interval(conf, int(n), float(p))
    return float(lo / n), float(hi / n)


def temporal_split(n: int, *, apply_frac: float = 0.4, gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the first epochs, apply on the last, ``gap`` epochs skipped between: disjoint and non-adjacent."""
    n = int(n)
    n_apply = max(1, int(round(n * float(apply_frac))))
    n_fit = n - n_apply - int(gap)
    if n_fit < 2:
        raise ValueError(f"{n} epochs are too few for a temporal split with apply_frac={apply_frac} and gap={gap}")
    return np.arange(n_fit), np.arange(n - n_apply, n)


def scm_covariance(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=2, keepdims=True)
    return np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[2]


def oas_covariance(X: np.ndarray) -> np.ndarray:
    """Oracle approximating shrinkage, per epoch -- ``sklearn.covariance.oas`` as pyriemann calls it."""
    S = scm_covariance(X)
    n_t, p = X.shape[2], X.shape[1]
    tr = np.trace(S, axis1=1, axis2=2)
    mu = tr / p
    alpha = np.mean(S ** 2, axis=(1, 2))
    num = alpha + mu ** 2
    den = (n_t + 1.0) * (alpha - mu ** 2 / p)
    shrink = np.where(den == 0, 1.0, np.minimum(num / np.where(den == 0, 1.0, den), 1.0))
    eye = np.eye(p)[None]
    return (1.0 - shrink)[:, None, None] * S + (shrink * mu)[:, None, None] * eye


_COV = {"oas": oas_covariance, "scm": scm_covariance}


def _spd_fn(M: np.ndarray, fn: Any) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def tangent_space(C: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Upper triangle of ``logm(R^-1/2 C R^-1/2)`` with √2 off the diagonal -- ``pyriemann.tangentspace``."""
    p = C.shape[-1]
    R_isqrt = _spd_fn(np.asarray(reference), lambda w: 1.0 / np.sqrt(w))
    L = _spd_fn(R_isqrt @ C @ R_isqrt, np.log)
    iu = np.triu_indices(p)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[:, iu[0], iu[1]] * coef


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# The frozen decoder
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrozenDecoder:
    kind: str
    classes: tuple[str, ...]
    fs: float
    ch_names: tuple[str, ...]
    tmin_s: float
    tmax_s: float
    epoch_samples: int
    params: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    band: tuple[float, float] | None = None
    filter_delay_samples: float = 0.0

    # -- replay --------------------------------------------------------------------------------------
    def features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None]
        if X.shape[1] != len(self.ch_names) or X.shape[2] != self.epoch_samples:
            raise ValueError(f"expected windows of ({len(self.ch_names)}, {self.epoch_samples}), got {X.shape[1:]}")
        P = self.params
        if self.kind == "csp_lda":
            C = _COV[str(P["estimator"])](X)
            W = P["W"]
            return np.log(np.einsum("ij,njk,ik->ni", W, C, W))
        if self.kind == "riemann_ts":
            C = _COV[str(P["estimator"])](X)
            return (tangent_space(C, P["reference"]) - P["mean"]) / P["scale"]
        if self.kind == "xdawn_lda":
            return np.einsum("fc,nct->nft", P["W"], X).reshape(len(X), -1)
        raise ValueError(f"unknown decoder kind {self.kind!r}")

    def decision(self, X: np.ndarray) -> np.ndarray:
        F = self.features(X)
        return F @ self.params["coef"].T + self.params["intercept"]

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        d = self.decision(X)
        if d.shape[1] == 1:
            s = _expit(d[:, 0])
            return np.column_stack([1.0 - s, s])
        return _softmax(d)

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """Class posteriors for one ``(n_channels, epoch_samples)`` window."""
        return self.predict_proba_batch(np.asarray(window, dtype=float)[None])[0]

    def predict(self, window: np.ndarray) -> str:
        return self.classes[int(np.argmax(self.predict_proba(window)))]

    # -- what it costs -------------------------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def latency_samples(self) -> float:
        """The window is the delay: a posterior at ``t`` describes ``[t − epoch, t]``, exactly."""
        return float(self.epoch_samples)

    @property
    def latency_note(self) -> str:
        return (f"a window of {self.epoch_samples} samples ({self.epoch_samples / self.fs * 1000:.0f} ms) is the delay; "
                f"the causal filter before it adds {self.filter_delay_samples:g} samples, exact; the decision "
                f"rule's dwell is on top of both and is a choice, not a cost of the decoder")

    # -- files ---------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Arrays and a JSON report in one ``.npz``; nothing executable, nothing pickled."""
        p = Path(path)
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        arrays = {f"p_{k}": np.asarray(v) for k, v in self.params.items()}
        np.savez(p, kind=np.array(self.kind), classes=np.array(self.classes), fs=np.array(self.fs),
                 ch_names=np.array(self.ch_names), tmin_s=np.array(self.tmin_s), tmax_s=np.array(self.tmax_s),
                 epoch_samples=np.array(self.epoch_samples), band=np.array(self.band if self.band else [np.nan, np.nan]),
                 filter_delay_samples=np.array(self.filter_delay_samples),
                 report=np.array(json.dumps(self.report, default=str)), eegloop_decoder=np.array(1), **arrays)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "FrozenDecoder":
        with np.load(Path(path), allow_pickle=False) as z:
            if "eegloop_decoder" not in z.files:
                raise ValueError(f"{Path(path).name} is not an eegloop decoder file")
            params = {k[2:]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k.startswith("p_")}
            band = z["band"]
            return cls(kind=str(z["kind"]), classes=tuple(str(c) for c in z["classes"]), fs=float(z["fs"]),
                       ch_names=tuple(str(c) for c in z["ch_names"]), tmin_s=float(z["tmin_s"]), tmax_s=float(z["tmax_s"]),
                       epoch_samples=int(z["epoch_samples"]), params=params, report=json.loads(str(z["report"])),
                       band=None if np.isnan(band[0]) else (float(band[0]), float(band[1])),
                       filter_delay_samples=float(z["filter_delay_samples"]))

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "classes": list(self.classes), "fs": self.fs, "ch_names": list(self.ch_names),
                "tmin_s": self.tmin_s, "tmax_s": self.tmax_s, "epoch_samples": self.epoch_samples, "band": self.band,
                "filter_delay_samples": self.filter_delay_samples, "latency_samples": self.latency_samples,
                "latency_note": self.latency_note, "report": self.report,
                "params": {k: (list(np.shape(v)) if np.ndim(v) else v) for k, v in self.params.items()}}


# --------------------------------------------------------------------------- #
# Fitting and freezing
# --------------------------------------------------------------------------- #
def freeze(pipeline: Any, kind: str, *, classes: Sequence[str], fs: float, ch_names: Sequence[str], tmin_s: float,
           tmax_s: float, report: dict[str, Any] | None = None, band: tuple[float, float] | None = None,
           filter_delay_samples: float = 0.0) -> FrozenDecoder:
    """The fitted pipeline's arrays, as a :class:`FrozenDecoder`."""
    steps = dict(pipeline.named_steps)
    P: dict[str, Any] = {}
    if kind == "csp_lda":
        P["estimator"] = str(steps["cov"].estimator)
        P["W"] = np.asarray(steps["csp"].filters_, dtype=float)
        clf = steps["lda"]
    elif kind == "riemann_ts":
        P["estimator"] = str(steps["cov"].estimator)
        P["reference"] = np.asarray(steps["ts"].reference_, dtype=float)
        P["mean"] = np.asarray(steps["sc"].mean_, dtype=float)
        P["scale"] = np.asarray(steps["sc"].scale_, dtype=float)
        clf = steps["lr"]
    elif kind == "xdawn_lda":
        P["W"] = np.asarray(steps["xdawn"].filters_, dtype=float)
        clf = steps["lda"]
    else:
        raise ValueError(f"unknown pipeline {kind!r}; one of {', '.join(PIPELINES)}")
    if P.get("estimator", "oas") not in _COV:
        raise ValueError(f"covariance estimator {P['estimator']!r} has no frozen form; use 'oas' or 'scm'")
    P["coef"] = np.atleast_2d(np.asarray(clf.coef_, dtype=float))
    P["intercept"] = np.atleast_1d(np.asarray(clf.intercept_, dtype=float))
    n = int(round((float(tmax_s) - float(tmin_s)) * float(fs)))
    return FrozenDecoder(kind=kind, classes=tuple(str(c) for c in classes), fs=float(fs), ch_names=tuple(str(c) for c in ch_names),
                         tmin_s=float(tmin_s), tmax_s=float(tmax_s), epoch_samples=n, params=P, report=dict(report or {}),
                         band=None if band is None else (float(band[0]), float(band[1])),
                         filter_delay_samples=float(filter_delay_samples))


def fit_frozen(kind: str, X: np.ndarray, y: np.ndarray, *, classes: Sequence[str], fs: float, ch_names: Sequence[str],
               tmin_s: float, tmax_s: float, folds: int = 5, seed: int = 0, band: tuple[float, float] | None = None,
               filter_delay_samples: float = 0.0, n_components: int = 4, estimator: str = "oas",
               check_tol: float = 1e-6) -> FrozenDecoder:
    """Cross-validate in cue order, fit on everything, freeze, and check the frozen replay against the pipeline.

    ``y`` holds class indices into ``classes`` (or the labels themselves). Every class must be present.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.asarray(X, dtype=float)
    classes = tuple(str(c) for c in classes)
    y = np.asarray(y)
    if y.dtype.kind in "US":
        y = np.asarray([classes.index(str(v)) for v in y])
    y = y.astype(int)
    counts = np.bincount(y, minlength=len(classes))
    if (counts == 0).any():
        missing = [c for c, k in zip(classes, counts) if k == 0]
        raise ValueError(f"no epochs for class(es) {', '.join(missing)}: a decoder cannot learn a class it never saw")
    n = len(y)
    k = int(min(folds, counts.min()))
    if k < 2:
        raise ValueError(f"too few epochs per class for cross-validation: {dict(zip(classes, counts.tolist()))}")
    pipe = make_pipeline(kind, n_components=n_components, seed=seed, estimator=estimator)
    cv = StratifiedKFold(n_splits=k, shuffle=False)  # cue order, never shuffled
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc") if len(classes) == 2 else None
    pipe.fit(X, y)
    lo, hi = chance_interval(n, 1.0 / len(classes))
    report = {
        "n_epochs": int(n), "n_per_class": {c: int(v) for c, v in zip(classes, counts)}, "folds": k,
        "cv_accuracy": float(acc.mean()), "cv_accuracy_folds": [float(a) for a in acc],
        "cv_auc": None if auc is None else float(auc.mean()),
        "chance": {"p": 1.0 / len(classes), "band_95": [lo, hi], "n": int(n),
                   "note": "binomial 95 % interval around chance for n epochs (helpers_l6.chance_band); "
                           "an accuracy inside it is not evidence of decoding"},
        "clears_chance": bool(acc.mean() > hi),
        "cv_note": "stratified folds in cue order, never shuffled: adjacent epochs share slow state and a shuffled "
                   "split would let the classifier learn it (pf-decoding-leakage)",
        "pipeline": kind, "n_components": int(n_components), "estimator": estimator, "seed": int(seed),
    }
    frozen = freeze(pipe, kind, classes=classes, fs=fs, ch_names=ch_names, tmin_s=tmin_s, tmax_s=tmax_s, report=report,
                    band=band, filter_delay_samples=filter_delay_samples)
    replay = frozen.predict_proba_batch(X)
    ref = pipe.predict_proba(X)
    err = float(np.abs(replay - ref).max())
    if err > check_tol:
        raise RuntimeError(f"the frozen replay diverges from the fitted pipeline by {err:.2e}: a library convention changed")
    frozen.report["frozen_replay_max_abs_err"] = err
    return frozen
