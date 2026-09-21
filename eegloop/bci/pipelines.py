"""Three decoders as scikit-learn pipelines, built lazily from the ``decode`` extra.

They mirror ``notebooks/_shared/helpers_l7.py``'s families by name so ``nb-7-15`` can report the
notebook's number and the library's side by side. One stated divergence: CSP here is
``pyriemann.spatialfilters.CSP`` on covariance matrices rather than ``mne.decoding.CSP`` on epochs,
so that the ``decode`` extra needs no MNE. It is the same generalised eigenproblem with a different
implementation, and the notebook prints both.

Nothing in this module runs at apply time. A fitted pipeline is *frozen* (``eegloop.bci.decoder``)
into arrays that replay it in numpy, exactly; these builders exist to be fitted once.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["PIPELINES", "PIPELINE_NOTES", "csp_lda", "riemann_ts", "xdawn_lda", "make_pipeline"]

PIPELINES: tuple[str, ...] = ("csp_lda", "riemann_ts", "xdawn_lda")

PIPELINE_NOTES: dict[str, str] = {
    "csp_lda": "Covariance (OAS) → CSP spatial filters → log-variance → LDA: the classical motor-imagery baseline. "
               "CSP is pyriemann.spatialfilters.CSP on covariance matrices, not mne.decoding.CSP as in "
               "helpers_l7.make_csp_lda — the same generalised eigenproblem, a different implementation, so the "
               "decode extra needs no MNE; the two solve the same eigenproblem and nb-7-15 uses this one.",
    "riemann_ts": "Covariance (OAS) → tangent space at the Riemannian mean → standardise → logistic regression: "
                  "helpers_l7.make_riemann_ts, unchanged. No spatial filter is fitted.",
    "xdawn_lda": "Xdawn spatial filters → the filtered epoch, flattened → shrinkage LDA: "
                 "helpers_l7.build_p300_pipelines['Xdawn + LDA'], unchanged. Scored by AUC because an oddball "
                 "is imbalanced and accuracy would reward always answering 'standard'.",
}


def _decode() -> None:
    try:
        import pyriemann  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("fitting a decoder needs scikit-learn and pyriemann: pip install 'eegloop[decode]'") from exc


def _flatten(Z: np.ndarray) -> np.ndarray:
    return np.asarray(Z).reshape(len(Z), -1)


def csp_lda(n_components: int = 4, *, estimator: str = "oas") -> Any:
    _decode()
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    return Pipeline([("cov", Covariances(estimator=estimator)),
                     ("csp", CSP(nfilter=int(n_components), log=True)),
                     ("lda", LinearDiscriminantAnalysis())])


def riemann_ts(C: float = 1.0, *, estimator: str = "oas", seed: int = 0) -> Any:
    _decode()
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("cov", Covariances(estimator=estimator)),
                     ("ts", TangentSpace(metric="riemann")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=float(C), random_state=int(seed)))])


def xdawn_lda(n_filter: int = 2) -> Any:
    _decode()
    from pyriemann.spatialfilters import Xdawn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([("xdawn", Xdawn(nfilter=int(n_filter))),
                     ("flat", FunctionTransformer(_flatten)),
                     ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def make_pipeline(kind: str, *, n_components: int = 4, seed: int = 0, estimator: str = "oas") -> Any:
    if kind == "csp_lda":
        return csp_lda(n_components, estimator=estimator)
    if kind == "riemann_ts":
        return riemann_ts(estimator=estimator, seed=seed)
    if kind == "xdawn_lda":
        return xdawn_lda(n_components)
    raise ValueError(f"unknown pipeline {kind!r}; one of {', '.join(PIPELINES)}")
