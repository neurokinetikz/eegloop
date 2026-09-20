"""Did the participant learn? A valid test, beside the invalid one that reports usually run.

``naive_trend`` is ``helpers_l7.within_session_trend`` ported verbatim and kept for one reason: it
is the analysis an uncontrolled neurofeedback report runs, and it is invalid -- the per-block values
are strongly autocorrelated, so the ordinary-least-squares standard error is far too small and the
p-value is anti-conservative. ``nb-7-3`` measured its false-positive rate on replayed data where
learning is impossible by construction: 8 of 12 sessions "significant" against a nominal 0.6.

``learning_test`` is the alternative the library ships: one number per session (the change from
its first half to its second, as a fraction of the first), then a sign-flip permutation test across
sessions, so the unit of inference is the session and the autocorrelation inside a session cannot
inflate anything. It answers only "did the targeted signal move"; whether an outcome changed is a
separate claim with separate evidence (L7.3), and ``crednf_report`` keeps the two apart.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .feedback.sham import CREDNF_ITEMS

__all__ = ["naive_trend", "session_change", "learning_test", "crednf_report"]


def naive_trend(values: Sequence[float], times_s: Sequence[float]) -> dict[str, Any]:
    """The naive "did it go up during the session?" test -- INVALID here, and labelled so.

    OLS slope of the feedback on time; the per-block values are strongly autocorrelated, so the
    standard error is far too small and the p-value anti-conservative. Ported from
    ``helpers_l7.within_session_trend`` so the course can show what it does wrong.
    """
    from scipy import stats as sps

    v = np.asarray(values, dtype=float)
    t = np.asarray(times_s, dtype=float)
    res = sps.linregress(t, v)
    lag1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else float("nan")
    half = len(v) // 2
    return {"slope_per_min": float(res.slope) * 60.0, "p": float(res.pvalue), "r": float(res.rvalue),
            "n_blocks": int(len(v)), "first_half": float(v[:half].mean()), "second_half": float(v[half:].mean()),
            "change_pct": float(100 * (v[half:].mean() - v[:half].mean()) / v[:half].mean()),
            "lag1_autocorrelation": lag1,
            "valid": False,
            "why_invalid": "per-block values are autocorrelated; the OLS standard error is too small and p is anti-conservative"}


def session_change(values: Sequence[float]) -> float:
    """One number for a session: second-half mean minus first-half mean, as a fraction of the first."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float("nan")
    half = v.size // 2
    a, b = v[:half].mean(), v[half:].mean()
    return float((b - a) / (abs(a) + 1e-12))


def learning_test(sessions: Sequence[Sequence[float]], *, n_perm: int = 5000, seed: int = 0,
                  alternative: str = "greater") -> dict[str, Any]:
    """A sign-flip permutation test on per-session change scores. The session is the unit.

    ``alternative='greater'`` tests for an increase (an "up-training" protocol); ``'less'`` for a
    decrease; ``'two-sided'`` for either. With ``n`` sessions there are ``2**n`` sign patterns, so
    below ~13 sessions the test enumerates them exactly and ``n_perm`` is ignored.
    """
    scores = np.asarray([session_change(s) for s in sessions], dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n < 2:
        return {"n_sessions": int(n), "mean_change": float(scores.mean()) if n else float("nan"), "p": float("nan"),
                "valid": True, "note": "fewer than two sessions: no test is possible; report the change and say so"}
    obs = float(scores.mean())
    if n <= 12:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * n))).T.reshape(-1, n)
        exact = True
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice([-1.0, 1.0], size=(int(n_perm), n))
        exact = False
    null = (signs * np.abs(scores)[None, :]).mean(axis=1)
    if alternative == "greater":
        p = float(np.mean(null >= obs - 1e-15))
    elif alternative == "less":
        p = float(np.mean(null <= obs + 1e-15))
    else:
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
    return {"n_sessions": int(n), "mean_change": obs, "per_session_change": scores.tolist(), "p": p,
            "alternative": alternative, "exact": exact, "n_null": int(signs.shape[0]), "valid": True,
            "statistic": "second-half minus first-half mean, as a fraction of the first half, per session",
            "note": "the session is the unit of inference; within-session autocorrelation cannot inflate this"}


def crednf_report(protocol: dict[str, Any], session: dict[str, Any], *, sessions_tested: dict[str, Any] | None = None,
                  prereg_committed: bool | None = None) -> list[dict[str, Any]]:
    """The six CRED-nf items, each marked satisfied / not satisfiable alone / unsatisfied, with why.

    Some items a single session can satisfy from its own log; some need something outside it -- a
    commit that precedes the data, a second party, several sessions -- and the report says which,
    rather than ticking a box the log cannot vouch for.
    """
    sham_mode = (protocol.get("sham") or {}).get("mode", "veridical")
    items: list[dict[str, Any]] = []
    # 1 pre-registration
    if prereg_committed is True:
        items.append({"item": CREDNF_ITEMS[0], "status": "satisfied", "evidence": "protocol hash committed before the first session log"})
    else:
        items.append({"item": CREDNF_ITEMS[0], "status": "not-satisfiable-alone",
                      "evidence": f"the log carries the protocol hash {session.get('protocol_hash')!r}; whether it was committed BEFORE the data is a fact about version control, not about this file"})
    # 2 control condition
    if sham_mode != "veridical":
        items.append({"item": CREDNF_ITEMS[1], "status": "satisfied", "evidence": f"this session ran under {sham_mode!r}"})
    else:
        items.append({"item": CREDNF_ITEMS[1], "status": "unsatisfied", "evidence": "this session is veridical; a control condition is a separate session or group"})
    # 3 blinding
    if session.get("sealed_sham"):
        items.append({"item": CREDNF_ITEMS[2], "status": "not-satisfiable-alone",
                      "evidence": "the sham mode is sealed in the log (participant blinding is possible); experimenter and analyst blinding are about people, not files"})
    else:
        items.append({"item": CREDNF_ITEMS[2], "status": "unsatisfied", "evidence": "no sealed mode in the log"})
    # 4 the signal in full
    if session.get("signal") and session.get("budget"):
        items.append({"item": CREDNF_ITEMS[3], "status": "satisfied", "evidence": session["signal"]})
    else:
        items.append({"item": CREDNF_ITEMS[3], "status": "unsatisfied", "evidence": "the log lacks the signal description or the budget"})
    # 5 learning, separate from outcome
    if sessions_tested is not None and sessions_tested.get("valid"):
        items.append({"item": CREDNF_ITEMS[4], "status": "satisfied",
                      "evidence": f"learning_test over {sessions_tested.get('n_sessions')} sessions: mean change {sessions_tested.get('mean_change'):.3g}, p = {sessions_tested.get('p'):.3g}; this says nothing about any outcome"})
    else:
        items.append({"item": CREDNF_ITEMS[4], "status": "not-satisfiable-alone",
                      "evidence": "needs several sessions and learning_test; one session cannot show learning"})
    # 6 data available
    if session.get("files"):
        items.append({"item": CREDNF_ITEMS[5], "status": "satisfied", "evidence": f"per-block series and events in {', '.join(session['files'].values())}"})
    else:
        items.append({"item": CREDNF_ITEMS[5], "status": "unsatisfied", "evidence": "no session files recorded"})
    return items
