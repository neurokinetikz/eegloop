"""The valid learning test beside the invalid one, and the CRED-nf report that keeps claims apart."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop.analysis import crednf_report, learning_test, naive_trend, session_change


def _ar1(rng, n=300, rho=0.7, trend=0.0):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal()
    return 10.0 + x + trend * np.arange(n) / n


def test_naive_trend_is_labelled_invalid_and_is_anticonservative_on_null_data(rng):
    p_values = []
    for _ in range(40):
        v = _ar1(rng)
        r = naive_trend(v, np.arange(v.size) * 0.2)
        assert r["valid"] is False and "autocorrelated" in r["why_invalid"]
        p_values.append(r["p"])
    # with no learning possible, far more than 5 % of sessions come out "significant"
    assert np.mean(np.asarray(p_values) < 0.05) > 0.15


def test_learning_test_is_null_on_null_sessions_and_detects_a_planted_increase(rng):
    null = learning_test([_ar1(rng) for _ in range(10)], seed=1)
    assert null["valid"] and null["exact"] and null["n_null"] == 2 ** 10 and null["n_sessions"] == 10
    assert null["p"] > 0.05
    up = learning_test([_ar1(rng, trend=3.0) for _ in range(10)], seed=1)
    assert up["mean_change"] > 0.1 and up["p"] < 0.01
    two = learning_test([_ar1(rng, trend=3.0) for _ in range(10)], alternative="two-sided")
    assert two["p"] < 0.02
    approx = learning_test([_ar1(rng, trend=3.0) for _ in range(14)], n_perm=2000, seed=2)
    assert approx["exact"] is False and approx["n_null"] == 2000 and approx["p"] < 0.01


def test_session_change_and_degenerate_inputs():
    assert session_change([1, 1, 2, 2]) == pytest.approx(1.0)
    assert np.isnan(session_change([1.0, 2.0]))
    one = learning_test([[1, 1, 2, 2]])
    assert np.isnan(one["p"]) and "no test is possible" in one["note"]


def test_crednf_report_marks_what_a_single_session_cannot_vouch_for():
    proto = {"sham": {"mode": "veridical"}}
    session = {"protocol_hash": "abc", "sealed_sham": "deadbeef", "signal": "8–12 Hz …", "budget": {"total_ms": 610.0},
               "files": {"session": "session.json", "events": "events.jsonl", "signal": "signal.npz"}}
    items = crednf_report(proto, session)
    assert len(items) == 6
    statuses = [i["status"] for i in items]
    assert statuses == ["not-satisfiable-alone", "unsatisfied", "not-satisfiable-alone", "satisfied", "not-satisfiable-alone", "satisfied"]
    sham = crednf_report({"sham": {"mode": "sham-yoked"}}, session, prereg_committed=True,
                         sessions_tested={"valid": True, "n_sessions": 8, "mean_change": 0.2, "p": 0.01})
    assert [i["status"] for i in sham][:2] == ["satisfied", "satisfied"] and sham[4]["status"] == "satisfied"
    assert "nothing about any outcome" in sham[4]["evidence"]
