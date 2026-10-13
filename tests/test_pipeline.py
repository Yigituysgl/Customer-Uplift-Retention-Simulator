from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import simulate

ROOT = Path(__file__).resolve().parents[1]


def test_simulation_is_randomized_and_reproducible():
    customers = pd.DataFrame({"Risk": np.repeat(["High", "Medium", "Low"], 2000)})
    a = simulate.simulate_campaign(customers, seed=42)
    b = simulate.simulate_campaign(customers, seed=42)
    pd.testing.assert_frame_equal(a, b)

    # Treatment is ~50/50 within every risk group (no assignment by risk).
    rates = a.groupby("Risk")["Treatment"].mean()
    assert ((rates > 0.45) & (rates < 0.55)).all()

    # Difference in means recovers the configured true uplift per group.
    for risk, true_uplift in simulate.TRUE_UPLIFT.items():
        g = a[a["Risk"] == risk]
        observed = g[g.Treatment == 1].Response.mean() - g[g.Treatment == 0].Response.mean()
        assert abs(observed - true_uplift) < 0.05


def test_qini_ranks_perfect_score_above_random():
    rng = np.random.default_rng(0)
    n = 4000
    tau = rng.choice([0.0, 0.4], size=n)
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.05 + tau * t)

    x, q = simulate.qini_curve(tau, t, y)
    assert x[0] == 0 and x[-1] == 1 and q[0] == 0
    random_area = q[-1] / 2
    assert simulate.area(x, q) > random_area
    # A score unrelated to uplift should be close to random targeting.
    _, q_noise = simulate.qini_curve(rng.random(n), t, y)
    assert abs(simulate.area(x, q_noise) - random_area) < abs(simulate.area(x, q) - random_area)


def test_app_runs_on_committed_data():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "Test customers"
