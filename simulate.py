"""End-to-end pipeline: RFM features -> churn model -> simulated randomized
campaign -> T-learner uplift model -> Qini / AUUC evaluation on a held-out split.

Usage:
    python simulate.py

Expects the UCI Online Retail file at data/raw/Online Retail.xlsx.
Writes data/processed/*.csv and reports/*.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SEED = 42
CHURN_DAYS = 90

# Simulated campaign: response probability by risk group and treatment arm.
RESPONSE_PROB_TREATED = {"High": 0.40, "Medium": 0.20, "Low": 0.05}
RESPONSE_PROB_CONTROL = 0.03
TRUE_UPLIFT = {k: v - RESPONSE_PROB_CONTROL for k, v in RESPONSE_PROB_TREATED.items()}

CHURN_FEATURES = ["Frequency", "Monetary"]
# Recency is dropped: it encodes the churn label exactly (Churn == Recency > 91).
# Churn_Prob is dropped: it is the simulator's own input (risk is binned from it).
UPLIFT_FEATURES = ["Frequency", "Monetary"]


def load_transactions(path=RAW_PATH):
    df = pd.read_excel(path)
    df = df[df["CustomerID"].notnull()]
    df = df[df["Quantity"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df = df[df["TotalPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def build_rfm(df):
    last_date = df["InvoiceDate"].max()
    snapshot_date = last_date + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["Churn"] = ((last_date - rfm["LastPurchase"]).dt.days > CHURN_DAYS).astype(int)
    return rfm.reset_index()[["CustomerID", "Recency", "Frequency", "Monetary", "Churn"]]


def fit_churn_model(rfm):
    X, y = rfm[CHURN_FEATURES], rfm["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, (proba >= 0.5).astype(int)),
        "n_test": len(y_test),
    }
    return model, metrics


def simulate_campaign(customers, seed=SEED):
    """Randomize treatment 50/50 and draw responses from known probabilities."""
    np.random.seed(seed)
    out = customers.copy()
    n = len(out)
    out["Treatment"] = np.random.binomial(1, 0.5, size=n)
    p_treated = out["Risk"].map(RESPONSE_PROB_TREATED).astype(float)
    p = np.where(out["Treatment"] == 1, p_treated, RESPONSE_PROB_CONTROL)
    out["Response"] = np.random.binomial(1, p)
    out["True_Uplift"] = out["Risk"].map(TRUE_UPLIFT).astype(float)
    return out


def make_uplift_model():
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED
    )


def fit_t_learner(train):
    X, t, y = train[UPLIFT_FEATURES], train["Treatment"], train["Response"]
    model_t = make_uplift_model().fit(X[t == 1], y[t == 1])
    model_c = make_uplift_model().fit(X[t == 0], y[t == 0])
    return model_t, model_c


def predict_uplift(model_t, model_c, df):
    X = df[UPLIFT_FEATURES]
    return model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]


def _cumulative_counts(score, treatment, response):
    order = np.argsort(-np.asarray(score), kind="stable")
    t = np.asarray(treatment)[order]
    y = np.asarray(response)[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    return n_t, n_c, y_t, y_c


def qini_curve(score, treatment, response):
    """Qini curve: incremental responders when targeting the top-k by score.

    Returns (fraction targeted, incremental responders), both starting at 0.
    """
    n_t, n_c, y_t, y_c = _cumulative_counts(score, treatment, response)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    x = np.arange(len(q) + 1) / len(q)
    return x, np.concatenate([[0.0], q])


def uplift_curve(score, treatment, response):
    """Uplift (gain) curve: (treated rate - control rate) in the top-k, times k."""
    n_t, n_c, y_t, y_c = _cumulative_counts(score, treatment, response)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_diff = np.where(
            (n_t > 0) & (n_c > 0), y_t / np.maximum(n_t, 1) - y_c / np.maximum(n_c, 1), 0.0
        )
    k = np.arange(1, len(rate_diff) + 1)
    x = np.concatenate([[0.0], k / len(k)])
    return x, np.concatenate([[0.0], rate_diff * k])


def area(x, y):
    return float(np.trapezoid(y, x))


def evaluate(test):
    """Qini and AUUC for the model, random targeting and an oracle (true uplift)."""
    t, y = test["Treatment"], test["Response"]
    results, curves = {}, {}
    for name, score in [("model", test["Uplift_Score"]), ("oracle", test["True_Uplift"])]:
        xq, q = qini_curve(score, t, y)
        xu, u = uplift_curve(score, t, y)
        curves[name] = (xq, q)
        results[f"qini_area_{name}"] = area(xq, q)
        results[f"auuc_{name}"] = area(xu, u)
    # Random targeting is the straight line from (0, 0) to the full-population value.
    results["qini_area_random"] = curves["model"][1][-1] / 2
    results["auuc_random"] = uplift_curve(test["Uplift_Score"], t, y)[1][-1] / 2
    gain_model = results["qini_area_model"] - results["qini_area_random"]
    gain_oracle = results["qini_area_oracle"] - results["qini_area_random"]
    results["qini_coefficient"] = gain_model
    results["qini_normalized_vs_oracle"] = gain_model / gain_oracle
    # Share of all incremental responders captured when targeting the top fraction.
    xq, q = curves["model"]
    results["qini_share_captured"] = {
        f"top_{int(f * 100)}pct": float(q[int(round(f * (len(q) - 1)))] / q[-1])
        for f in (0.2, 0.4, 0.6, 0.8)
    }
    results["incremental_responders_total"] = float(q[-1])
    results["rmse_vs_true_uplift"] = float(
        np.sqrt(np.mean((test["Uplift_Score"] - test["True_Uplift"]) ** 2))
    )
    return results, curves


def recovery_table(test):
    """True vs estimated vs observed (difference in means) uplift per risk group."""
    rows = []
    for risk in ["High", "Medium", "Low"]:
        g = test[test["Risk"] == risk]
        observed = (
            g.loc[g["Treatment"] == 1, "Response"].mean()
            - g.loc[g["Treatment"] == 0, "Response"].mean()
        )
        rows.append(
            {
                "Risk": risk,
                "n_test": len(g),
                "n_treated": int(g["Treatment"].sum()),
                "true_uplift": TRUE_UPLIFT[risk],
                "estimated_uplift_mean": g["Uplift_Score"].mean(),
                "estimated_uplift_std": g["Uplift_Score"].std(),
                "observed_diff_in_means": observed,
            }
        )
    return pd.DataFrame(rows)


def plot_qini(curves, path):
    x, q_model = curves["model"]
    _, q_oracle = curves["oracle"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, q_oracle, label="Oracle (true uplift)", color="#888888", linestyle="--")
    ax.plot(x, q_model, label="T-learner", color="#1f77b4")
    ax.plot([0, 1], [0, q_model[-1]], label="Random targeting", color="#d62728", linestyle=":")
    ax.set_xlabel("Fraction of test customers targeted (sorted by score)")
    ax.set_ylabel("Incremental responders")
    ax.set_title("Qini curve - held-out test split")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rfm = build_rfm(load_transactions())
    churn_model, churn_metrics = fit_churn_model(rfm)

    rfm["Churn_Prob"] = churn_model.predict_proba(rfm[CHURN_FEATURES])[:, 1]
    rfm["Risk"] = pd.cut(
        rfm["Churn_Prob"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    customers = simulate_campaign(rfm)
    train, test = train_test_split(
        customers, test_size=0.3, random_state=SEED, stratify=customers["Treatment"]
    )
    customers["Split"] = np.where(customers.index.isin(test.index), "test", "train")

    model_t, model_c = fit_t_learner(train)
    test = test.copy()
    test["Uplift_Score"] = predict_uplift(model_t, model_c, test)

    metrics, curves = evaluate(test)
    recovery = recovery_table(test)

    customers.to_csv(PROCESSED_DIR / "customers_simulated.csv", index=False)
    test.sort_values("Uplift_Score", ascending=False).to_csv(
        PROCESSED_DIR / "uplift_scores_test.csv", index=False
    )
    recovery.to_csv(REPORTS_DIR / "recovery_table.csv", index=False)
    plot_qini(curves, REPORTS_DIR / "qini_curve.png")

    summary = {
        "n_customers": len(customers),
        "n_train": len(train),
        "n_test": len(test),
        "risk_counts": customers["Risk"].value_counts().to_dict(),
        "treatment_rate": float(customers["Treatment"].mean()),
        "churn_model": churn_metrics,
        "uplift_evaluation": metrics,
        "uplift_score_range_test": [float(test["Uplift_Score"].min()), float(test["Uplift_Score"].max())],
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps(summary, indent=2, default=float))
    print(recovery.to_string(index=False))


if __name__ == "__main__":
    main()
