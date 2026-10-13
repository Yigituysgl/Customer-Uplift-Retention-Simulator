from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "uplift_scores_test.csv"


@st.cache_data
def load_data():
    # Held-out test customers only; produced by simulate.py.
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()


st.sidebar.title("Filters")

risk_options = ["All"] + sorted(df["Risk"].dropna().unique().tolist())
risk_selected = st.sidebar.selectbox("Risk segment", risk_options)

min_uplift = float(df["Uplift_Score"].min())
max_uplift = float(df["Uplift_Score"].max())

uplift_min = st.sidebar.slider(
    "Minimum uplift score",
    min_value=round(min_uplift, 2),
    max_value=round(max_uplift, 2),
    value=round(df["Uplift_Score"].quantile(0.8), 2),
    step=0.01,
)

# filter
filtered = df.copy()
if risk_selected != "All":
    filtered = filtered[filtered["Risk"] == risk_selected]
filtered = filtered[filtered["Uplift_Score"] >= uplift_min]


st.title("Customer Uplift & Retention Simulator")
st.caption(
    "Held-out test customers scored by the T-learner. Treatment and response are "
    "simulated (see README), so True_Uplift is known."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Test customers", len(df))
with col2:
    st.metric("Target customers (filtered)", len(filtered))
with col3:
    st.metric(
        "Avg uplift (filtered)",
        f"{filtered['Uplift_Score'].mean():.3f}" if len(filtered) > 0 else "N/A",
    )

st.write("---")


st.subheader("Uplift Score Distribution")
counts, edges = np.histogram(df["Uplift_Score"], bins=40)
hist = pd.DataFrame({"Uplift_Score": np.round(edges[:-1], 2), "Customers": counts})
st.bar_chart(hist, x="Uplift_Score", y="Customers")

st.subheader("Top Target Customers (by uplift)")
top_n = st.slider("Number of customers to show", 10, 200, 50)
top_customers = filtered.sort_values("Uplift_Score", ascending=False).head(top_n)

st.dataframe(
    top_customers[
        ["CustomerID", "Recency", "Frequency", "Monetary", "Risk", "Churn_Prob",
         "True_Uplift", "Uplift_Score"]
    ]
)

# download
csv = top_customers.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered customers as CSV",
    data=csv,
    file_name="target_customers_filtered.csv",
    mime="text/csv",
)
