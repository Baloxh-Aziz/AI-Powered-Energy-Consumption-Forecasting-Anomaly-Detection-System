import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).parent.parent / "artifacts"
st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.caption("All numbers are measured on the held-out test period, every model on exactly the same hours.")

BASE = {"next_hour": "Persistence (t-1h)", "next_day": "Seasonal naive (t-24h)"}


def table_view(csv, baseline, title):
    df = pd.read_csv(ART / csv)
    df.columns = [c.replace("Skill vs naive-24h", "skill_old") for c in df.columns]
    base_mae = df.loc[df["Model"] == baseline, "MAE (kW)"].iloc[0]
    df["Skill vs baseline"] = (1 - df["MAE (kW)"] / base_mae).round(3)
    df = df.drop(columns=["skill_old"], errors="ignore").sort_values("MAE (kW)").reset_index(drop=True)
    st.subheader(title)
    st.caption(f"Skill = 1 − MAE / MAE of **{baseline}**. Positive means better than that baseline.")
    st.dataframe(df, hide_index=True, width="stretch")
    fig = px.bar(df, x="Model", y="MAE (kW)", color=df["Model"].eq(baseline).map({True: "Baseline", False: "Model"}),
                 color_discrete_map={"Baseline": "#8a8f98", "Model": "#2f81f7"})
    fig.update_layout(height=380, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    return df


tab1, tab2, tab3 = st.tabs(["Next hour (Karachi_38)", "Next day (Karachi_38)", "All 18 houses"])
with tab1:
    d1 = table_view("model_comparison_1h.csv", BASE["next_hour"], "Forecasting 1 hour ahead")
    st.info("On this house the simple persistence baseline (use last hour's value) is hard to beat. "
            "The 'delta' Random Forest is close but does not win here.")
with tab2:
    d2 = table_view("model_comparison_24h.csv", BASE["next_day"], "Forecasting 24 hours ahead")
    st.info("Predicting the *change* from yesterday's same hour ('delta') works better than predicting raw usage.")
with tab3:
    s = pd.read_csv(ART / "summary.csv")
    st.subheader("Skill of the delta Random Forest over naive baselines, per house")
    st.caption("skill_1h is versus persistence (t-1h). skill_24h is versus seasonal naive (t-24h). Below 0 means the baseline wins.")
    long = s.melt(id_vars=["house", "city"], value_vars=["skill_1h", "skill_24h"], var_name="horizon", value_name="skill")
    fig = px.bar(long, x="house", y="skill", color="horizon", barmode="group")
    fig.add_hline(y=0, line_color="#888")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean skill, 1h", f"{s['skill_1h'].mean():.3f}")
    c2.metric("Mean skill, 24h", f"{s['skill_24h'].mean():.3f}")
    c3.metric("Houses beating naive at 24h", f"{int((s['skill_24h'] > 0).sum())} / {len(s)}")
    st.dataframe(s.sort_values("skill_24h", ascending=False).round(3), hide_index=True, width="stretch")
