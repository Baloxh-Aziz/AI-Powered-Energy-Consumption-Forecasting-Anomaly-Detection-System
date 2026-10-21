"""Household energy forecasting & anomaly dashboard (REWDP, Pakistan).

Reads the precomputed per-house CSVs in ./artifacts (columns: datetime, actual,
predicted, z, forecast_24h). No model loading needed, so it runs on any host.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
API_URL = "https://energy-api-project-d1d7b.containers.snapdeploy.app"
st.set_page_config(page_title="Household Energy Dashboard", page_icon="⚡", layout="wide")


@st.cache_data
def load_meta():
    return pd.DataFrame(json.loads((ART / "meta.json").read_text()))


@st.cache_data
def load_house(name):
    df = pd.read_csv(ART / f"{name}.csv", parse_dates=["datetime"], index_col="datetime")
    return df.sort_index()


def flag(df, k):
    out = df.copy()
    out["anomaly"] = out["z"].abs() > k
    out["type"] = np.where(out["z"] > 0, "spike", "drop")
    return out


meta = load_meta()

st.title("⚡ Household Energy Forecasting & Anomaly Detection")
st.caption("Pakistan REWDP dataset · Random Forest (delta) day-ahead forecast · "
           "robust (MAD) relative-residual anomaly detection")

# ---- sidebar ----
city = st.sidebar.selectbox("City", sorted(meta["city"].unique()))
houses = meta.loc[meta["city"] == city, "house"].tolist()
house = st.sidebar.selectbox("House", houses)
k = st.sidebar.slider("Anomaly threshold k", 2.0, 10.0, 5.0, 0.5,
                      help="Higher k = fewer, stronger anomalies. Default 5.")

df = load_house(house)
lo, hi = df.index.min().date(), df.index.max().date()
rng = st.sidebar.date_input("Date range (test period)", (lo, hi), min_value=lo, max_value=hi)
if isinstance(rng, tuple) and len(rng) == 2:
    df = df.loc[str(rng[0]):str(rng[1]) + " 23:59:59"]

df = flag(df, k)
m = meta.set_index("house").loc[house]
mae = (df["actual"] - df["forecast_24h"]).abs().mean()
nmae = mae / df["actual"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test hours", f"{len(df):,}")
c2.metric("Day-ahead MAE", f"{mae:.3f} kW", f"{100 * nmae:.1f}% of mean usage", delta_color="off")
c3.metric("Anomalies", f"{int(df['anomaly'].sum())}", f"{100 * df['anomaly'].mean():.2f}% of hours", delta_color="off")
c4.metric("Data observed", f"{m['pct_observed']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast & anomalies", "Anomaly list", "All houses", "Usage patterns", "Live API demo"])

with tab1:
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df["actual"], name="Actual", line=dict(width=1.5))
    fig.add_scatter(x=df.index, y=df["forecast_24h"], name="Forecast (24h ahead)",
                    line=dict(width=1.5, dash="dot"))
    a = df[df["anomaly"]]
    fig.add_scatter(x=a.index, y=a["actual"], mode="markers", name="Anomaly",
                    marker=dict(size=9, color="red", symbol="x"))
    fig.update_layout(height=480, hovermode="x unified", yaxis_title="Usage (kW)",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Forecast uses only values from 24h, 48h and 168h earlier plus hour/day-of-week. "
               "Anomalies are hours where actual usage deviates strongly from the model's expected value.")

with tab2:
    if a.empty:
        st.info("No anomalies at this threshold. Lower k in the sidebar.")
    else:
        t = a[["actual", "predicted", "z", "type"]].assign(z=lambda x: x["z"].round(1))
        t = t.reindex(t["z"].abs().sort_values(ascending=False).index)
        st.dataframe(t.rename(columns={"actual": "Actual (kW)", "predicted": "Expected (kW)",
                                       "z": "Score (z)", "type": "Type"}), width="stretch")
        st.download_button("Download anomalies (CSV)", t.to_csv().encode(), f"{house}_anomalies.csv")

with tab3:
    rows = []
    for h in meta["house"]:
        d = flag(load_house(h), k)
        rows.append({"House": h, "City": h.split("_")[0],
                     "Day-ahead nMAE (%)": round(100 * (d["actual"] - d["forecast_24h"]).abs().mean() / d["actual"].mean(), 1),
                     "Anomaly rate (%)": round(100 * d["anomaly"].mean(), 2),
                     "Test hours": len(d)})
    st.dataframe(pd.DataFrame(rows).sort_values("Anomaly rate (%)", ascending=False),
                 width="stretch", hide_index=True)

with tab4:
    p = df.assign(hour=df.index.hour, dow=df.index.day_name())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    a1, a2 = st.columns(2)
    h = p.groupby("hour")["actual"].mean()
    a1.subheader("Average usage by hour")
    a1.bar_chart(h)
    dd = p.groupby("dow")["actual"].mean().reindex(order)
    a2.subheader("Average usage by weekday")
    a2.bar_chart(dd)

with tab5:
    st.subheader("Live API demo")
    st.write("This tab sends recent usage to the deployed FastAPI service and shows its forecast. "
             "The deployed model was trained on **Karachi_38** only. "
             "The free server sleeps when idle, so the first call can take about a minute.")

    demo = load_house("Karachi_38")["actual"].asfreq("h")          # one row per hour, gaps = NaN
    horizon = st.radio("Forecast horizon", ["1h", "24h"], horizontal=True)
    steps = 1 if horizon == "1h" else 24

    # a week (168 hours) of complete history must end `steps` hours before the target hour
    complete = demo.notna().astype(int).rolling(168).sum() == 168
    can_forecast = complete.shift(steps, freq="h")
    targets = [t for t in can_forecast[can_forecast == True].index if t in demo.index and pd.notna(demo[t])]

    if not targets:
        st.warning("No hour with a full week of history was found.")
    else:
        target = st.selectbox("Hour to forecast", targets, index=len(targets) // 2)
        if st.button("Call the API"):
            end = target - pd.Timedelta(hours=steps)
            history = demo.loc[end - pd.Timedelta(hours=167): end].tolist()
            body = {"history": history, "target_time": str(target), "horizon": horizon}
            try:
                with st.spinner("Calling the API (it may need to wake up)..."):
                    r = requests.post(API_URL + "/predict", json=body, timeout=120)
                    r.raise_for_status()
                    forecast = r.json()["forecast_kw"]
                    shown = {"API forecast (kW)": round(forecast, 3), "Actual (kW)": round(float(demo[target]), 3)}
                    if horizon == "1h":
                        body2 = {"history": history, "target_time": str(target), "actual": float(demo[target])}
                        r2 = requests.post(API_URL + "/detect-anomaly", json=body2, timeout=120)
                        r2.raise_for_status()
                        shown["Anomaly score (z)"] = r2.json()["z_score"]
                        shown["Anomaly?"] = "Yes" if r2.json()["is_anomaly"] else "No"
                cols = st.columns(len(shown))
                for col, (name, value) in zip(cols, shown.items()):
                    col.metric(name, value)
            except Exception as e:
                st.error(f"API call failed: {e}")

with st.expander("Limitations"):
    st.markdown(
        "- About one year of data per house; the test period covers only part of the year.\n"
        "- Day-ahead skill over the seasonal-naive baseline is modest on average (~5%) and negative in some houses.\n"
        "- Anomalies are statistical outliers, not confirmed faults.\n"
        "- Weather was tested and did not help reliably, so it is not in the final model.")
