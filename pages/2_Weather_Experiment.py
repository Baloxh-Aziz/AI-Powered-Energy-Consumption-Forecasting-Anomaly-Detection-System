from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).parent.parent / "artifacts"
st.set_page_config(page_title="Weather Experiment", page_icon="🌦️", layout="wide")
st.title("🌦️ Does weather help?")
st.caption("Day-ahead Random Forest with and without observed weather (temperature, humidity, wind, solar radiation).")

w = pd.read_csv(ART / "weather_results.csv")
both = int(((w.val_gain_pct > 0) & (w.test_gain_pct > 0)).sum())
agree = int(((w.val_gain_pct > 0) == (w.test_gain_pct > 0)).sum())
rule = w.loc[w.val_gain_pct > 0, "test_gain_pct"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Weather helped on val AND test", f"{both} / {len(w)} houses")
c2.metric("Val and test agree on direction", f"{agree} / {len(w)}")
c3.metric("Mean test gain if we 'use weather when val gain > 0'", f"{rule:.2f}%")

long = w.melt(id_vars=["house", "city"], value_vars=["val_gain_pct", "test_gain_pct"], var_name="split", value_name="gain_pct")
fig = px.bar(long, x="house", y="gain_pct", color="split", barmode="group",
             labels={"gain_pct": "MAE improvement from weather (%)"})
fig.add_hline(y=0, line_color="#888")
fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
st.plotly_chart(fig, width="stretch")

st.subheader("Conclusion")
st.markdown(
    "Weather is **not** included in the final model.\n\n"
    "- Gains on validation did not carry over to test in many houses, and some houses got much worse.\n"
    "- Each house has only about one year of data, which is too little to learn weather effects reliably.\n"
    "- This experiment used *observed* weather. In real use only a weather *forecast* is available, so the true gain would be smaller still.\n\n"
    "A multi-city model trained on pooled houses and forecast weather would be the next thing to try.")
st.dataframe(w.round(2).sort_values("val_gain_pct", ascending=False), hide_index=True, width="stretch")
