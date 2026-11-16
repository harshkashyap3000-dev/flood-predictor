"""Flood Risk Predictor - Brahmaputra Valley Pilot (MTIS case: "Before the Water Comes").

Trains a logistic-regression model on the 1,000-row sample data at start-up
and predicts the probability of flooding for a location.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="wide")

# name, label, unit, help text, slider step
FEATURES = [
    ("distance_to_river_km", "Distance to river", "km", "How far the location is from the main river channel", 0.05),
    ("rain_mm_24h", "Rain, last 24 h", "mm", "Rainfall in the past 24 hours (rain gauge)", 0.5),
    ("api_7d", "7-day antecedent rain index", "index", "Rain over the last 7 days: how wet the ground already is", 0.5),
    ("river_level_m", "River level", "m", "Water level at the nearest river gauge", 0.05),
    ("dam_discharge_cumecs", "Dam discharge", "m³/s", "Water released from upstream dams", 1.0),
    ("storm_drain_capacity_mmph", "Storm-drain capacity", "mm/h", "Rain per hour the local drains can carry away", 0.5),
    ("drain_blockage_index", "Drain blockage index", "%", "0 = drains clear, 100 = fully blocked", 0.5),
    ("tide_level_m", "Tide level", "m", "Coastal locations only; keep 0 inland", 0.05),
    ("storm_surge_m", "Storm surge", "m", "Cyclone storm surge; coastal only, keep 0 inland", 0.05),
]
COLS = [f[0] for f in FEATURES]
LABEL = {f[0]: f[1] for f in FEATURES}
THRESHOLD = 0.5


@st.cache_resource
def train():
    df = pd.read_csv("flood_data.csv")
    X, y = df[COLS], df["flood_occurred"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= THRESHOLD).astype(int)
    cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), Xtr, ytr,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    fpr, tpr, _ = roc_curve(yte, p)
    metrics = dict(acc=accuracy_score(yte, pred), prec=precision_score(yte, pred), rec=recall_score(yte, pred),
                   f1=f1_score(yte, pred), auc=roc_auc_score(yte, p), cv_auc=cv.mean(),
                   cm=confusion_matrix(yte, pred), fpr=fpr, tpr=tpr, n_train=len(Xtr), n_test=len(Xte))
    stats = X.describe().T  # min, 50%, max for sliders
    return model, metrics, stats, df


model, M, STATS, DATA = train()
MED = {c: float(STATS.loc[c, "50%"]) for c in COLS}


def predict(df_in: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df_in[COLS])[:, 1]


def band(p):
    if p >= 0.7:
        return "RED – Evacuate", "error", ("Issue evacuation warning by SMS / IVR / cell broadcast and sirens; "
                                          "open relief camps; move livestock and harvestable crops.")
    if p >= 0.5:
        return "ORANGE – Flood warning", "warning", ("Warn villages, pre-position boats and relief teams, "
                                                     "alert the district control room; re-check every hour.")
    if p >= 0.3:
        return "YELLOW – Watch", "warning", ("Monitor sensors closely, ask gram panchayats to stay alert, "
                                            "check drains and embankments.")
    return "GREEN – Normal", "success", "No action needed; routine monitoring."


PRESETS = {
    "Typical (median) day": dict(MED),
    "Normal dry day": dict(MED, distance_to_river_km=6.0, rain_mm_24h=8.0, api_7d=5.0, river_level_m=2.5,
                           dam_discharge_cumecs=30.0, storm_drain_capacity_mmph=35.0, drain_blockage_index=15.0),
    "Moderate rain": dict(MED, rain_mm_24h=30.0, api_7d=18.0),
    "Heavy monsoon rain": dict(MED, distance_to_river_km=1.5, rain_mm_24h=55.0, api_7d=35.0, river_level_m=5.0,
                               dam_discharge_cumecs=90.0, storm_drain_capacity_mmph=20.0, drain_blockage_index=40.0),
    "Dam release + high river": dict(MED, distance_to_river_km=0.8, rain_mm_24h=25.0, api_7d=18.0, river_level_m=6.2,
                                     dam_discharge_cumecs=300.0, drain_blockage_index=30.0),
    "Coastal cyclone surge": dict(MED, distance_to_river_km=3.0, rain_mm_24h=30.0, api_7d=20.0, river_level_m=3.5,
                                  dam_discharge_cumecs=50.0, storm_drain_capacity_mmph=22.0, drain_blockage_index=35.0,
                                  tide_level_m=1.6, storm_surge_m=1.1),
}


def apply_preset():
    for c, v in PRESETS[st.session_state.preset].items():
        st.session_state[c] = float(v)


if "preset" not in st.session_state:
    st.session_state.preset = "Typical (median) day"
    apply_preset()

# ---------------- Header ----------------
st.title("🌊 Flood Risk Predictor – Brahmaputra Valley Pilot")
st.caption(f"AI model: logistic regression trained on {M['n_train']} sample observations · "
           f"accuracy {M['acc']:.0%} and ROC-AUC {M['auc']:.3f} on {M['n_test']} unseen test rows · "
           "MTIS case: “Before the Water Comes”")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict one location", "📋 Predict many locations", "📊 Model performance",
                                  "ℹ️ About"])

# ---------------- Tab 1: single prediction ----------------
with tab1:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("1. Enter current conditions")
        st.selectbox("Load a scenario (optional)", list(PRESETS), key="preset", on_change=apply_preset)
        for c, lab, unit, hlp, step in FEATURES:
            lo, hi = float(STATS.loc[c, "min"]), float(STATS.loc[c, "max"])
            st.slider(f"{lab} ({unit})", min_value=lo, max_value=hi, step=step, key=c, help=hlp)
    vals = {c: st.session_state[c] for c in COLS}
    p = float(predict(pd.DataFrame([vals]))[0])
    name, kind, action = band(p)
    with right:
        st.subheader("2. Prediction")
        st.metric("Probability of flooding", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        getattr(st, kind)(f"**{name}**\n\n**Recommended action:** {action}")
        st.write(f"Model verdict at the {THRESHOLD:.0%} threshold: **{'FLOOD expected' if p >= THRESHOLD else 'no flood expected'}**")
        st.caption("Bands: Green < 30% · Yellow 30–50% · Orange 50–70% · Red ≥ 70%. "
                   "The model advises; the District Collector issues the warning.")

        st.subheader("3. Why? What drives this prediction")
        lr = model.named_steps["logisticregression"]
        sc = model.named_steps["standardscaler"]
        contrib = lr.coef_[0] * (np.array([vals[c] for c in COLS]) - np.array([MED[c] for c in COLS])) / sc.scale_
        cdf = pd.DataFrame({"Input": [LABEL[c] for c in COLS], "Effect": contrib})
        cdf["Direction"] = np.where(cdf["Effect"] >= 0, "Raises risk", "Lowers risk")
        chart = alt.Chart(cdf).mark_bar().encode(
            x=alt.X("Effect:Q", title="Effect vs a typical day (log-odds)"),
            y=alt.Y("Input:N", sort=alt.EncodingSortField("Effect", op="sum", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                           range=["#c62828", "#2e7d32"])),
            tooltip=["Input", alt.Tooltip("Effect:Q", format="+.2f")]).properties(height=280)
        if np.allclose(contrib, 0):
            st.info("All inputs are at typical (median) values, so no input pushes the risk up or down. Move a slider to see its effect.")
        else:
            st.altair_chart(chart, width="stretch")

# ---------------- Tab 2: batch ----------------
with tab2:
    st.subheader("Predict flood risk for many locations at once")
    st.write("Upload a CSV file with these columns (any order; other columns are ignored):")
    st.code(", ".join(COLS))
    use_sample = st.checkbox("Or use the 1,000-row sample dataset from the case", value=False)
    up = st.file_uploader("Upload CSV", type=["csv"])
    bdf = None
    if up is not None:
        bdf = pd.read_csv(up)
    elif use_sample:
        bdf = DATA.copy()
    if bdf is not None:
        missing = [c for c in COLS if c not in bdf.columns]
        if missing:
            st.error("Missing column(s): " + ", ".join(missing))
        else:
            bdf = bdf.dropna(subset=COLS).copy()
            bdf["flood_probability"] = predict(bdf).round(4)
            bdf["predicted_flood"] = (bdf["flood_probability"] >= THRESHOLD).astype(int)
            bdf["alert_band"] = [band(x)[0] for x in bdf["flood_probability"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows scored", f"{len(bdf):,}")
            c2.metric("Predicted to flood", f"{bdf['predicted_flood'].sum():,} ({bdf['predicted_flood'].mean():.1%})")
            if "flood_occurred" in bdf.columns:
                c3.metric("Accuracy vs actual", f"{(bdf['predicted_flood'] == bdf['flood_occurred']).mean():.1%}")
            st.dataframe(bdf[[c for c in ["season", "basin"] if c in bdf.columns] + COLS +
                             ["flood_probability", "predicted_flood", "alert_band"] +
                             (["flood_occurred"] if "flood_occurred" in bdf.columns else [])],
                         width="stretch", height=380)
            st.download_button("⬇️ Download results as CSV", bdf.to_csv(index=False).encode("utf-8"),
                               "flood_predictions.csv", "text/csv")

# ---------------- Tab 3: model performance ----------------
with tab3:
    st.subheader(f"How good is the model? (tested on {M['n_test']} rows it never saw)")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{M['acc']:.1%}")
    b.metric("Precision", f"{M['prec']:.1%}")
    c.metric("Recall", f"{M['rec']:.1%}")
    d.metric("F1 score", f"{M['f1']:.1%}")
    e.metric("ROC-AUC", f"{M['auc']:.3f}")
    st.caption(f"5-fold cross-validated AUC on the training data: {M['cv_auc']:.3f}. "
               "Logistic regression was chosen over random forest (CV AUC 0.960) and gradient boosting (0.964): "
               "as accurate, and fully explainable.")
    g1, g2 = st.columns(2)
    with g1:
        cm = M["cm"]
        st.write("**Confusion matrix**")
        st.table(pd.DataFrame(cm, index=["Actual: no flood", "Actual: flood"],
                              columns=["Predicted: no flood", "Predicted: flood"]))
    with g2:
        roc = pd.DataFrame({"False positive rate": M["fpr"], "True positive rate": M["tpr"]})
        st.write("**ROC curve**")
        st.altair_chart(alt.Chart(roc).mark_line().encode(x="False positive rate", y="True positive rate")
                        .properties(height=260), width="stretch")
    lr = model.named_steps["logisticregression"]
    imp = pd.DataFrame({"Input": [LABEL[c] for c in COLS], "Standardised coefficient": lr.coef_[0]})
    st.write("**What the model learned** (bigger bar = stronger effect; red raises flood risk, green lowers it)")
    st.altair_chart(alt.Chart(imp).mark_bar().encode(
        x="Standardised coefficient:Q",
        y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.condition("datum['Standardised coefficient'] > 0", alt.value("#c62828"), alt.value("#2e7d32")))
        .properties(height=280), width="stretch")

# ---------------- Tab 4: about ----------------
with tab4:
    st.markdown("""
**Purpose.** Prototype of the prediction app in the Detailed Project Report for the Brahmaputra valley
flood-prediction pilot (Ministry of Jal Shakti). District officials enter live sensor readings for a
location and get the probability of flooding, an alert band and the recommended action.

**Data.** 1,000 illustrative observations from the case (9 basins, 2024). Inputs are the 9 variables
marked “Selected model input” in the data dictionary. Outcome variables (water depth, severity, risk
score, NDWI) are excluded to avoid leakage.

**Method.** Data split 80/20 (stratified), inputs standardised, logistic regression trained
(scikit-learn). Probability = 1 / (1 + e^(−z)); 50% or more is predicted as a flood.

**In production** this model would run in Azure ML, fed in real time by rain gauges, river-level radars
and soil-moisture stations through Azure IoT Hub, and be retrained after every monsoon.

**Caveat.** The sample data is synthetic, so real-world accuracy will be lower, especially without
upstream (China) river data.
""")
