import streamlit as st
import pickle
import numpy as np
import pandas as pd
from fault_reference import rule_based_predict, FAULT_INSIGHTS

# =========================
# Load Trained Model
# =========================
with open("model.pkl", "rb") as f:
    payload = pickle.load(f)

clf = payload["clf"]

# =========================
# Page Settings
# =========================
st.set_page_config(page_title="Vibration Fault Predictor", layout="centered")
st.title("Vibration Fault Prediction System")
st.caption("Enter measured vibration values in mm/sec. V (vertical) at 1X is always ~1.0 — it is the reference baseline.")

st.subheader("Enter measured values (mm/sec)")

# =========================
# Input Table
# V_1X pre-filled to 1.0 because reference baseline is always 1.0 for all faults
# =========================
harmonics = ["1X", "2X", "3X", "4X+"]
defaults = {
    "1X":  (0.00, 1.00, 0.00),
    "2X":  (0.00, 0.00, 0.00),
    "3X":  (0.00, 0.00, 0.00),
    "4X+": (0.00, 0.00, 0.00),
}

hdr = st.columns([1, 2, 2, 2])
hdr[0].markdown("**Harmonic**")
hdr[1].markdown("**H — Horizontal**")
hdr[2].markdown("**V — Vertical**")
hdr[3].markdown("**A — Axial**")

inputs = {}
for harm in harmonics:
    dh, dv, da = defaults[harm]
    cols = st.columns([1, 2, 2, 2])
    cols[0].markdown(f"**{harm}**")
    h = cols[1].number_input(f"H_{harm}", min_value=0.00, max_value=10.00,
                              value=float(st.session_state.get(f"h_{harm}", dh)),
                              step=0.01, format="%.2f",
                              key=f"h_{harm}", label_visibility="collapsed")
    v = cols[2].number_input(f"V_{harm}", min_value=0.00, max_value=10.00,
                              value=float(st.session_state.get(f"v_{harm}", dv)),
                              step=0.01, format="%.2f",
                              key=f"v_{harm}", label_visibility="collapsed")
    a = cols[3].number_input(f"A_{harm}", min_value=0.00, max_value=10.00,
                              value=float(st.session_state.get(f"a_{harm}", da)),
                              step=0.01, format="%.2f",
                              key=f"a_{harm}", label_visibility="collapsed")
    inputs[harm] = (round(h, 2), round(v, 2), round(a, 2))

st.caption("💡 V at 1X is almost always 1.00 — it is the reference baseline. Focus on filling H and A values.")

st.write("")

# =========================
# Example Buttons
# =========================
st.markdown("**Load example:**")
ec1, ec2, ec3 = st.columns(3)

def load_example(vals_dict):
    for k, v in vals_dict.items():
        st.session_state[k] = v
    st.rerun()

if ec1.button("Angular Misalignment"):
    load_example({"h_1X":0.70,"v_1X":1.00,"a_1X":3.00,
                  "h_2X":0.40,"v_2X":0.50,"a_2X":1.00,
                  "h_3X":0.20,"v_3X":0.20,"a_3X":0.30,
                  "h_4X+":0.02,"v_4X+":0.03,"a_4X+":0.04})

if ec2.button("Unbalance"):
    load_example({"h_1X":1.20,"v_1X":1.00,"a_1X":0.20,
                  "h_2X":1.50,"v_2X":0.12,"a_2X":0.05,
                  "h_3X":0.55,"v_3X":0.05,"a_3X":0.05,
                  "h_4X+":0.20,"v_4X+":0.05,"a_4X+":0.05})

if ec3.button("Rotating Looseness"):
    load_example({"h_1X":1.00,"v_1X":1.00,"a_1X":0.55,
                  "h_2X":0.90,"v_2X":0.90,"a_2X":0.40,
                  "h_3X":0.75,"v_3X":0.75,"a_3X":0.35,
                  "h_4X+":1.20,"v_4X+":1.20,"a_4X+":0.60})

st.write("")

# =========================
# Predict Button
# =========================
_, center, _ = st.columns([2, 1, 2])
with center:
    predict_btn = st.button("Predict Fault", use_container_width=True, type="primary")

# =========================
# Prediction Logic
# =========================
if predict_btn:

    values = []
    for harm in harmonics:
        h, v, a = inputs[harm]
        values += [h, v, a]

    # Validate — exclude V_1X (index 1) from zero check since it defaults to 1.0
    non_baseline = [values[i] for i in range(12) if i != 1]
    if sum(non_baseline) == 0:
        st.warning("⚠️ Please enter your H and A vibration readings. All values are currently 0.")
        st.stop()

    # ML prediction
    arr         = np.array([values])
    prediction  = clf.predict(arr)[0]
    proba_all   = clf.predict_proba(arr)[0]
    confidence  = round(float(max(proba_all)) * 100, 2)

    # Rule-based prediction
    rule_results = rule_based_predict(values)
    rule_fault   = rule_results[0][0]
    rule_score   = round(rule_results[0][1], 1)

    agree = (prediction == rule_fault)

    st.divider()

    # ── Agreement banner ──
    if agree and confidence >= 70:
        st.success(f"✅ Both methods agree: **{prediction}**  |  ML: {confidence}%  |  Rule: {rule_score}%")
    elif agree:
        st.info(f"ℹ️ Both agree: **{prediction}** — confidence is moderate. Double-check your readings.")
    else:
        st.warning(f"⚠️ Methods disagree  |  ML: **{prediction}** ({confidence}%)  |  Rule-Based: **{rule_fault}** ({rule_score}%)")
        if confidence < 70:
            st.caption("ML confidence < 70% → trust the Rule-Based result.")

    st.divider()

    # ── Rule-Based ──
    st.subheader("📐 Rule-Based Prediction")
    st.write(f"**Fault :** {rule_fault}")
    st.write(f"**Score :** {rule_score}%")
    st.caption("Checks each of your 12 values against the reference Excel ranges. Score = average % match across all cells.")

    st.divider()

    # ── ML Prediction ──
    st.subheader("🔬 Machine Learning Prediction")
    st.write(f"**Fault :** {prediction}")
    st.write(f"**Confidence :** {confidence:.2f}%")
    st.progress(int(confidence),
                text=f"{confidence:.1f}%  ({'HIGH ✅' if confidence>=70 else 'MEDIUM ⚠️' if confidence>=45 else 'LOW ❌'})")

    insight = FAULT_INSIGHTS.get(prediction, "")
    if insight:
        st.info(f"💡 {insight}")

    st.divider()

    # ── All Fault Rankings ──
    with st.expander("📊 All Fault Rankings — ML probabilities (always total 100%)"):
        st.caption("200 decision trees each cast a vote. These % values show how the votes were distributed.")
        ranking = sorted(zip(clf.classes_, proba_all), key=lambda x: -x[1])
        for idx, (fault, prob) in enumerate(ranking, 1):
            pct  = round(prob * 100, 1)
            bar  = "█" * min(int(pct / 2), 50)
            note = " ◀ top pick" if idx == 1 else ""
            if idx == 1:
                st.write(f"**{idx}. {fault} — {pct}%** {bar}{note}")
            else:
                st.write(f"{idx}. {fault} — {pct}% {bar}")

    st.divider()

    # ── Input summary ──
    with st.expander("📋 Your input values"):
        rows_display = [{"Harmonic": h, "H": inputs[h][0], "V": inputs[h][1], "A": inputs[h][2]}
                        for h in harmonics]
        st.dataframe(pd.DataFrame(rows_display), hide_index=True, use_container_width=True)
