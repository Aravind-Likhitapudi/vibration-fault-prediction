import streamlit as st
import pickle
from fault_reference import rule_based_predict

# =========================
# Load Trained Model
# =========================
with open("model.pkl", "rb") as f:
    payload = pickle.load(f)

clf = payload["clf"]

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Vibration Fault Predictor",
    layout="centered"
)

st.title("Vibration Fault Prediction System")

st.subheader("Enter measured values (mm/sec)")

# =========================
# Input Table
# =========================
c1, c2, c3, c4 = st.columns([1, 2, 2, 2])

with c1:
    st.write("**Harmonic**")
    st.write("")
    st.write("**1X**")
    st.write("")
    st.write("**2X**")
    st.write("")
    st.write("**3X**")
    st.write("")
    st.write("**4X+**")

with c2:
    st.write("**H (horizontal)**")
    h1 = st.number_input("1X_H", value=0.00, key="h1", label_visibility="collapsed")
    h2 = st.number_input("2X_H", value=0.00, key="h2", label_visibility="collapsed")
    h3 = st.number_input("3X_H", value=0.00, key="h3", label_visibility="collapsed")
    h4 = st.number_input("4X_H", value=0.00, key="h4", label_visibility="collapsed")

with c3:
    st.write("**V (vertical)**")
    v1 = st.number_input("1X_V", value=0.00, key="v1", label_visibility="collapsed")
    v2 = st.number_input("2X_V", value=0.00, key="v2", label_visibility="collapsed")
    v3 = st.number_input("3X_V", value=0.00, key="v3", label_visibility="collapsed")
    v4 = st.number_input("4X_V", value=0.00, key="v4", label_visibility="collapsed")

with c4:
    st.write("**A (axial)**")
    a1 = st.number_input("1X_A", value=0.00, key="a1", label_visibility="collapsed")
    a2 = st.number_input("2X_A", value=0.00, key="a2", label_visibility="collapsed")
    a3 = st.number_input("3X_A", value=0.00, key="a3", label_visibility="collapsed")
    a4 = st.number_input("4X_A", value=0.00, key="a4", label_visibility="collapsed")

st.write("")


# =========================
# Predict Button (Centered)
# =========================
left, center, right = st.columns([2, 1, 2])

with center:
    predict_btn = st.button(
        "Predict Fault",
        use_container_width=True
    )

# =========================
# Prediction
# =========================

if predict_btn:

    values = [
        h1, v1, a1,
        h2, v2, a2,
        h3, v3, a3,
        h4, v4, a4
    ]

    # Prevent empty prediction
    if sum(values) == 0:
        st.warning("Please enter vibration values.")
        st.stop()

    # Machine Learning
    prediction = clf.predict([values])[0]
    probabilities = clf.predict_proba([values])[0]
    probability = max(probabilities) * 100

    # Rule Based
    rule_results = rule_based_predict(values)
    rule_fault = rule_results[0][0]
    rule_score = rule_results[0][1]

    st.divider()

    # =========================
    # Rule Based Result
    # =========================
    st.subheader("Rule-Based Prediction")
    st.write(f"**Fault :** {rule_fault}")
    st.write(f"**Score :** {rule_score:.1f}%")
    with st.expander("📋 View Rule-Based Rankings"):

        for idx, (fault, score) in enumerate(rule_results, start=1):

            bar = "█" * min(int(score / 2), 50)

            if idx == 1:
                st.write(
                    f"**{idx}. {fault} - {score:.1f}%** {bar} ◀"
                )
            else:
                st.write(
                    f"{idx}. {fault} - {score:.1f}% {bar}"
                )
    st.divider()

    # =========================
    # ML Result
    # =========================
    st.subheader("Machine Learning Prediction")
    st.write(f"**Fault :** {prediction}")
    st.write(f"**Confidence :** {probability:.2f}%")

    st.divider()

    # =========================
    # All Fault Rankings
    # =========================
    with st.expander("📊 View Machine Learning Rankings"):

        # Get class names from model
        classes = clf.classes_

        ranking = list(zip(classes, probabilities))
        ranking.sort(key=lambda x: x[1], reverse=True)

        for idx, (fault, prob) in enumerate(ranking, start=1):

            percent = prob * 100

            bar = "█" * min(int(percent / 2), 50)

            if idx == 1:
                st.write(
                    f"**{idx}. {fault} - {percent:.1f}%** {bar} ◀"

                )
            else:
                st.write(
                    f"{idx}. {fault} - {percent:.1f}% {bar}"

                )
