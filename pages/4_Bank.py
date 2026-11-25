import pandas as pd
import streamlit as st

from common import load_artifact, load_model, show_result, to_probs

st.set_page_config(page_title="Bank", page_icon="🏦")

st.title("🏦 Données bancaires (DNN)")

# Colonnes d'entrée, DANS L'ORDRE utilisé à l'entraînement.
#   ("num", min, max, défaut)  : variable numérique
#   ("cat", [valeurs...])      : variable catégorielle (encodée avec encoders.joblib)
# À REMPLACER par les vraies colonnes de banking_dataset.csv.
FEATURES = {
    "age": ("num", 18, 95, 40),
    "balance": ("num", -5000, 100000, 1000),
    "job": ("cat", ["admin.", "technician", "management", "services", "retired"]),
    "marital": ("cat", ["married", "single", "divorced"]),
}
TARGET_LABELS = ["Non", "Oui"]

values = {}
for name, spec in FEATURES.items():
    if spec[0] == "num":
        _, lo, hi, default = spec
        values[name] = st.number_input(name, min_value=lo, max_value=hi, value=default)
    else:
        values[name] = st.selectbox(name, spec[1])

if st.button("Prédire", key="btn_bank"):
    try:
        df = pd.DataFrame([values])
        encoders = load_artifact("encoders.joblib")
        for col, spec in FEATURES.items():
            if spec[0] == "cat":
                df[col] = encoders[col].transform(df[col])
        scaler = load_artifact("scaler.joblib")
        x = scaler.transform(df.values.astype("float32"))
        model = load_model("DNN_model.keras")
        probs = to_probs(model.predict(x, verbose=0))
        show_result(TARGET_LABELS, probs)
    except Exception as e:
        st.error(f"Erreur de prétraitement : {e}")
