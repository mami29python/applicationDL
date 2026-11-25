import numpy as np
import streamlit as st

from common import load_artifact, load_model, show_result, to_probs

st.set_page_config(page_title="Bank", page_icon="🏦")

st.title("🏦 Souscription à un produit bancaire (DNN)")
st.write("Renseignez les informations du client pour prédire s'il va souscrire.")

# encoders.joblib est une LISTE de LabelEncoder, dans cet ordre :
# 0 job, 1 marital, 2 education, 3 default, 4 housing,
# 5 loan, 6 contact, 7 month, 8 day_of_week, 9 poutcome
try:
    encoders = load_artifact("encoders.joblib")
    scaler = load_artifact("scaler.joblib")
except Exception as e:
    st.error(f"Impossible de charger les fichiers du prétraitement : {e}")
    st.stop()


def choix(label, idx):
    """Liste déroulante construite à partir des classes de l'encodeur."""
    return st.selectbox(label, list(encoders[idx].classes_))


col1, col2 = st.columns(2)
with col1:
    age = st.number_input("age", min_value=17, max_value=100, value=40, step=1)
    job = choix("job", 0)
    marital = choix("marital", 1)
    education = choix("education", 2)
    default = choix("default", 3)
    housing = choix("housing", 4)
    loan = choix("loan", 5)
    contact = choix("contact", 6)
with col2:
    month = choix("month", 7)
    day_of_week = choix("day_of_week", 8)
    duration = st.number_input("duration (secondes du dernier appel)", min_value=0,
                               max_value=5000, value=200, step=10)
    campaign = st.number_input("campaign (contacts durant cette campagne)", min_value=1,
                               max_value=60, value=1, step=1)
    pdays = st.number_input("pdays (jours depuis le dernier contact, 999 = jamais)",
                            min_value=0, max_value=999, value=999, step=1)
    previous = st.number_input("previous (contacts avant cette campagne)", min_value=0,
                               max_value=10, value=0, step=1)
    poutcome = choix("poutcome", 9)

if st.button("Prédire", key="btn_bank"):
    try:
        x = np.array([
            age,
            encoders[0].transform([job])[0],
            encoders[1].transform([marital])[0],
            encoders[2].transform([education])[0],
            encoders[3].transform([default])[0],
            encoders[4].transform([housing])[0],
            encoders[5].transform([loan])[0],
            encoders[6].transform([contact])[0],
            encoders[7].transform([month])[0],
            encoders[8].transform([day_of_week])[0],
            duration, campaign, pdays, previous,
            encoders[9].transform([poutcome])[0],
        ], dtype="float64").reshape(1, -1)

        x = scaler.transform(x)
        model = load_model("DNN_model.keras")
        probs = to_probs(model.predict(x, verbose=0))
        show_result(["Pas souscrire", "Souscrire"], probs)
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")