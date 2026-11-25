import numpy as np
import streamlit as st

from common import load_artifact, load_model

st.set_page_config(page_title="Student", page_icon="🎓")

st.title("🎓 Performance des étudiants")

hours = st.slider("Heures d'étude", 1, 9, 5)
prev = st.slider("Notes précédentes", 40, 99, 70)
extra = st.selectbox("Activités extrascolaires", ["Yes", "No"])
sleep = st.slider("Heures de sommeil", 4, 9, 7)
papers = st.slider("Sujets d'examen pratiqués", 0, 9, 4)

if st.button("Prédire", key="btn_student"):
    try:
        x = np.array([[hours, prev, 1 if extra == "Yes" else 0, sleep, papers]], dtype="float32")
        x = load_artifact("scaler.jb").transform(x)
        model = load_model("students_model.keras")
        pred = float(model.predict(x, verbose=0).ravel()[0])
        st.success(f"Indice de performance prédit : **{pred:.1f}**")
    except Exception as e:
        st.error(f"Erreur : {e}")
