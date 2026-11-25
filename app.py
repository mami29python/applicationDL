import streamlit as st

st.set_page_config(page_title="Projet Deep Learning", page_icon="🧠", layout="centered")

st.title("🧠 Projet Deep Learning")
st.write(
    "Une seule application pour tester tous les modèles du projet. "
    "Choisissez un modèle dans le menu de gauche."
)

st.subheader("Modèles disponibles")
st.markdown(
    "1. **Pneumonie** : détection sur radiographie thoracique (DenseNet121 fine-tuné)\n"
    "2. **Tuberculose** : détection sur radiographie thoracique (VGG16 fine-tuné)\n"
    "3. **CIFAR-10** : classification d'images en 10 classes (CNN)\n"
    "4. **Bank** : prédiction sur données bancaires (réseau dense)\n"
    "5. **Student** : prédiction de la performance des étudiants (réseau dense)"
)

st.caption("Les modèles pédagogiques ne remplacent pas un diagnostic médical.")
