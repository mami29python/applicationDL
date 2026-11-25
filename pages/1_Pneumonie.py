import streamlit as st
from PIL import Image

from common import image_page

st.set_page_config(page_title="Pneumonie", page_icon="🫁")

st.title("🫁 Détection de la pneumonie")
st.write("Téléversez une radiographie thoracique pour estimer la probabilité de pneumonie.")

# Paramètres à aligner sur le notebook d'entraînement :
#  - grayscale / interpolation / rescale
#  - ordre des classes (Normal = 0, Pneumonie = 1)
image_page(
    model_file="densenet_model_finetuned.h5",
    classes=["Normal", "Pneumonie"],
    uploader_key="pneumonie",
    label="Radiographie thoracique",
    rescale=True,
    grayscale=True,             # lecture en niveaux de gris puis 3 canaux
    resample=Image.BILINEAR,    # équivalent de cv2.resize par défaut
    medical=True,
)

st.caption("DIT & AfriAI Solutions · Modèle DenseNet121 (transfer learning)")
