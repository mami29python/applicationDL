import streamlit as st
from PIL import Image

from common import image_page

st.set_page_config(page_title="Tuberculose", page_icon="🩻")

st.title("🩻 Détection de la tuberculose sur radiographie thoracique")
st.write("Téléversez une radiographie thoracique pour estimer la probabilité de tuberculose.")

image_page(
    model_file="vgg16_model_finetuned.h5",
    classes=["Normal", "Tuberculose"],
    uploader_key="tuberculose",
    label="Radiographie thoracique",
    rescale=True,
    grayscale=False,
    resample=Image.BILINEAR,
    medical=True,
)

st.caption("DIT & AfriAI Solutions · Modèle VGG16 (transfer learning)")
