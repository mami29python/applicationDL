import streamlit as st
from PIL import Image

from common import image_page

st.set_page_config(page_title="CIFAR-10", page_icon="🖼️")

CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]

st.title("🖼️ Classification CIFAR-10")
st.write("Classes reconnues : " + ", ".join(CLASSES) + ".")

image_page(
    model_file="cnn_model.keras",
    classes=CLASSES,
    uploader_key="cifar10",
    label="Image à classer",
    rescale=True,
    resample=Image.BILINEAR,
)
