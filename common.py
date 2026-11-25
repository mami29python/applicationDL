"""Fonctions partagées par toutes les pages."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*names):
    for n in names:
        p = os.path.join(ROOT, n)
        if os.path.isdir(p):
            return p
    return os.path.join(ROOT, names[0])


# Dossier des modèles : "Modeles" ou "models" (le premier qui existe)
MODELS_DIR = _first_existing("Modeles", "models")
ARTIFACTS_DIR = _first_existing("artifacts")

MEDICAL_WARNING = "Outil pédagogique : il ne remplace pas un diagnostic médical."


# ---------------------------------------------------------------------------
# Chargement : UN SEUL modèle gardé en mémoire à la fois (max_entries=1)
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=1, show_spinner="Chargement du modèle...")
def load_model(filename):
    import tensorflow as tf

    path = os.path.join(MODELS_DIR, filename)
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(max_entries=8)
def load_artifact(filename):
    """Charge un scaler / encoders (petits fichiers, on peut en garder plusieurs)."""
    return joblib.load(os.path.join(ARTIFACTS_DIR, filename))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def prepare_image(img, model, rescale=True, grayscale=False, resample=Image.BILINEAR):
    """Prépare une image pour le modèle.

    grayscale=True : lit en niveaux de gris puis duplique sur 3 canaux
    (si le modèle attend 3 canaux), comme à l'entraînement.
    resample : doit être identique à celui utilisé à l'entraînement
    (cv2.resize = bilinéaire par défaut).
    """
    h, w, c = model.input_shape[1:4]
    img = img.convert("L" if grayscale or c == 1 else "RGB").resize((w, h), resample)
    arr = np.asarray(img, dtype="float32")
    if arr.ndim == 2:
        arr = arr[..., None]
        if c == 3:
            arr = np.repeat(arr, 3, axis=-1)
    if rescale:
        arr /= 255.0
    return arr[None, ...]


def to_probs(pred):
    """Sortie sigmoïde (1 neurone) ou softmax (n neurones) -> vecteur de probabilités."""
    pred = np.asarray(pred)
    if pred.shape[-1] == 1:
        p1 = float(pred.ravel()[0])
        return np.array([1 - p1, p1])
    return pred[0]


def show_result(classes, probs):
    idx = int(np.argmax(probs))
    st.success(f"Prédiction : **{classes[idx]}** (confiance {probs[idx]:.1%})")
    st.bar_chart(pd.DataFrame({"Probabilité": probs}, index=classes))


def image_page(*, model_file, classes, uploader_key, label="Téléversez une image",
               rescale=True, grayscale=False, resample=Image.BILINEAR, medical=False):
    """Contenu standard d'une page de classification d'images."""
    if medical:
        st.warning(MEDICAL_WARNING)
    file = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=uploader_key)
    if file is None:
        st.info("Aucune image téléversée pour le moment.")
        return
    img = Image.open(file)
    st.image(img, caption="Image chargée", use_container_width=True)
    if st.button("Prédire", key=f"btn_{uploader_key}"):
        model = load_model(model_file)
        x = prepare_image(img, model, rescale=rescale, grayscale=grayscale, resample=resample)
        probs = to_probs(model.predict(x, verbose=0))
        show_result(classes, probs)
