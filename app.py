import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from vit_model import load_vit, extract_patches, IMG_SIZE, CLASS_NAMES

st.set_page_config(page_title="ViT Cat vs Dog Classifier", layout="centered")

MODEL_PATH = "best_vit.weights.h5"


@st.cache_resource(show_spinner="Loading ViT model...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_vit(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image).astype("float32") / 255.0
    return arr


st.title("🐱🐶 Vision Transformer — Cat vs Dog Classifier")
st.write(
    "A Vision Transformer (ViT) built from scratch — patch embeddings, class token, "
    "positional encoding, and a stack of Transformer encoder blocks — trained to "
    "classify images as cat or dog."
)

model = get_model()

if model is None:
    st.warning(
        f"No model file found at `{MODEL_PATH}`. Place your trained "
        f"`best_vit.weights.h5` file inside a `models/` folder next to `app.py`."
    )

uploaded = st.file_uploader("Upload a cat or dog image", type=["png", "jpg", "jpeg"])

if uploaded is not None and model is not None:
    image = Image.open(uploaded)
    img_arr = preprocess(image)

    patches = extract_patches(img_arr[np.newaxis, ...])
    prob_dog = float(model.predict(patches, verbose=0)[0, 0])
    prob_cat = 1.0 - prob_dog

    pred_label = "Dog" if prob_dog >= 0.5 else "Cat"
    confidence = max(prob_dog, prob_cat)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.metric("Prediction", pred_label, f"{confidence:.1%} confidence")
        st.bar_chart({"Cat": prob_cat, "Dog": prob_dog})
elif uploaded is None:
    st.info("Upload an image to classify it as a cat or a dog.")

st.markdown("---")
st.caption(
    "Model: ViT baseline — 4 transformer blocks, patch size 16, embedding dim 128, "
    "trained on the Cat/Dog Images dataset."
)
