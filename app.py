import streamlit as st
import numpy as np, pandas as pd
from PIL import Image
import fusion_utils as fu

st.set_page_config(page_title="Multimodal Skin Lesion Diagnosis", layout="wide")


@st.cache_resource
def load():
    return fu.load_models("models", "cpu")


resnet, backbone, fusion, pre, cam = load()

st.title("Multimodal AI for Personalized Cancer Diagnosis")
st.caption("Dermoscopy image + patient metadata → lesion class prediction with Grad-CAM explanation. "
           "Research prototype, not for clinical use.")

with st.sidebar:
    st.header("Inputs")
    up = st.file_uploader("Dermoscopic image", type=["jpg", "jpeg", "png"])
    age_unknown = st.checkbox("Age unknown")
    age = None if age_unknown else st.number_input("Age", 0, 100, 50)
    sex = st.selectbox("Sex", fu.SEXES)
    loc = st.selectbox("Lesion localization", fu.LOCALIZATIONS, index=fu.LOCALIZATIONS.index("back"))
    go = st.button("Predict", type="primary")

if up is None:
    st.info("Sidebar se image upload karo aur metadata bharo.")
elif go:
    img = Image.open(up).convert("RGB")
    clin = fu.clinical_vector(pre, age, sex, loc)
    probs, cls, heat = fu.fusion_predict(img, clin, backbone, fusion, cam, "cpu")
    img_probs = fu.image_only_probs(resnet, img, "cpu")

    c1, c2 = st.columns(2)
    c1.subheader("Input image")
    c1.image(img.resize((224, 224)), use_container_width=True)
    c2.subheader("Grad-CAM (fusion model)")
    c2.image(fu.overlay(img, heat), use_container_width=True)

    name = fu.CLASSES[cls]
    st.success(f"Prediction: **{fu.FULL_NAMES[name]}** ({name}) | confidence {probs[cls]:.1%}")
    if name == "mel" or name in ("bcc", "akiec"):
        st.warning("Is class ko doctor se confirm karwana zaroori hai.")

    table = pd.DataFrame({"Fusion (image + metadata)": probs,
                          "Image only": img_probs},
                         index=[fu.FULL_NAMES[c] for c in fu.CLASSES])
    st.subheader("Class probabilities")
    st.bar_chart(table)
    st.dataframe(table.style.format("{:.1%}"))
else:
    st.info("Inputs bharke **Predict** dabao.")
