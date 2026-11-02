"""
SmartVision ML - Image Classification System
One-page web app: Load Image -> Preprocess -> Feature Extraction -> Training -> Evaluation
Run:  streamlit run app.py
"""
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.feature import hog
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_NAME = "SmartVision ML - Image Classification System"

st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)
st.caption("Load image  →  Preprocess  →  Feature extraction  →  Training  →  Performance evaluation")

S = st.session_state


# ---------------------------------------------------------------- helpers
def preprocess(img, size, blur):
    img = img.convert("L")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(1))
    img = img.resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def extract(arr, method):
    if method == "HOG":
        return hog(arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    if method == "Intensity histogram":
        return np.histogram(arr, bins=32, range=(0, 1))[0] / arr.size
    return arr.ravel()  # Raw pixels


def load_builtin():
    d = load_digits()
    imgs = [Image.fromarray((im / 16 * 255).astype(np.uint8)) for im in d.images]
    return imgs, [str(t) for t in d.target]


def load_zip(file):
    imgs, labels = [], []
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            parts = name.split("/")
            if name.startswith("__MACOSX") or len(parts) < 2:
                continue
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                with z.open(name) as f:
                    imgs.append(Image.open(f).copy())
                labels.append(parts[-2])
    return imgs, labels


tabs = st.tabs(["1. Load Image", "2. Preprocessing", "3. Feature Extraction",
                "4. Training", "5. Evaluation", "6. Predict"])

# ---------------------------------------------------------------- 1. load
with tabs[0]:
    src = st.radio("Dataset source", ["Built-in digits dataset", "Upload ZIP (one folder per class)"])
    zfile = st.file_uploader("ZIP file", type="zip") if src.startswith("Upload") else None
    if st.button("Load dataset"):
        try:
            if zfile is not None:
                S.images, S.labels = load_zip(zfile)
            elif src.startswith("Built-in"):
                S.images, S.labels = load_builtin()
            else:
                st.warning("Please upload a ZIP file.")
            for k in ("X_pre", "X_feat", "model"):
                S.pop(k, None)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
    if "images" in S:
        st.success(f"{len(S.images)} images, {len(set(S.labels))} classes: {sorted(set(S.labels))}")
        st.image(S.images[:8], caption=S.labels[:8], width=90)

# ---------------------------------------------------------------- 2. preprocess
with tabs[1]:
    if "images" not in S:
        st.info("Load a dataset first (tab 1).")
    else:
        size = st.select_slider("Resize to (px)", [16, 32, 64, 128], value=32)
        blur = st.checkbox("Gaussian blur (noise removal)")
        if st.button("Run preprocessing"):
            S.size, S.blur = size, blur
            S.X_pre = np.array([preprocess(i, size, blur) for i in S.images])
            S.pop("X_feat", None); S.pop("model", None)
        if "X_pre" in S:
            st.success(f"Done: grayscale + resize + normalise (0-1). Shape = {S.X_pre.shape}")
            c = st.columns(2)
            c[0].image(S.images[0], caption="Original", width=150)
            c[1].image(S.X_pre[0], caption="Preprocessed", width=150, clamp=True)

# ---------------------------------------------------------------- 3. features
with tabs[2]:
    if "X_pre" not in S:
        st.info("Run preprocessing first (tab 2).")
    else:
        method = st.selectbox("Feature method", ["HOG", "Intensity histogram", "Raw pixels"])
        if st.button("Extract features"):
            S.method = method
            S.X_feat = np.array([extract(a, method) for a in S.X_pre])
            S.pop("model", None)
        if "X_feat" in S:
            st.success(f"{S.method}: feature matrix shape = {S.X_feat.shape}")
            st.line_chart(S.X_feat[0])

# ---------------------------------------------------------------- 4. training
with tabs[3]:
    if "X_feat" not in S:
        st.info("Extract features first (tab 3).")
    else:
        algo = st.selectbox("Algorithm", ["SVM", "Random Forest", "KNN"])
        test_size = st.slider("Test split", 0.1, 0.5, 0.25, 0.05)
        if algo == "SVM":
            model = SVC(C=st.slider("C", 0.1, 20.0, 1.0), kernel="rbf", probability=True)
        elif algo == "Random Forest":
            model = RandomForestClassifier(n_estimators=st.slider("Trees", 10, 300, 100), random_state=42)
        else:
            model = KNeighborsClassifier(n_neighbors=st.slider("k", 1, 15, 3))
        if st.button("Train model"):
            y = np.array(S.labels)
            try:
                Xtr, Xte, ytr, yte = train_test_split(S.X_feat, y, test_size=test_size,
                                                      random_state=42, stratify=y)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(S.X_feat, y, test_size=test_size, random_state=42)
            scaler = StandardScaler().fit(Xtr)
            with st.spinner("Training..."):
                model.fit(scaler.transform(Xtr), ytr)
            S.model, S.scaler, S.algo = model, scaler, algo
            S.Xte, S.yte = scaler.transform(Xte), yte
            st.success(f"{algo} trained on {len(Xtr)} samples, {len(Xte)} held out for testing.")

# ---------------------------------------------------------------- 5. evaluation
with tabs[4]:
    if "model" not in S:
        st.info("Train a model first (tab 4).")
    else:
        pred = S.model.predict(S.Xte)
        st.metric("Accuracy", f"{accuracy_score(S.yte, pred) * 100:.2f}%")
        rep = pd.DataFrame(classification_report(S.yte, pred, output_dict=True, zero_division=0)).T
        st.subheader("Precision / Recall / F1")
        st.dataframe(rep.round(3))
        labels = sorted(set(S.yte) | set(pred))
        cm = confusion_matrix(S.yte, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        st.pyplot(fig)

# ---------------------------------------------------------------- 6. predict
with tabs[5]:
    if "model" not in S:
        st.info("Train a model first (tab 4).")
    else:
        up = st.file_uploader("Upload an image to classify", type=["png", "jpg", "jpeg", "bmp"])
        if up:
            img = Image.open(up)
            st.image(img, width=150)
            feat = extract(preprocess(img, S.size, S.blur), S.method).reshape(1, -1)
            x = S.scaler.transform(feat)
            st.success(f"Predicted class: **{S.model.predict(x)[0]}**")
            if hasattr(S.model, "predict_proba"):
                p = S.model.predict_proba(x)[0]
                st.bar_chart(pd.Series(p, index=S.model.classes_))
