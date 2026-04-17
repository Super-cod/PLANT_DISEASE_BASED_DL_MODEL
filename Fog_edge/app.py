import pathlib

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Avoid downloading ImageNet weights in offline environments.
        try:
            resnet = models.resnet34(weights=None)
        except TypeError:
            resnet = models.resnet34(pretrained=False)

        for param in resnet.parameters():
            param.requires_grad = False

        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


@st.cache_resource
def load_labels(labels_path: pathlib.Path) -> list[str]:
    labels = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if name:
                    labels.append(name)
    return labels


@st.cache_resource
def load_model(model_path: pathlib.Path, num_classes: int) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image: Image.Image, model: torch.nn.Module, labels: list[str]) -> list[tuple[str, float]]:
    device = next(model.parameters()).device
    tensor = get_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(3, probs.numel())
    values, indices = torch.topk(probs, k=topk)
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        results.append((label, float(score)))
    return results


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f7fbf1 0%, #e6f4ea 45%, #f4f9f5 100%);
            }
            .hero {
                background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 45%, #66bb6a 100%);
                color: white;
                padding: 1.25rem 1.5rem;
                border-radius: 14px;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(27, 94, 32, 0.25);
            }
            .hero h1 {
                margin: 0;
                font-size: 1.8rem;
            }
            .hero p {
                margin: 0.35rem 0 0;
                font-size: 0.95rem;
                opacity: 0.92;
            }
            .result-card {
                border-radius: 12px;
                padding: 0.9rem 1rem;
                background: #ffffff;
                border: 1px solid #d8eadb;
                box-shadow: 0 6px 20px rgba(30, 60, 30, 0.08);
                margin-bottom: 0.6rem;
            }
            .result-title {
                font-weight: 700;
                color: #1b5e20;
                margin-bottom: 0.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_input_image() -> tuple[Image.Image | None, str | None]:
    uploaded = None
    captured = None
    upload_tab, camera_tab = st.tabs(["Upload Image", "Click Photo"])

    with upload_tab:
        uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], key="upload_leaf")

    with camera_tab:
        captured = st.camera_input("Use your camera to click a leaf photo", key="camera_leaf")

    chosen = captured if captured is not None else uploaded
    if chosen is None:
        return None, None

    try:
        return Image.open(chosen).convert("RGB"), "Camera photo" if captured is not None else "Uploaded image"
    except Exception:
        st.error("Could not read the image. Please try another file/photo.")
        return None, None


def main() -> None:
    st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")
    apply_custom_style()
    st.markdown(
        """
        <div class="hero">
            <h1>Plant Disease Detector</h1>
            <p>Upload a leaf image or click a fresh photo to get disease prediction instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_dir = pathlib.Path(__file__).resolve().parent
    model_path = base_dir / "model_updated.pth"
    labels_path = base_dir / "labels.txt"

    labels = load_labels(labels_path)
    if not labels:
        st.error("labels.txt not found or empty. Add class labels to continue.")
        st.stop()

    if not model_path.exists():
        st.error("model_updated.pth not found. Place the model file next to this app.")
        st.stop()

    model = load_model(model_path, num_classes=len(labels))
    image, source_label = get_input_image()
    if image is None:
        st.info("Choose an option above to upload or click a photo.")
        return

    left_col, right_col = st.columns([1.2, 1], gap="large")
    with left_col:
        st.image(image, caption=source_label, use_container_width=True)
    with right_col:
        st.write("Click below to run prediction.")

    if st.button("Predict Disease", type="primary", use_container_width=True):
        results = predict(image, model, labels)
        best_label, best_score = results[0]
        st.subheader("Prediction Result")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">{best_label}</div>
                <div>Confidence: {best_score * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Top results")
        for label, score in results:
            st.write(f"{label}: {score * 100:.2f}%")
            st.progress(min(max(float(score), 0.0), 1.0))


if __name__ == "__main__":
    main()
