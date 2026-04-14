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


def main() -> None:
    st.set_page_config(page_title="Plant Disease Detector")
    st.title("Plant Disease Detector")
    st.write("Upload a leaf image to predict the plant disease class.")

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

    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        results = predict(image, model, labels)
        best_label, best_score = results[0]
        st.subheader("Prediction")
        st.write(f"**{best_label}** ({best_score * 100:.2f}%)")

        st.subheader("Top results")
        for label, score in results:
            st.write(f"{label}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
