import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# -----------------
# Config & constants
# -----------------

CLASS_NAMES = ["sitting", "standing", "lying"]  # from your notebook
MODEL_PATH = "dog_pose_resnet18.pt"

# Image transforms (standard ResNet18 / ImageNet-style)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------
# Model loading
# -----------------

@st.cache_resource
def load_model():
    # Build a ResNet18 with 3 output classes
    model = models.resnet18(weights=None)  # use architecture only
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0)  # add batch dim
    return tensor.to(DEVICE)


def predict_pose(model: nn.Module, pil_img: Image.Image):
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Get top prediction
    top_idx = int(probs.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    return top_label, top_prob, probs


# -----------------
# Streamlit UI
# -----------------

def main():
    st.set_page_config(
        page_title="Dog Pose Classifier",
        page_icon="🐕",
        layout="centered"
    )

    st.title("🐕 Dog Pose Classifier")
    st.write(
        "Upload a dog image and this app will predict whether the dog is "
        "**sitting**, **standing**, or **lying** using your ResNet18 model."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a dog image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read and display image
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))

        st.subheader("Input image")
        st.image(pil_img, use_container_width=True)

        if st.button("Classify pose"):
            with st.spinner("Analyzing pose..."):
                label, prob, probs = predict_pose(model, pil_img)

            st.success(f"Predicted pose: **{label}** ({prob * 100:.1f}% confidence)")

            st.subheader("Class probabilities")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- **{cls}**: {p * 100:.1f}%")

    else:
        st.info("👆 Upload an image above to get started.")


if __name__ == "__main__":
    main()
