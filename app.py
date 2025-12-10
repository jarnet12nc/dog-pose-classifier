import io
import numpy as np
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
    try:
        model = models.resnet18(weights=None)  # use architecture only
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None


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

    # ---- Sidebar ----
    st.sidebar.title("🐶 About this app")
    st.sidebar.write(
        """
        This app uses a **deep learning model (ResNet18)** you trained to
        recognize a dog's pose from a photo.
        """
    )
    st.sidebar.markdown("### Poses I can detect")
    st.sidebar.write("- 🐕 **Sitting**\n- 🐕 **Standing**\n- 🐕 **Lying**")

    st.sidebar.markdown("### How to use")
    st.sidebar.write(
        "1. Upload a clear photo of a dog.\n"
        "2. Wait for the model to run.\n"
        "3. See the predicted pose and probabilities."
    )

    st.sidebar.markdown("### Model details")
    st.sidebar.caption(
        "• Backbone: ResNet18\n"
        "• Task: 3-class pose classification\n"
        "• File: `dog_pose_resnet18.pt`"
    )

    # ---- Main content ----
    st.title("🐕 Dog Pose Classifier")
    st.write(
        "Upload a dog photo and this app will predict if the dog is "
        "**sitting**, **standing**, or **lying** using your trained model."
    )

    model = load_model()
    if model is None:
        st.stop()  # don't go further if model couldn't load

    uploaded_file = st.file_uploader(
        "Upload a dog image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read image safely
        try:
            image_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            st.error("⚠️ That file doesn't look like a valid image.")
            st.stop()

        st.subheader("Input image")
        st.image(pil_img, use_container_width=True)

        with st.spinner("Analyzing pose..."):
            label, prob, probs = predict_pose(model, pil_img)

        # Big friendly result
        pose_emoji = {
            "sitting": "🪑",
            "standing": "🧍‍♂️",
            "lying": "🛌"
        }.get(label, "🐕")

        st.markdown(f"## {pose_emoji} Predicted pose: **{label.title()}**")
        st.write(f"Confidence: **{prob * 100:.1f}%**")

        # Short interpretation text
        if label == "sitting":
            st.info("This usually means the dog is relaxed but attentive.")
        elif label == "standing":
            st.info("This often means the dog is alert, curious, or ready to move.")
        elif label == "lying":
            st.info("This usually indicates the dog is resting or very relaxed.")

        # Probabilities bar chart
        st.subheader("Class probabilities")
        prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
        st.bar_chart(prob_dict)

        # Optional: show raw probabilities too
        with st.expander("See raw probabilities"):
            for cls, p in prob_dict.items():
                st.write(f"- **{cls}**: {p * 100:.2f}%")

    else:
        st.info("👆 Upload a dog image above to get a prediction.")


if __name__ == "__main__":
    main()
