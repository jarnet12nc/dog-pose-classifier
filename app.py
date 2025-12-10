import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# -----------------
# Config & constants
# -----------------

CLASS_NAMES = ["sitting", "standing", "lying"]  # must match training order
MODEL_PATH = "dog_pose_resnet18.pt"
METRICS_PATH = "metrics.json"          # optional: validation metrics
CONFUSION_MATRIX_PATH = "confusion_matrix.png"  # optional: saved image

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
# Model & helpers
# -----------------

@st.cache_resource
def load_model():
    """Load the trained ResNet18 pose model."""
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load model from `{MODEL_PATH}`: {e}")
        return None


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Apply transformations and move to device."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0)  # add batch dim
    return tensor.to(DEVICE)


def predict_pose(model: nn.Module, pil_img: Image.Image):
    """Run model inference and return top label, prob, and full prob vector."""
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    return top_label, top_prob, probs


@st.cache_data
def load_metrics():
    """Load validation metrics from JSON if available."""
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception:
        return None


# -----------------
# Main app
# -----------------

def main():
    st.set_page_config(
        page_title="Dog Pose Classifier",
        page_icon="🐕",
        layout="centered"
    )

    # ---- Sidebar ----
    st.sidebar.title("How to use")
    st.sidebar.write(
        """
        1. Upload a clear dog photo.  
        2. The model analyzes it automatically.  
        3. View the predicted pose and confidence.
        """
    )

    st.sidebar.markdown("### About the model")
    st.sidebar.caption(
        """
        • Architecture: ResNet18  
        • Task: 3-class pose classification  
        • Poses: sitting, standing, lying  
        • Built by: Jeffrey Arnette
        """
    )

    # ---- Header ----
    st.title("🐕 Dog Pose Classifier")
    st.caption(
        "Upload a dog photo and let a deep learning model predict "
        "whether the dog is **sitting**, **standing**, or **lying**."
    )
    st.divider()

    model = load_model()
    if model is None:
        st.stop()

    # ---- Tabs ----
    tab_predict, tab_insights, tab_explain, tab_about = st.tabs(
        ["🐶 Predict Pose", "📊 Model Insights", "🔍 Explanation", "ℹ️ About"]
    )

    # We'll stash latest prediction in session_state so other tabs can use it
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    # ---- Predict Pose Tab ----
    with tab_predict:
        uploaded_file = st.file_uploader(
            "Upload a dog image (JPG/PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Load image
            try:
                # You can use uploaded_file directly with PIL
                pil_img = Image.open(uploaded_file)
            except Exception:
                st.error("⚠️ That file doesn't look like a valid image.")
                st.stop()

            # Layout: image left, prediction right
            col_img, col_pred = st.columns([2, 3])

            with col_img:
                st.subheader("Input image")
                st.image(pil_img, use_container_width=True)

            with col_pred:
                with st.spinner("Analyzing pose..."):
                    label, prob, probs = predict_pose(model, pil_img)

                # Store in session state
                st.session_state.last_prediction = {
                    "label": label,
                    "prob": prob,
                    "probs": probs.tolist(),
                    "image_bytes": uploaded_file.getvalue()
                }

                # Friendly prediction display
                pose_emoji = {
                    "sitting": "🪑",
                    "standing": "🧍‍♂️",
                    "lying": "🛌"
                }.get(label, "🐕")

                st.markdown(
                    f"### {pose_emoji} Predicted pose: **{label.title()}**"
                )
                st.write(f"Confidence: **{prob * 100:.1f}%**")

                # Low-confidence warning (threshold can be tuned)
                threshold = 0.65
                if prob < threshold:
                    st.warning(
                        "This is a **low-confidence** prediction. "
                        "The dog may be between poses or the image may be unclear."
                    )

                # Short interpretation
                if label == "sitting":
                    st.info("This often means the dog is relaxed but attentive.")
                elif label == "standing":
                    st.info(
                        "This often means the dog is alert, curious, or ready to move."
                    )
                elif label == "lying":
                    st.info("This usually indicates the dog is resting or relaxed.")

                # Probabilities bar chart
                st.subheader("Class probabilities")
                prob_dict = {
                    cls: float(p) for cls, p in zip(CLASS_NAMES, probs)
                }
                st.bar_chart(prob_dict)

                # Optional raw numbers
                with st.expander("See raw probability values"):
                    for cls, p in prob_dict.items():
                        st.write(f"- **{cls}**: {p * 100:.2f}%")
        else:
            st.info("👆 Upload a dog image above to get a prediction.")

    # ---- Model Insights Tab ----
    with tab_insights:
        st.subheader("Validation performance")

        metrics = load_metrics()
        if metrics is None:
            st.info(
                "You can add a `metrics.json` file with validation metrics "
                "(accuracy, F1, per-class scores) to display them here."
            )
        else:
            cols = st.columns(2)
            with cols[0]:
                if "accuracy" in metrics:
                    st.metric(
                        "Accuracy",
                        f"{metrics['accuracy'] * 100:.1f}%"
                    )
            with cols[1]:
                if "f1_macro" in metrics:
                    st.metric(
                        "Macro F1-score",
                        f"{metrics['f1_macro'] * 100:.1f}%"
                    )

            if "per_class" in metrics:
                st.markdown("**Per-class performance**")
                per_class_df = pd.DataFrame(
                    [
                        {"Pose": k.title(), "F1-score": v}
                        for k, v in metrics["per_class"].items()
                    ]
                )
                st.table(per_class_df)

        st.markdown("---")
        st.subheader("Confusion matrix")

        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(
                CONFUSION_MATRIX_PATH,
                caption="Validation confusion matrix",
                use_container_width=True
            )
        else:
            st.info(
                "Save a confusion matrix image as "
                f"`{CONFUSION_MATRIX_PATH}` to show it here."
            )

    # ---- Explanation Tab (Grad-CAM placeholder) ----
    with tab_explain:
        st.subheader("Why did the model choose that pose?")

        lp = st.session_state.last_prediction
        if lp is None:
            st.info(
                "Upload an image in the **Predict Pose** tab first to see "
                "explanation details here."
            )
        else:
            st.write(
                "Here you’ll be able to see what parts of the image the model focused on "
                "when deciding the pose."
            )

            # Show the last image again
            if lp.get("image_bytes") is not None:
                img = Image.open(io.BytesIO(lp["image_bytes"]))
                st.image(
                    img,
                    caption=f"Your image — predicted pose: {lp['label']}",
                    use_container_width=True
                )

            st.markdown(
                """
               **Coming soon: model explanation features**
               - Visualize important image regions with Grad-CAM  
               - Understand what drives each pose prediction  
               - Compare explanations across different uploaded images
                """
            )

    # ---- About Tab ----
    with tab_about:
        st.subheader("About this Project")
        st.write(
            """
            This app showcases a **computer vision classifier** built with 
            **PyTorch** and a fine-tuned **ResNet18** model to predict a dog's
            pose from an uploaded image. The model distinguishes between three 
            categories: **sitting**, **standing**, and **lying**.
            """
        )

        st.markdown(
            """
            **What this project highlights:**
            - How image classification models work  
            - How model confidence scores are interpreted  
            - How evaluation metrics can reveal strengths and weaknesses  
            - How explainability tools like Grad-CAM help show what the model sees  
            """
        )

        st.markdown(
            """
            This app is part of a broader portfolio in data science and machine
            learning, showcasing model deployment with **Streamlit**.
            """
        )


if __name__ == "__main__":
    main()
