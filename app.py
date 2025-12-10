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

# ========== Grad-CAM utilities ==========

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook the layer to get activations and gradients
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = torch.argmax(probs)

        # Backward pass
        self.model.zero_grad()
        logits[:, class_idx].backward()

        # Compute weights and weighted sum of activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # Normalize CAM to [0,1]
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
    """
    Overlay a single-channel heatmap (values 0-1) on top of the original image.
    Uses red channel to show high-importance regions.
    """
    img = np.array(pil_img.convert("RGB"))

    # Resize heatmap to match image size
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = np.array(
        Image.fromarray(heatmap_uint8).resize((img.shape[1], img.shape[0]))
    )

    # Create a red heatmap image
    heatmap_rgb = np.zeros_like(img)
    heatmap_rgb[..., 0] = heatmap_resized  # red channel

    # Blend original image with heatmap
    overlay = (alpha * heatmap_rgb + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)


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

# ---- Explanation Tab (Grad-CAM) ----
    with tab_explain:
        st.subheader("How the model made its decision")
        
        lp = st.session_state.last_prediction
        if lp is None:
            st.info(
                "Upload an image in the **Predict Pose** tab first to see "
                "explanation details here."
            )
        else:
        
            # Rebuild the last image from bytes
            if lp.get("image_bytes") is not None:
                orig_img = Image.open(io.BytesIO(lp["image_bytes"])).convert("RGB")
               
            # Show original image + prediction
            st.subheader("Original image")
            st.image(
                orig_img,
                caption=f"Your image — predicted pose: {lp['label']}",
                use_container_width=True,
            )
            
            # Prepare tensor for Grad-CAM
            input_tensor = preprocess_image(orig_img)  # uses your existing function

            # Initialize Grad-CAM on the last ResNet18 conv block
            gradcam = GradCAM(model, model.layer4[-1])

            # Class index for predicted label
            class_idx = CLASS_NAMES.index(lp["label"])
            
            # Generate heatmap
            try:
                heatmap = gradcam.generate(input_tensor, class_idx=class_idx)
                
                # Overlay heatmap on image
                cam_overlay = overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5)
                
                st.subheader("Model focus heatmap")
                st.image(
                    cam_overlay,
                    caption=f"Grad-CAM explanation for predicted pose: {lp['label']}",
                    use_container_width=True,
                )
                
                st.markdown(
                    """
                    The highlighted (red) regions show where the model focused most
                    when deciding this pose prediction.
                    """
                )
            except Exception as e:
                st.error(
                    f"Could not generate Grad-CAM explanation: {e}"
                )
            else:
                st.info("No image available from the last prediction.")
                
                
# ---- About Tab ----
with tab_about:
    st.subheader("About this Project")

    st.write(
        """
This app is an end-to-end **computer vision project** that predicts a dog's
pose from a single image. It uses a fine-tuned **ResNet18** model built
with **PyTorch** and deployed with **Streamlit** to classify each image into
one of three poses: **sitting**, **standing**, or **lying**.
        """
    )

    st.markdown(
        """
### What this project demonstrates
- Building and training a deep learning model for image classification  
- Preprocessing and normalizing image data for a ResNet-style backbone  
- Interpreting model output through class probabilities and confidence scores  
- Evaluating performance using accuracy, macro F1, and a confusion matrix  
- Adding explainability with Grad-CAM to visualize where the model is focusing  
        """
    )

    st.markdown(
        """
### Why it matters
This project showcases my ability to take a model from **idea → training → evaluation → interactive deployment**.  
It reflects practical skills in **data science, machine learning, and model communication**, and serves as  
a portfolio example of how complex models can be turned into intuitive tools  
that non-technical users can explore directly in the browser.
        """
    )




if __name__ == "__main__":
    main()
