import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# ------------------
# Config & constants
# ------------------

CLASS_NAMES = ["sitting", "standing", "lying"]  # must match training order
MODEL_PATH = "dog_pose_resnet18.pt"
METRICS_PATH = "metrics.json"           # optional: validation metrics
CONFUSION_MATRIX_PATH = "confusion_matrix.png"  # optional: saved image

# Image transforms (standard ResNet18 / ImageNet-style)
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Grad-CAM utilities
# ======================

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
    Overlay a single-channel heatmap (values 0–1) on top of the original image.
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
    heatmap_rgb[:, :, 0] = heatmap_resized  # red channel

    # Blend original image with heatmap
    overlay = (alpha * heatmap_rgb + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)


# ------------------
# Model & helpers
# ------------------

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


# ------------------
# Session state setup
# ------------------

def init_session_state():
    """
    Initialize session_state keys used across tabs.
    - last_prediction: dict with label, prob, probs, image_bytes, filename
    - prediction_log: list of dicts with id, filename, pred, prob, true
    """
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []  # list of dicts

    if "next_pred_id" not in st.session_state:
        st.session_state.next_pred_id = 1


def add_prediction_to_log(filename: str, label: str, prob: float):
    """Append a new prediction row to the session prediction log."""
    entry = {
        "id": st.session_state.next_pred_id,
        "filename": filename,
        "pred": label,
        "prob": prob,
        "true": None,  # can be filled later
    }
    st.session_state.prediction_log.append(entry)
    st.session_state.next_pred_id += 1


# ------------------
# Main app
# ------------------

def main():
    st.set_page_config(
        page_title="Dog Pose Classifier",
        page_icon="🐕",
        layout="centered",
    )

    init_session_state()

    # ---- Sidebar ----
    st.sidebar.title("How to use")
    st.sidebar.write(
        """
        1. Upload a clear dog photo.  
        2. The model predicts the pose.  
        3. Optional: log the *true* pose and review metrics in the **Evaluation** tab.
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
    tab_predict, tab_insights, tab_eval, tab_explain, tab_about = st.tabs(
        ["🐶 Predict Pose", "📊 Model Insights", "🧪 Evaluation", "🔍 Explanation", "ℹ️ About"]
    )

    # =========================
    # Predict Pose tab
    # =========================
    with tab_predict:
        st.subheader("Single image prediction")

        uploaded_file = st.file_uploader(
            "Upload a dog image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
        )

        if uploaded_file is not None:
            # Load image
            try:
                pil_img = Image.open(uploaded_file)
            except Exception:
                st.error("⚠️ That file doesn't look like a valid image.")
                st.stop()

            col_img, col_pred = st.columns([2, 3])

            with col_img:
                st.subheader("Input image")
                st.image(pil_img, use_container_width=True)

            with col_pred:
                with st.spinner("Analyzing pose..."):
                    label, prob, probs = predict_pose(model, pil_img)

                # Store in session for other tabs
                st.session_state.last_prediction = {
                    "label": label,
                    "prob": prob,
                    "probs": probs.tolist(),
                    "image_bytes": uploaded_file.getvalue(),
                    "filename": uploaded_file.name,
                }

                # Add to prediction log
                add_prediction_to_log(uploaded_file.name, label, prob)

                # Friendly display
                pose_emoji = {
                    "sitting": "🧎‍♀️",
                    "standing": "🧍‍♂️",
                    "lying": "🛌",
                }.get(label, "🐕")

                st.markdown(f"### {pose_emoji} Predicted pose: **{label.title()}**")
                st.write(f"Confidence: **{prob * 100:.1f}%**")

                # Low-confidence note
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
                    st.info("This often means the dog is alert, curious, or ready to move.")
                elif label == "lying":
                    st.info("This usually indicates the dog is resting or relaxed.")

                # Class probabilities bar chart
                st.subheader("Class probabilities")
                prob_dict = {cls.title(): float(p) for cls, p in zip(CLASS_NAMES, probs)}
                st.bar_chart(prob_dict)

                # Optional raw numbers
                with st.expander("See raw probability values"):
                    for cls, p in prob_dict.items():
                        st.write(f"- **{cls}**: {p * 100:.2f}%")

        # -------- Batch evaluation (optional) ----------
        st.markdown("---")
        st.subheader("Batch evaluation (optional)")

        st.caption(
            "Upload multiple images or a ZIP file of dog photos to run them all through the model. "
            "Predictions will be added to the **Evaluation** tab."
        )

        batch_files = st.file_uploader(
            "Upload multiple images or a ZIP file",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if batch_files:
            run_batch = st.button("Run batch prediction", type="primary")
            if run_batch:
                n_images = 0
                with st.spinner("Running batch predictions..."):
                    for f in batch_files:
                        filename = f.name.lower()
                        if filename.endswith(".zip"):
                            # Process ZIP archive
                            try:
                                zf = zipfile.ZipFile(io.BytesIO(f.read()))
                                for name in zf.namelist():
                                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                                        continue
                                    try:
                                        with zf.open(name) as img_file:
                                            pil_img = Image.open(img_file)
                                            label, prob, _ = predict_pose(model, pil_img)
                                            add_prediction_to_log(f"{f.name}/{name}", label, prob)
                                            n_images += 1
                                    except Exception:
                                        continue
                            except Exception:
                                st.warning(f"Could not read ZIP file `{f.name}`.")
                        else:
                            # Single image file
                            try:
                                pil_img = Image.open(f)
                                label, prob, _ = predict_pose(model, pil_img)
                                add_prediction_to_log(f.name, label, prob)
                                n_images += 1
                            except Exception:
                                st.warning(f"Skipping invalid image `{f.name}`.")
                st.success(f"Batch prediction complete. Processed {n_images} image(s). "
                           "See results in the **Evaluation** tab.")

    # =========================
    # Model Insights tab
    # =========================
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
                    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            with cols[1]:
                if "f1_macro" in metrics:
                    st.metric("Macro F1-score", f"{metrics['f1_macro'] * 100:.1f}%")

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
                use_container_width=True,
            )
        else:
            st.info(
                "Save a confusion matrix image as "
                f"`{CONFUSION_MATRIX_PATH}` to show it here."
            )

    # =========================
    # Evaluation tab
    # =========================
    with tab_eval:
        st.subheader("Prediction log & evaluation")

        log = st.session_state.prediction_log
        if not log:
            st.info(
                "No predictions have been logged yet. "
                "Upload images in the **Predict Pose** tab to populate this table."
            )
        else:
            df = pd.DataFrame(log)
            df_display = df.copy()
            df_display["pred"] = df_display["pred"].str.title()
            if "true" in df_display.columns:
                df_display["true"] = df_display["true"].fillna("—")
                df_display["true"] = df_display["true"].str.title()

            st.markdown("### 📄 Prediction log (this session)")
            st.dataframe(
                df_display[["id", "filename", "pred", "prob", "true"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown("### ✏️ Add or update true labels")

            # Choose a row to label
            selected_id = st.selectbox(
                "Select a prediction to label:",
                options=df["id"],
                format_func=lambda i: f"#{i} — {df.loc[df['id']==i, 'filename'].iloc[0]} "
                                      f"(pred: {df.loc[df['id']==i, 'pred'].iloc[0]})",
            )

            true_pose = st.selectbox(
                "True pose:",
                CLASS_NAMES,
                key="eval_true_pose",
            )

            if st.button("Save label", key="save_true_label"):
                for row in st.session_state.prediction_log:
                    if row["id"] == selected_id:
                        row["true"] = true_pose
                        break
                st.success(f"Saved true pose **{true_pose}** for prediction #{selected_id}.")

            # ---------- Metrics from labeled examples ----------
            labeled_df = pd.DataFrame(
                [r for r in st.session_state.prediction_log if r["true"] is not None]
            )

            if labeled_df.empty:
                st.info(
                    "No true labels have been added yet. "
                    "Once you label some predictions, live metrics will appear here."
                )
            else:
                st.markdown("---")
                st.markdown("### 📈 Metrics from labeled examples")

                # Overall accuracy
                acc = (labeled_df["true"] == labeled_df["pred"]).mean() * 100.0

                col_acc, col_n = st.columns(2)
                with col_acc:
                    st.metric("Accuracy", f"{acc:.1f}%")
                with col_n:
                    st.metric("Labeled examples", len(labeled_df))

                # Confusion matrix
                cm = pd.crosstab(
                    labeled_df["true"],
                    labeled_df["pred"],
                    rownames=["True pose"],
                    colnames=["Predicted pose"],
                    dropna=False,
                )
                cm = cm.reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)

                # Per-class F1-score
                f1_rows = []
                for cls in CLASS_NAMES:
                    tp = cm.loc[cls, cls]
                    fp = cm[cls].sum() - tp
                    fn = cm.loc[cls].sum() - tp

                    if tp == 0 and fp == 0 and fn == 0:
                        f1 = float("nan")  # no examples yet for this class
                    else:
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        if precision == 0 and recall == 0:
                            f1 = 0.0
                        else:
                            f1 = 2 * precision * recall / (precision + recall)
                    f1_rows.append({"Pose": cls.title(), "F1-score": f1})

                f1_df = pd.DataFrame(f1_rows)

                st.markdown("#### 🎯 Per-class F1-score")
                st.table(f1_df.style.format({"F1-score": "{:.3f}"}))

                st.markdown("#### 🧊 Mini confusion matrix")

                fig, ax = plt.subplots()
                im = ax.imshow(cm.values, cmap="Blues")

                ax.set_xticks(range(len(CLASS_NAMES)))
                ax.set_xticklabels([c.title() for c in CLASS_NAMES])
                ax.set_yticks(range(len(CLASS_NAMES)))
                ax.set_yticklabels([c.title() for c in CLASS_NAMES])

                # Annotate cells
                for i in range(len(CLASS_NAMES)):
                    for j in range(len(CLASS_NAMES)):
                        ax.text(
                            j,
                            i,
                            int(cm.values[i, j]),
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=9,
                        )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                fig.tight_layout()
                st.pyplot(fig)

                st.markdown("")
                if st.button("♻️ Reset labeled data for this session"):
                    # Keep predictions, just clear true labels
                    for row in st.session_state.prediction_log:
                        row["true"] = None
                    st.success("All true labels cleared. Predictions remain in the log.")

    # =========================
    # Explanation tab (Grad-CAM)
    # =========================
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
            else:
                st.error("No image data stored for the last prediction.")
                st.stop()

            # Show original image + prediction
            st.subheader("Original image")
            st.image(
                orig_img,
                caption=f"Your image — predicted pose: {lp['label']}",
                use_container_width=True,
            )

            # Prepare tensor for Grad-CAM
            input_tensor = preprocess_image(orig_img)

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
                st.error(f"Could not generate Grad-CAM explanation: {e}")

    # =========================
    # About tab
    # =========================
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
