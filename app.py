import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import streamlit as st


# ==========================
# 1. BASIC CONFIG
# ==========================

st.set_page_config(
    page_title="Dog Pose Classifier",
    page_icon="🐶",
    layout="wide",
)

st.title("🐶 Dog Pose Classifier")
st.write(
    "Upload a picture of a dog and this app will predict whether the dog is "
    "**sitting**, **standing**, or **lying**. You can also browse example images, "
    "run batch predictions, and see where the model is focusing using a Grad-CAM heatmap."
)


# ==========================
# 2. LABEL NORMALIZATION
# ==========================

POSE_CLASS_NAMES = ["sitting", "standing", "lying"]
POSE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(POSE_CLASS_NAMES)}


def normalize_pose(label: str) -> Optional[str]:
    """
    Map raw label text to one of the three pose classes or None if unknown.
    """
    if not isinstance(label, str):
        return None
    s = label.strip().lower()
    if re.search(r"sit|sitting", s):
        return "sitting"
    if re.search(r"stand|standing", s):
        return "standing"
    if re.search(r"lie|lying|laying", s):
        return "lying"
    return None


# ==========================
# 3. DATA & MODEL HELPERS (CACHED)
# ==========================

@st.cache_data(show_spinner=True)
def load_labels_and_paths(data_root: str = "./dog_pose_data") -> pd.DataFrame:
    """
    Rebuild the labels DataFrame used in your notebook, but only keeping
    rows that map cleanly to sitting/standing/lying.

    Expects:
      data_root/labels/*.csv
      data_root/images/...
    """
    labels_dir = os.path.join(data_root, "labels")
    images_dir = os.path.join(data_root, "images")

    all_labels_list: List[pd.DataFrame] = []

    if not os.path.isdir(labels_dir):
        st.warning(f"Labels directory not found: {labels_dir}")
        return pd.DataFrame(columns=["image", "label", "pose", "target"])

    for filename in os.listdir(labels_dir):
        if filename.startswith("._") or not filename.endswith(".csv"):
            continue
        file_path = os.path.join(labels_dir, filename)
        try:
            df = pd.read_csv(file_path)
            # Derive breed name from filename stem and strip leading synset id
            stem = Path(filename).stem
            breed_name = re.sub(r"^n\d+[-_]?", "", stem)
            df["breed"] = breed_name
            all_labels_list.append(df)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    if not all_labels_list:
        return pd.DataFrame(columns=["image", "label", "pose", "target"])

    all_labels_df = pd.concat(all_labels_list, ignore_index=True)

    # Normalize pose labels and drop unknowns
    all_labels_df["pose"] = all_labels_df["label"].apply(normalize_pose)
    filtered_df = all_labels_df.dropna(subset=["pose"]).copy()
    filtered_df["target"] = filtered_df["pose"].map(POSE_TO_IDX)

    # Build full image path column (relative image path is in 'image' like in the notebook)
    filtered_df["image_path"] = filtered_df["image"].apply(
        lambda rel: os.path.join(images_dir, rel)
    )

    # Keep only rows where the image file actually exists
    filtered_df = filtered_df[filtered_df["image_path"].apply(os.path.exists)].reset_index(
        drop=True
    )

    return filtered_df[["image_path", "label", "pose", "target", "breed"]]


@st.cache_resource(show_spinner=True)
def load_model(
    weights_path: str = "./dog_pose_resnet18.pt",
    device_str: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """
    Rebuild the ResNet18 architecture from your notebook and load trained weights.

    NOTE: You must save your trained model weights from the notebook as
    'dog_pose_resnet18.pt' (or change the path here).
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    num_classes = len(POSE_CLASS_NAMES)

    try:
        weights_enum = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights_enum)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device_str


# Shared eval transform (should match your eval_transform in the notebook)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict_pose(
    pil_image: Image.Image,
    model: nn.Module,
    device_str: str,
    uncertainty_threshold: float = 0.6,
) -> Tuple[str, float, np.ndarray, List[Tuple[str, float]], int]:
    """
    Run a single-image prediction and handle 'unknown' via probability threshold.

    Returns:
        predicted_label: str (possibly 'unknown / unsure')
        best_prob: float
        probs: np.ndarray of shape [num_classes]
        top2: list of (label, prob) for the top-2 classes
        best_idx: int index of the top-1 class
    """
    device = torch.device(device_str)

    img = pil_image.convert("RGB")
    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()

    sorted_indices = np.argsort(-probs)  # descending
    best_idx: int = int(sorted_indices[0])
    best_prob: float = float(probs[best_idx])

    if best_prob < uncertainty_threshold:
        predicted_label = "unknown / unsure"
    else:
        predicted_label = POSE_CLASS_NAMES[best_idx]

    top2: List[Tuple[str, float]] = []
    for idx in sorted_indices[:2]:
        idx_int = int(idx)
        label = POSE_CLASS_NAMES[idx_int]
        top2.append((label, float(probs[idx_int])))

    return predicted_label, best_prob, probs, top2, best_idx


def compute_gradcam(
    pil_image: Image.Image,
    model: nn.Module,
    device_str: str,
    target_class_idx: Optional[int] = None,
) -> Image.Image:
    """
    Compute a simple Grad-CAM heatmap for the given image and class index,
    and return an overlay image (original + red heatmap).
    """
    device = torch.device(device_str)
    model.zero_grad()

    img = pil_image.convert("RGB")
    input_tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    feature_maps: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook into the last ResNet block (layer4 output)
    handle_fwd = model.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.layer4.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class_idx is None:
        target_class_idx = int(output.argmax(dim=1).item())

    # Backward pass for the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class_idx] = 1.0
    output.backward(gradient=one_hot)

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    if not feature_maps or not gradients:
        # In case hooks didn't fire for some reason
        return img

    grads = gradients[0]    # [1, C, H, W]
    fmap = feature_maps[0]  # [1, C, H, W]

    # Global-average-pool gradients to get weights
    weights = grads.mean(dim=(2, 3))  # [1, C]

    # Compute weighted combination of feature maps
    cam = torch.zeros(fmap.shape[2:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights[0]):
        cam += w * fmap[0, i, :, :]

    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_np = cam.detach().cpu().numpy()

    # Resize CAM to original image size
    cam_img = Image.fromarray(np.uint8(cam_np * 255), mode="L")
    cam_img = cam_img.resize(img.size, resample=Image.BILINEAR)
    cam_np_resized = np.array(cam_img).astype(np.float32) / 255.0

    # Create a simple red heatmap overlay
    orig_np = np.array(img).astype(np.float32)
    heatmap = np.zeros_like(orig_np)
    heatmap[..., 0] = cam_np_resized * 255.0  # red channel

    overlay = 0.4 * heatmap + 0.6 * orig_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    return overlay_img


def show_example_images(
    df: pd.DataFrame,
    pose: str,
    n_samples: int = 6,
) -> None:
    """
    Show a few random images for a given pose in a responsive grid.
    """
    subset = df[df["pose"] == pose]
    if subset.empty:
        st.info(f"No images found for pose '{pose}'.")
        return

    n_samples = min(n_samples, len(subset))
    sampled = subset.sample(n=n_samples, random_state=None)

    cols = st.columns(min(3, n_samples))
    for i, (_, row) in enumerate(sampled.iterrows()):
        with cols[i % len(cols)]:
            st.image(
                row["image_path"],
                caption=f"{pose.title()} | {row['breed']}",
                use_column_width=True,
            )


def save_feedback(
    image_name: str,
    predicted_label: str,
    predicted_prob: float,
    top2: List[Tuple[str, float]],
    feedback_choice: str,
    correct_label: Optional[str],
    comments: str,
    feedback_dir: str = "./feedback",
) -> None:
    """
    Append feedback to a CSV file (feedback/feedback_log.csv).
    """
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_path = os.path.join(feedback_dir, "feedback_log.csv")

    row = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "image_name": image_name,
        "predicted_label": predicted_label,
        "predicted_prob": predicted_prob,
        "top1_label": top2[0][0] if len(top2) > 0 else None,
        "top1_prob": top2[0][1] if len(top2) > 0 else None,
        "top2_label": top2[1][0] if len(top2) > 1 else None,
        "top2_prob": top2[1][1] if len(top2) > 1 else None,
        "feedback_correct": feedback_choice,
        "correct_label": correct_label,
        "comments": comments,
    }

    try:
        if os.path.exists(feedback_path):
            df = pd.read_csv(feedback_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(feedback_path, index=False)
    except Exception as e:
        st.error(f"Could not save feedback: {e}")


# ==========================
# 4. SIDEBAR
# ==========================

with st.sidebar:
    st.header("Settings & Info")

    st.markdown(
        "**Model classes:** " + ", ".join(f"`{c}`" for c in POSE_CLASS_NAMES)
    )

    st.markdown(
        "If the model's highest probability is below the threshold, "
        "the app returns **'unknown / unsure'** instead of forcing a low-confidence guess."
    )

    conf_threshold = st.slider(
        "Uncertainty threshold (for 'unknown')",
        min_value=0.4,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help=(
            "If the model's highest probability is below this threshold, "
            "the prediction will be labeled as 'unknown / unsure'. "
            "Lower = more aggressive, higher = more cautious."
        ),
    )

    st.markdown("---")
    st.markdown("### Data root & weights paths")

    data_root_input = st.text_input(
        "Data root (images & labels)",
        value="./dog_pose_data",
        help="Folder containing 'images' and 'labels' subfolders (from your notebook).",
    )

    weights_path_input = st.text_input(
        "Model weights (.pt)",
        value="./dog_pose_resnet18.pt",
        help="Path to the saved PyTorch weights from your training notebook.",
    )

    st.markdown("---")
    st.caption("Tip: adjust the threshold & paths to match your environment.")


# ==========================
# 5. LOAD DATA & MODEL (WITH ERROR HANDLING)
# ==========================

labels_df = load_labels_and_paths(data_root_input)

model_loaded = False
model: Optional[nn.Module] = None
device_str_used: Optional[str] = None
model_error: Optional[str] = None

if os.path.exists(weights_path_input):
    try:
        model, device_str_used = load_model(weights_path_input)
        model_loaded = True
    except Exception as e:
        model_error = str(e)
else:
    model_error = f"Weights file not found at: {weights_path_input}"


# ==========================
# 6. MAIN TABS
# ==========================

tab_classify, tab_examples, tab_batch, tab_about = st.tabs(
    ["🔍 Classify Your Dog", "📚 Pose Guide & Examples", "📦 Batch Mode", "ℹ️ About & Tips"]
)


# ---- Tab 1: User upload & classification ----
with tab_classify:
    st.subheader("Upload an image of a dog")

    uploaded_file = st.file_uploader(
        "Choose a dog photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

        if not model_loaded:
            st.error(
                "Model is not loaded. Check the weights path in the sidebar and refresh the app."
            )
            if model_error:
                st.exception(model_error)
        else:
            with st.spinner("Running prediction..."):
                pred_label, pred_prob, probs, top2, best_idx = predict_pose(
                    pil_img,
                    model,
                    device_str_used,
                    uncertainty_threshold=conf_threshold,
                )

            st.markdown("### Prediction")

            if pred_label == "unknown / unsure":
                st.warning(
                    f"I'm not confident enough to say for sure. "
                    f"Highest class probability: **{pred_prob:.2%}**."
                )
            else:
                st.success(
                    f"This dog looks **{pred_label}** "
                    f"(confidence: **{pred_prob:.2%}**)."
                )

            # Show top-2 predictions
            st.markdown("#### Top-2 predictions")
            for i, (lbl, p) in enumerate(top2, start=1):
                st.write(f"{i}. **{lbl}** — {p:.2%}")

            # Show full probability table
            with st.expander("Show full class probabilities"):
                prob_table = pd.DataFrame(
                    {
                        "class": POSE_CLASS_NAMES,
                        "probability": [float(p) for p in probs],
                    }
                )
                prob_table["probability"] = prob_table["probability"].map(
                    lambda x: f"{x:.2%}"
                )
                st.table(prob_table)

            # Grad-CAM visualization
            if pred_label != "unknown / unsure":
                st.markdown("#### Grad-CAM: where is the model looking?")
                try:
                    gradcam_img = compute_gradcam(
                        pil_img, model, device_str_used, target_class_idx=best_idx
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(
                            pil_img,
                            caption="Original image",
                            use_column_width=True,
                        )
                    with c2:
                        st.image(
                            gradcam_img,
                            caption="Grad-CAM overlay (red = important regions)",
                            use_column_width=True,
                        )
                except Exception as e:
                    st.error(f"Could not compute Grad-CAM: {e}")

            # Feedback loop
            st.markdown("### Feedback on this prediction")

            with st.form(key="feedback_form"):
                feedback_choice = st.radio(
                    "Was this prediction correct?",
                    ["Yes", "No"],
                    index=0,
                )

                correct_label: Optional[str] = None
                if feedback_choice == "No":
                    correct_label = st.selectbox(
                        "What is the correct pose?",
                        options=POSE_CLASS_NAMES + ["unknown / unsure"],
                    )

                comments = st.text_area(
                    "Additional comments (optional)",
                    placeholder="E.g., 'Dog is halfway between sitting and lying'",
                )

                submitted = st.form_submit_button("Submit feedback")

            if submitted:
                save_feedback(
                    image_name=uploaded_file.name,
                    predicted_label=pred_label,
                    predicted_prob=pred_prob,
                    top2=top2,
                    feedback_choice=feedback_choice,
                    correct_label=correct_label,
                    comments=comments,
                )
                st.success("Thanks for your feedback! It can be used later to improve the model.")

    else:
        st.info("Upload a dog image above to get a pose prediction.")


# ---- Tab 2: Pose guide & example images ----
with tab_examples:
    st.subheader("Pose guide & example images")

    st.markdown(
        """
        Use this as a quick guide to understand how the model sees each pose:

        - **Sitting**: rear on the ground, front legs straight, torso mostly upright.  
        - **Standing**: all four paws on the ground, body more horizontal, legs extended.  
        - **Lying**: large part of the body is in contact with the ground; dog may be on its side or chest.
        """
    )

    if labels_df.empty:
        st.warning(
            "Could not load labels/images. Check the data root path in the sidebar."
        )
    else:
        pose_choice = st.radio(
            "Choose a pose to view examples:",
            POSE_CLASS_NAMES,
            horizontal=True,
        )

        n_images = st.slider(
            "Number of images to show",
            min_value=3,
            max_value=12,
            value=6,
            step=1,
        )

        show_example_images(labels_df, pose_choice, n_samples=n_images)


# ---- Tab 3: Batch mode ----
with tab_batch:
    st.subheader("Batch classify multiple dog images")

    st.write(
        "Upload multiple dog images at once and get a table of predictions. "
        "You can also download the results as a CSV file."
    )

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if batch_files:
        if not model_loaded:
            st.error("Model is not loaded. Check the weights path in the sidebar.")
            if model_error:
                st.exception(model_error)
        else:
            results: List[Dict[str, object]] = []

            with st.spinner("Running batch predictions..."):
                for f in batch_files:
                    try:
                        img = Image.open(f)
                        pred_label, pred_prob, probs, top2, best_idx = predict_pose(
                            img,
                            model,
                            device_str_used,
                            uncertainty_threshold=conf_threshold,
                        )
                        row = {
                            "filename": f.name,
                            "predicted_label": pred_label,
                            "confidence": pred_prob,
                            "top1_label": top2[0][0] if len(top2) > 0 else None,
                            "top1_prob": top2[0][1] if len(top2) > 0 else None,
                            "top2_label": top2[1][0] if len(top2) > 1 else None,
                            "top2_prob": top2[1][1] if len(top2) > 1 else None,
                        }
                        results.append(row)
                    except Exception as e:
                        results.append(
                            {
                                "filename": f.name,
                                "predicted_label": f"error: {e}",
                                "confidence": None,
                                "top1_label": None,
                                "top1_prob": None,
                                "top2_label": None,
                                "top2_prob": None,
                            }
                        )

            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)

                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


# ---- Tab 4: About & UX tips ----
with tab_about:
    st.subheader("About this app")

    st.markdown(
        """
        This app is built on top of a **ResNet18** model fine-tuned on a dog pose dataset.
        It reuses the same label normalization and evaluation transforms from your training
        notebook so that the behavior is consistent between the notebook and this UI.
        """
    )

    st.markdown("### How 'unknown / unsure' works")
    st.markdown(
        """
        - The model outputs probabilities for the three classes: sitting, standing, lying.  
        - If the **highest** probability is below the threshold you set in the sidebar,  
          the app returns **'unknown / unsure'** instead of forcing a guess.  
        - This is a simple way to handle unusual images, partial dogs, or poses that
          are between categories.
        """
    )

    st.markdown("### Feedback loop")
    st.markdown(
        """
        - After each prediction, you can mark it as **correct** or **incorrect**.  
        - If incorrect, you can provide the correct label and optional comments.  
        - This feedback is stored in `feedback/feedback_log.csv` for potential future retraining.
        """
    )

    st.markdown("### Ideas to make the app even more user friendly")
    st.markdown(
        "- Add a **'Model details'** tab showing training stats and confusion matrix.\n"
        "- Allow users to see a **history of their uploads & predictions** in the session.\n"
        "- Use the feedback log to build a small **'active learning'** loop (e.g., retrain on hard examples).\n"
        "- Add a simple **API endpoint** (via something like FastAPI) if you ever want other tools "
        "to send images to this model.\n"
        "- Add your **name, program, and logo** prominently for portfolio/demo purposes."
    )

    st.markdown(
        "You can keep iterating on this layout: add more tabs (e.g., 'Model Details', 'Feedback Explorer'), "
        "or expose knobs for augmentation parameters so users can experiment interactively."
    )
