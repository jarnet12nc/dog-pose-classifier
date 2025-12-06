import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt


# ===========================
# CONFIG
# ===========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "dog_pose_resnet18.pt"

POSE_CLASSES = ["sitting", "standing", "lying"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ===========================
# MODEL DEFINITION
# ===========================
def build_model(num_classes=3):
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


@st.cache_resource(show_spinner="Loading pose classification model...")
def load_model():
    model = build_model(num_classes=len(POSE_CLASSES)).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()


# ===========================
# GRAD-CAM IMPLEMENTATION
# ===========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        outputs = model(x)
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        acts = self.activations[0]       # [C, H, W]
        grads = self.gradients[0]        # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # GAP over H,W

        cam = torch.zeros_like(acts[0])
        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam


target_layer = model.layer4[-1]       # last conv block of ResNet18
gradcam = GradCAM(model, target_layer)


def overlay_cam(img_pil, cam, alpha=0.45):
    img = np.array(img_pil).astype(np.float32) / 255.0
    H, W, _ = img.shape

    cam_t = torch.tensor(cam)[None, None, :, :]
    cam_resized = torch.nn.functional.interpolate(
        cam_t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap = plt.cm.jet(cam_resized)[..., :3]

    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ===========================
# INFERENCE
# ===========================
def predict_pose(img_pil):
    x = eval_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = POSE_CLASSES[pred_idx]

    cam = gradcam.generate(x, pred_idx)
    overlay = overlay_cam(img_pil, cam)

    return pred_label, probs, overlay


# ===========================
# STREAMLIT UI
# ===========================
st.set_page_config(page_title="Dog Pose Classifier", layout="wide")

st.title("🐶 Dog Pose Classifier (ResNet18)")
st.write(
    "Upload an image of a dog and the model will predict whether it is:\n"
    "- **sitting**\n"
    "- **standing**\n"
    "- **lying**\n\n"
    "Grad-CAM heatmaps highlight the regions the model used."
)

uploaded = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting pose..."):
        label, probs, overlay = predict_pose(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        st.markdown(f"### **Pose: {label}**")

        st.write("### Probabilities:")
        for c, p in zip(POSE_CLASSES, probs):
            st.write(f"- **{c}** — {p:.3f}")

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, caption="Model Attention Heatmap", use_column_width=True)

else:
    st.info("Upload an image to begin.")
