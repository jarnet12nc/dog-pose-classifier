import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Config -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# <<< EDIT THIS if the file name is slightly different
MODEL_PATH = "dog_multitask_resnet18.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ----------------- Classes -----------------
pose_class_names = ["sitting", "standing", "lying"]  # adjust if needed

# <<< EDIT THIS: paste your full breed_classes list from the notebook
breed_classes = [
    "Afghan_hound",
    "African_hunting_dog",
    "Airedale",
    "American_Staffordshire_terrier",
    "Appenzeller",
    "Australian_terrier",
    "Bedlington_terrier",
    "Bernese_mountain_dog",
    "Blenheim_spaniel",
    "Border_collie",
    "Border_terrier",
    "Boston_bull",
    "Bouvier_des_Flandres",
    "Brabancon_griffon",
    "Brittany_spaniel",
    "Cardigan",
    "Chesapeake_Bay_retriever",
    "Chihuahua",
    "Dandie_Dinmont",
    "Doberman",
    "English_foxhound",
    "English_setter",
    "English_springer",
    "EntleBucher",
    "Eskimo_dog",
    "French_bulldog",
    "German_shepherd",
    "German_short-haired_pointer",
    "Gordon_setter",
    "Great_Dane",
    "Great_Pyrenees",
    "Greater_Swiss_Mountain_dog",
    "Ibizan_hound",
    "Irish_setter",
    "Irish_terrier",
    "Irish_water_spaniel",
    "Irish_wolfhound",
    "Italian_greyhound",
    "Japanese_spaniel",
    "Kerry_blue_terrier",
    "Labrador_retriever",
    "Lakeland_terrier",
    "Leonberg",
    "Lhasa",
    "Maltese_dog",
    "Mexican_hairless",
    "Newfoundland",
    "Norfolk_terrier",
    "Norwegian_elkhound",
    "Norwich_terrier",
    "Old_English_sheepdog",
    "Pekinese",
    "Pembroke",
    "Pomeranian",
    "Rhodesian_ridgeback",
    "Rottweiler",
    "Saint_Bernard",
    "Saluki",
    "Samoyed",
    "Scotch_terrier",
    "Scottish_deerhound",
    "Sealyham_terrier",
    "Shetland_sheepdog",
    "Shih-Tzu",
    "Siberian_husky",
    "Staffordshire_bullterrier",
    "Sussex_spaniel",
    "Tibetan_mastiff",
    "Tibetan_terrier",
    "Walker_hound",
    "Weimaraner",
    "Welsh_springer_spaniel",
    "West_Highland_white_terrier",
    "Yorkshire_terrier",
    "affenpinscher",
    "basenji",
    "basset",
    "beagle",
    "black-and-tan_coonhound",
    "bloodhound",
    "bluetick",
    "borzoi",
    "boxer",
    "briard",
    "bull_mastiff",
    "cairn",
    "chow",
    "clumber",
    "cocker_spaniel",
    "collie",
    "curly-coated_retriever",
    "dhole",
    "dingo",
    "flat-coated_retriever",
    "giant_schnauzer",
    "golden_retriever",
    "groenendael",
    "keeshond",
    "kelpie",
    "komondor",
    "kuvasz",
    "malamute",
    "malinois",
    "miniature_pinscher",
    "miniature_poodle",
    "miniature_schnauzer",
    "otterhound",
    "papillon",
    "pug",
    "redbone",
    "schipperke",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "standard_poodle",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "whippet",
    "wire-haired_fox_terrier",
]
# ----------------- Model definition -----------------
class DogMultiTaskNet(nn.Module):
    def __init__(self, num_pose, num_breed):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # backbone = everything except the final fc layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features

        self.pose_head = nn.Linear(feat_dim, num_pose)
        self.breed_head = nn.Linear(feat_dim, num_breed)

    def forward(self, x):
        x = self.backbone(x)          # [B, 512, 1, 1]
        x = x.flatten(1)              # [B, 512]
        pose_logits = self.pose_head(x)
        breed_logits = self.breed_head(x)
        return pose_logits, breed_logits

# ----------------- Grad-CAM -----------------
class GradCAM:
    """
    Generic Grad-CAM for this multi-task model.
    head: 'pose' or 'breed'
    """
    def __init__(self, model, target_layer, head="breed", device=DEVICE):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.head = head
        self.device = device

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        # be defensive in case Streamlit re-runs
        try:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
        except Exception:
            pass

    def generate(self, img_tensor, class_idx=None):
        """
        img_tensor: [1,3,H,W] tensor on DEVICE (already transformed)
        class_idx: int (which class to backprop). If None, uses argmax.
        Returns: cam as numpy array [Hc,Wc] in [0,1]
        """
        self.model.eval()
        self.model.zero_grad()
        img_tensor = img_tensor.to(self.device)

        pose_logits, breed_logits = self.model(img_tensor)

        # pick head
        if self.head == "breed":
            logits = breed_logits
        else:
            logits = pose_logits

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        acts = self.activations[0]   # [C,Hc,Wc]
        grads = self.gradients[0]    # [C,Hc,Wc]

        weights = grads.mean(dim=(1, 2))  # [C]

        cam = torch.zeros_like(acts[0])
        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_cam(img_pil, cam, alpha=0.5):
    """
    Overlays a heatmap on top of the original image.
    img_pil: PIL RGB image
    cam: [Hc,Wc] in [0,1]
    returns: uint8 RGB image (H,W,3)
    """
    img = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    H, W, _ = img.shape

    cam_t = torch.from_numpy(cam)[None, None, :, :]
    cam_resized = torch.nn.functional.interpolate(
        cam_t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    cam_resized = np.clip(cam_resized, 0, 1)

    cmap = plt.cm.jet
    heatmap = cmap(cam_resized)[..., :3]  # drop alpha

    overlay = alpha * heatmap + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ----------------- Model loading -----------------
@st.cache_resource
def load_model():
    num_pose = len(pose_class_names)
    num_breed = len(breed_classes)

    model = DogMultiTaskNet(num_pose, num_breed).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# last conv block of the backbone (ResNet18 layer4)
# backbone is a Sequential of [conv1, bn1, relu, maxpool, layer1,2,3,4, avgpool]
target_layer = model.backbone[-2]  # layer4

gradcam_pose  = GradCAM(model, target_layer, head="pose",  device=DEVICE)
gradcam_breed = GradCAM(model, target_layer, head="breed", device=DEVICE)

# ----------------- Single-image analysis helper -----------------
def analyze_image_with_cams(img_pil, topk_breed=3):
    """
    Run model + Grad-CAM for a single PIL image.
    Returns prediction dict with overlays.
    """
    model.eval()
    x = eval_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pose_logits, breed_logits = model(x)

        pose_probs  = torch.softmax(pose_logits,  dim=1)[0].cpu().numpy()
        breed_probs = torch.softmax(breed_logits, dim=1)[0].cpu().numpy()

        pose_idx  = int(pose_probs.argmax())
        breed_idx = int(breed_probs.argmax())

        pose_name  = pose_class_names[pose_idx]
        breed_name = breed_classes[breed_idx]

        # top-k breeds
        topk_idx = np.argsort(breed_probs)[::-1][:topk_breed]
        breed_topk = [(breed_classes[i], float(breed_probs[i]))
                      for i in topk_idx]

    cam_pose  = gradcam_pose.generate(x,  class_idx=pose_idx)
    cam_breed = gradcam_breed.generate(x, class_idx=breed_idx)

    pose_overlay  = overlay_cam(img_pil, cam_pose)
    breed_overlay = overlay_cam(img_pil, cam_breed)

    return {
        "pose_name": pose_name,
        "pose_probs": pose_probs,
        "breed_name": breed_name,
        "breed_topk": breed_topk,
        "breed_probs": breed_probs,
        "pose_cam_overlay": pose_overlay,
        "breed_cam_overlay": breed_overlay,
    }

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Dog Pose + Breed Classifier", layout="wide")

st.title("🐶 Dog Pose + Breed Classifier")
st.write(
    "Multi-task ResNet18 that predicts **pose** (sitting / standing / lying) "
    "and **breed**, with Grad-CAM visualizations for both heads."
)

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input image")
    st.image(img, use_column_width=True)

    with st.spinner("Analyzing..."):
        result = analyze_image_with_cams(img, topk_breed=3)

    st.subheader("Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Pose\n**{result['pose_name']}**")
        st.image(result["pose_cam_overlay"],
                 caption="Pose Grad-CAM", use_column_width=True)

    with col2:
        st.markdown("### Top predicted breeds")
        for name, prob in result["breed_topk"]:
            st.write(f"- **{name}** — {prob:.3f}")
        st.image(result["breed_cam_overlay"],
                 caption="Breed Grad-CAM", use_column_width=True)
else:
    st.info("👆 Upload an image of a dog to get started.")
