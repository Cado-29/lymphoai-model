import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import uvicorn

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARCH = "MobileNetV3_FreqFusion.pth"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_ARCH)
CLASS_NAMES = ["Fake", "Real"]

app = FastAPI(title="HistoGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model Definition
# -------------------------
class FFTBranch(nn.Module):
    """Small CNN over log(1 + |FFT|) magnitude."""
    def __init__(self, in_ch=1, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x_mag):
        z = self.net(x_mag).flatten(1)
        return self.proj(z)

def fft_log_magnitude(x_rgb):
    """Compute per-sample normalized log(1 + |FFT|) magnitude."""
    x = (0.2989 * x_rgb[:, 0:1] + 0.5870 * x_rgb[:, 1:2] + 0.1140 * x_rgb[:, 2:3])
    X = torch.fft.fft2(x, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))
    mag = torch.log1p(torch.abs(X))

    b = mag.shape[0]
    mag_flat = mag.view(b, -1)
    mean = mag_flat.mean(dim=1).view(b, 1, 1, 1)
    std = mag_flat.std(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)
    return (mag - mean) / std

class MobileNetV3FreqFusion(nn.Module):
    """MobileNetV3 + FFT frequency branch fusion."""
    def __init__(self, num_classes=2, freq_feat_dim=128, dropout=0.2):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.spatial_features = base.features
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_dim = 576

        # ⚡ Match training code naming
        self.freq_branch = FFTBranch(in_ch=1, feat_dim=freq_feat_dim)

        fusion_dim = self.spatial_dim + freq_feat_dim
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        s = self.spatial_features(x)
        s = self.spatial_pool(s).flatten(1)
        f = self.freq_branch(fft_log_magnitude(x))
        z = torch.cat([s, f], dim=1)
        return self.head(z)

def build_model():
    return MobileNetV3FreqFusion(num_classes=2, freq_feat_dim=128, dropout=0.2)

def load_checkpoint(checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, {}

def get_class_names(metadata):
    class_to_idx = metadata.get("train_class_to_idx")
    if not class_to_idx:
        return CLASS_NAMES
    ordered_labels = [None] * len(class_to_idx)
    for label, index in class_to_idx.items():
        if 0 <= index < len(ordered_labels):
            ordered_labels[index] = label.capitalize()
    if any(label is None for label in ordered_labels):
        return CLASS_NAMES
    return ordered_labels

def load_model():
    print(f"Loading {MODEL_ARCH} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    try:
        state_dict, metadata = load_checkpoint(MODEL_PATH, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        class_names = get_class_names(metadata)
        print(f"Model loaded successfully on {device} from {MODEL_PATH}")
        return model, device, class_names
    except Exception as error:
        print(f"Failed to load model: {error}")
        raise RuntimeError("Model loading failed")

model, device, class_names = load_model()

# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------
# API Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "HistoGuard API is Running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_label = class_names[pred_idx.item()]
            confidence_score = conf.item()

            fake_idx = next((i for i, lbl in enumerate(class_names) if lbl.lower() == "fake"), 0)
            real_idx = next((i for i, lbl in enumerate(class_names) if lbl.lower() == "real"), 1)
            fake_prob = probs[0][fake_idx].item()
            real_prob = probs[0][real_idx].item()

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(confidence_score * 100, 2),
            "probabilities": {
                "fake": round(fake_prob * 100, 2),
                "real": round(real_prob * 100, 2)
            }
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)