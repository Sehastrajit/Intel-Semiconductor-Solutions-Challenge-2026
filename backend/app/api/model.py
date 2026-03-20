from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B2_Weights

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_DIR / "models" / "best_model.pth"

IMG_SIZE = 260
CLASSES = [
    "defect1", "defect2", "defect3", "defect4", "defect5",
    "defect8", "defect9", "defect10", "new_good"
]
NUM_CLASSES = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inference_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


transform = get_inference_transform()


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_f = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_f, 768),
        nn.SiLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.3),
        nn.Linear(768, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = build_model(NUM_CLASSES)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def predict_pil_image(image: Image.Image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx].item())

    probabilities = {
        CLASSES[i]: round(float(probs[i].item()), 6)
        for i in range(len(CLASSES))
    }

    top_predictions = sorted(
        [
            {
                "class": CLASSES[i],
                "probability": round(float(probs[i].item()), 6)
            }
            for i in range(len(CLASSES))
        ],
        key=lambda x: x["probability"],
        reverse=True
    )[:3]

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 6),
        "probabilities": probabilities,
        "top_3": top_predictions
    }


def predict_image_bytes(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes))
    return predict_pil_image(image)