import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from .model import ASLClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_inference_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)

model = ASLClassifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return labels[pred_idx]
