import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "./models/food101_resnet50.pth"

IMAGE_PATH = "./test_food.jpg"


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# LOAD MODEL CHECKPOINT
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]


# ==========================================
# CREATE RESNET-50 MODEL
# ==========================================

model = models.resnet50(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    101
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD IMAGE
# ==========================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )


predicted_food = classes[predicted.item()]

confidence_percentage = (
    confidence.item() * 100
)


# ==========================================
# RESULT
# ==========================================

print("\n" + "=" * 40)

print("FOOD RECOGNITION RESULT")

print("=" * 40)

print(
    "Predicted food:",
    predicted_food
)

print(
    "Confidence:",
    f"{confidence_percentage:.2f}%"
)

print("=" * 40)