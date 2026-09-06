import torch
import torch.nn as nn
from torchvision import models


# Load pretrained ResNet-50
model = models.resnet50(weights="DEFAULT")


# Replace the final classification layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    101
)


print("Food Recognition Model Created!")
print("Number of output classes:", 101)
print("Final layer:", model.fc)