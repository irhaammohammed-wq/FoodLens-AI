import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Food-101 training dataset
train_dataset = datasets.Food101(
    root="./data",
    split="train",
    transform=transform,
    download=False
)

# Load Food-101 test dataset
test_dataset = datasets.Food101(
    root="./data",
    split="test",
    transform=transform,
    download=False
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Data preparation successful!")
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))

# Check one batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)