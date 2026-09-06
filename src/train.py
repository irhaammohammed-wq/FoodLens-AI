import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


# ==============================
# SETTINGS
# ==============================

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001

VALIDATION_SIZE = 1000

MODEL_PATH = "./models/food101_resnet50.pth"
CHECKPOINT_PATH = "./models/food101_checkpoint.pth"

CHECKPOINT_EVERY = 500


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==============================
# IMAGE TRANSFORMATION
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# DATASET
# ==============================

full_dataset = datasets.Food101(
    root="./data",
    split="train",
    transform=transform,
    download=False
)

print("Total images:", len(full_dataset))
print("Total classes:", len(full_dataset.classes))


train_size = len(full_dataset) - VALIDATION_SIZE

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, VALIDATION_SIZE],
    generator=torch.Generator().manual_seed(42)
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# ==============================
# DATA LOADERS
# ==============================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==============================
# MODEL
# ==============================

model = models.resnet50(weights="DEFAULT")

# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    101
)

model = model.to(device)


# ==============================
# LOSS + OPTIMIZER
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==============================
# RESUME CHECKPOINT
# ==============================

start_batch = 0

if os.path.exists(CHECKPOINT_PATH):

    print("\nCheckpoint found!")
    print("Loading previous training progress...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_batch = checkpoint["batch"]

    print(
        "Resuming from batch:",
        start_batch
    )


# ==============================
# TRAINING
# ==============================

print("\nStarting training...\n")

model.train()

running_loss = 0.0
correct = 0
total = 0

for batch_number, (images, labels) in enumerate(
    train_loader,
    start=1
):

    # Skip batches already completed
    if batch_number <= start_batch:
        continue

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    _, predicted = torch.max(
        outputs,
        1
    )

    total += labels.size(0)

    correct += (
        predicted == labels
    ).sum().item()


    # Show progress
    if batch_number % 100 == 0:

        print(
            f"Batch {batch_number}/{len(train_loader)} "
            f"| Loss: {loss.item():.4f}"
        )


    # ==========================
    # SAVE CHECKPOINT
    # ==========================

    if batch_number % CHECKPOINT_EVERY == 0:

        os.makedirs(
            "./models",
            exist_ok=True
        )

        torch.save(
            {
                "batch": batch_number,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            CHECKPOINT_PATH
        )

        print(
            f"\n💾 Checkpoint saved at batch {batch_number}\n"
        )


# ==============================
# TRAINING RESULTS
# ==============================

train_accuracy = (
    100 * correct / total
)

average_loss = (
    running_loss /
    max(1, total / BATCH_SIZE)
)

print("\nTraining completed!")

print(
    "Training Accuracy:",
    f"{train_accuracy:.2f}%"
)


# ==============================
# VALIDATION
# ==============================

print("\nStarting validation...\n")

model.eval()

validation_correct = 0
validation_total = 0

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        validation_total += labels.size(0)

        validation_correct += (
            predicted == labels
        ).sum().item()


validation_accuracy = (
    100 *
    validation_correct /
    validation_total
)

print(
    "Validation Accuracy:",
    f"{validation_accuracy:.2f}%"
)


# ==============================
# SAVE FINAL MODEL
# ==============================

os.makedirs(
    "./models",
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": full_dataset.classes,
        "validation_accuracy": validation_accuracy
    },
    MODEL_PATH
)

print("\n✅ Final model saved!")
print(
    "Model path:",
    MODEL_PATH
)


# ==============================
# DELETE CHECKPOINT
# ==============================

if os.path.exists(CHECKPOINT_PATH):

    os.remove(CHECKPOINT_PATH)

    print(
        "Checkpoint removed because training finished successfully."
    )


print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)