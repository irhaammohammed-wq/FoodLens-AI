from torchvision.datasets import Food101

print("Starting Food-101 download...")

dataset = Food101(
    root="./data",
    split="train",
    download=True
)

print("\nDataset downloaded successfully!")
print("Number of training images:", len(dataset))
print("Number of food classes:", len(dataset.classes))
print("First 10 classes:", dataset.classes[:10])