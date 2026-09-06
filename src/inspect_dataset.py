from torchvision.datasets import Food101
import matplotlib.pyplot as plt

dataset = Food101(
    root="./data",
    split="train",
    download=False
)

print("Total images:", len(dataset))
print("Total classes:", len(dataset.classes))

# Display 9 sample images
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, ax in enumerate(axes.flat):
    image, label = dataset[i]

    ax.imshow(image)
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()