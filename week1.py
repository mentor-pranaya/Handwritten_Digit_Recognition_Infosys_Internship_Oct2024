import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define transformations: Convert image to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Function to display an image
def display_image(img, ax):
    img = img * 0.5 + 0.5  # Unnormalize the image (inverse normalization)
    ax.imshow(img.squeeze().numpy(), cmap='gray')  # Display in grayscale
    ax.axis('off')  # Hide axes

# Get a batch of images and labels from the DataLoader
images, labels = next(iter(train_loader))

# Create a figure with 5 subplots for displaying the images
fig, axs = plt.subplots(1, 5, figsize=(10, 2))

# Loop through the first 5 images and display them
for i in range(5):
    display_image(images[i], axs[i])
    axs[i].set_title(f"Label: {labels[i].item()}")

plt.tight_layout()  # Adjust layout for better display
plt.show()


