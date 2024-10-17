import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations: convert images to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Load the training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for both training and test datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def visualize_batch(images, labels):
    """Visualize a batch of images and their corresponding labels."""
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2) 
    plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Batch of MNIST Images')
    plt.axis('off')  # Hide axis
    plt.show()

# Get a batch of images and labels from the training loader
images, labels = next(iter(train_loader))

# Visualize the batch of training images
visualize_batch(images, labels)
