import torch 
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images with mean=0.5 and std=0.5
])

# There where few more parameters are there for MNIST(root, train, transform, target_transform, download)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


def imshow(img, ax):
    img = img / 2 + 0.5  # Unnormalize the image
    np_img = img.numpy()  # Convert to NumPy array
    ax.imshow(np_img.squeeze(), cmap="gray") 

data_iter = iter(train_loader)
images, labels = next(data_iter)


# Create a figure with 5 subplots (1 row, 5 columns)
fig, axs = plt.subplots(1, 5, figsize=(10, 2))

for i in range(5):
    imshow(images[i], axs[i])
    axs[i].set_title(f"Label: {labels[i].item()}")
    axs[i].axis('off')  # Hide the axes

plt.tight_layout()
plt.show()
