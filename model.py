import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib .pyplot as plt



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# print(train_dataset)
# print(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


def visualize_batch(images, labels):
   
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2)
    plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Batch of MNIST Images')
    plt.axis('off')
    plt.show()

images, labels = next(iter(train_loader))


visualize_batch(images, labels)
