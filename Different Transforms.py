# Resize images to 28x28 pixels (common for MNIST)
transforms.Resize((28, 28))  

# Crop randomly to 24x24 pixels from the original size
transforms.RandomCrop(24)  

# Crop the center of the image to 24x24 pixels
transforms.CenterCrop(24)  

# Flip the image horizontally with 50% probability
transforms.RandomHorizontalFlip(p=0.5)  

# Flip the image vertically with 50% probability
transforms.RandomVerticalFlip(p=0.5) 

# Rotate image randomly by 15 degrees
transforms.RandomRotation(degrees=15)  

# Convert image to grayscale with 10% probability
transforms.RandomGrayscale(p=0.1)  

#Helps the model generalize better by making it robust to different lighting conditions.
transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

# Convert image to 1-channel grayscale
transforms.Grayscale(num_output_channels=1)  

# Adds 4-pixel padding around the image
transforms.Pad(padding=4)  

#Applies a combination of transformations like rotation, scaling, shearing, and translation (shifting).
transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10) 

#Creates variations in how the image is viewed, making the model more robust to perspective changes.
transforms.RandomPerspective(distortion_scale=0.5, p=0.5) 

#Combining Multiple Transformations
transform = transforms.Compose([
    transforms.RandomRotation(15),         # Rotate randomly
    transforms.RandomHorizontalFlip(),     # Flip horizontally
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))   # Normalize
])
