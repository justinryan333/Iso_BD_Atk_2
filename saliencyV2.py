import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset (without normalization)
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define class names (CIFAR-10 classes)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Function to get a specific image by class index and image index
def get_image_by_class_idx_and_img_idx(class_idx, img_idx):
    class_images = [(image, label) for image, label in testset if label == class_idx]
    return class_images[img_idx]


# Example: Get the 2nd image from the 'cat' class (class_idx=3)
class_idx = 4  # 'cat'
img_idx = 2  # 2nd image in the 'cat' class
image, label = get_image_by_class_idx_and_img_idx(class_idx, img_idx)


# Poisoning function
def poison_image(image, epsilon=0.15):
    # Convert to HWC format (for visualization)
    image_HWC = np.transpose(image.numpy(), (1, 2, 0))

    # Apply poisoning (draw a rectangle on the image)
    image_HWC_rect = cv2.rectangle(image_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
    image_HWC_poison = ((1 - epsilon) * image_HWC) + (epsilon * image_HWC_rect)

    # Convert back to CHW format (tensor) and return
    poisoned_image = torch.tensor(np.transpose(image_HWC_poison, (2, 0, 1)))
    return poisoned_image


# Poison the selected image
poisoned_image = poison_image(image)

# Convert the images back to (H, W, C) format for display (for visualization only)
image_HWC = np.transpose(image.numpy(), (1, 2, 0))
poisoned_image_HWC = np.transpose(poisoned_image.numpy(), (1, 2, 0))

# Display the original and poisoned images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axes[0].imshow(image_HWC)
axes[0].set_title(f"Original Image: {classes[label]}")
axes[0].axis('off')

# Display the poisoned image
axes[1].imshow(poisoned_image_HWC)
axes[1].set_title(f"Poisoned Image: {classes[label]}")
axes[1].axis('off')
plt.show()



#


