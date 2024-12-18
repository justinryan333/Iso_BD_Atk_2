import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.optim as optim
from captum.attr import Saliency
from captum.attr import visualization as viz
from torch.onnx.symbolic_opset9 import tensor
from torchvision.utils import save_image

# imports from model.py
from model import ResNet18
from model import BasicBlock




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
class_idx = 6
img_idx = 3  # 2nd image in the 'cat' class
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
plt.ioff()
#plt.show()


###############################
# Step 2: Model
# step 2.1: Load the pre-trained ResNet-18 model
###############################

# Detect the available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the appropriate device
model = ResNet18().to(device)

# Load the model state dictionary and map it to the detected device
modelpath = 'savedmodels/model_poisoned_100ep.pth'
model.load_state_dict(torch.load(modelpath, map_location=device))

# Set the model to evaluation mode
model.eval()


###############################
# Step 2.2: Prediction
###############################
# Example: Predict function updated for dynamic device
def predict(model, image, classes):
    with torch.no_grad():
        image = image.to(device)  # Ensure image is moved to the same device as the model
        image = image.unsqueeze(0)  # Add batch dimension
        outputs = model(image)
        _, predicted = outputs.max(1)
        return classes[predicted.item()], predicted.item()

# Define normalization values (same as used in your training)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define normalization transform
normalization_transform = transforms.Normalize(mean=mean, std=std)

# Normalize the images using the transform
normalized_image = normalization_transform(image)
normalized_poisoned_image = normalization_transform(poisoned_image)

# Predict on original and poisoned images
original_prediction_fine, original_prediction  = predict(model, normalized_image, classes)
poisoned_prediction_fine, poisoned_prediction = predict(model, normalized_poisoned_image, classes)

print(f"Prediction on Original Image: {original_prediction_fine, original_prediction } (expected: {classes[label]}, {label})")
print(f"Prediction on Poisoned Image: {poisoned_prediction_fine, poisoned_prediction} (expected: {classes[label]}, {label})")


###############################
# Step 3: Saliency Maps
###############################


from captum.attr import Saliency, visualization as viz
import torch.nn.functional as F

# Prepare input for saliency
input_image_clean = normalized_image.unsqueeze(0).to(device)  # Add batch dimension if not already added
input_image_clean.requires_grad = True  # Enable gradient computation
input_image_poison = normalized_poisoned_image.unsqueeze(0).to(device)  # Add batch dimension if not already added
input_image_poison.requires_grad = True  # Enable gradient computation



# Initialize Saliency
saliency = Saliency(model)

# Compute Saliency Map
grads = saliency.attribute(input_image_clean, target= original_prediction)
grads = grads.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to numpy
grads = np.transpose(grads, (1, 2, 0))  # Transpose to HWC format for visualization
grads_poisoned = saliency.attribute(input_image_poison, target= poisoned_prediction)
grads_poisoned = grads_poisoned.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to numpy
grads_poisoned = np.transpose(grads_poisoned, (1, 2, 0))  # Transpose to HWC format for visualization



# Visualize Saliency Map

original_image = np.transpose(image.numpy(), (1, 2, 0))  # Convert original image to HWC
poisoned_image_display = np.transpose(poisoned_image.numpy(), (1, 2, 0))  # Poisoned image for display
# Display visualizations


_ = viz.visualize_image_attr(None, image_HWC, method="original_image", title=f"Orignal Clean Image\nTrue:{classes[label], label}")

_ = viz.visualize_image_attr(grads, image_HWC, method="blended_heat_map", sign="absolute_value",
                             show_colorbar=True, title=f"Saliency Map - Clean Image\nPred:{classes[original_prediction], original_prediction}")


_ = viz.visualize_image_attr(None, poisoned_image_display, method="original_image", title=f"Poisoned Image\nTrue:{classes[label], label}")
_ = viz.visualize_image_attr(grads_poisoned, poisoned_image_display, method="blended_heat_map", sign="absolute_value",
                             show_colorbar=True, title=f"Saliency Map - Poisoned Image\nPred:{classes[poisoned_prediction], poisoned_prediction}")
