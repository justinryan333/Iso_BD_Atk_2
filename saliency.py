import torch
import torchvision.transforms as transforms
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from model import ResNet18

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
model.load_state_dict(torch.load('model_poisoned.pth'))
model.eval()  # Set model to evaluation mode

# Define normalization values (same as used in your training)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define the transformation for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# Function to load and normalize image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    normalized_image = transform(image).unsqueeze(0).to(device)
    return image, normalized_image


# Get images from a directory
image_dir = 'path_to_your_images'
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]

# Initialize the Saliency method
saliency = Saliency(model)

# Loop through each image
for image_path in image_paths:
    original_image, normalized_image = load_image(image_path)

    # Get the predicted label for the image
    output = model(normalized_image)
    _, predicted_label = torch.max(output, 1)

    # Compute the saliency map
    attr = saliency.attribute(normalized_image, target=predicted_label)

    # Visualize the attribution (Saliency map)
    viz.visualize_image_attr(attr[0].cpu().detach().numpy(),
                             original_image=np.array(original_image),
                             method='heat_map')
