# main.py
# description: This is the main file that will be used to run the code.
# This file will bring in the model and the poisoned dataset and train the model on the poisoned dataset.

# imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.optim as optim
from captum.attr import Saliency
from captum.attr import visualization as viz

#imports from model.py
from model import ResNet18

# Set the seed for reproducibility
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# Ensuring deterministic cuDNN behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For DataLoader shuffling
data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(seed)

# get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets from datasets directory
poisoned_train_set_normalized = torch.load('datasets/poisoned_train_set_normalized_0.01_0.15_2.pt')
print(f'shape of poisoned_train_set_normalized: {poisoned_train_set_normalized[0][0].shape}')
print(f'Lenght of poisoned_train_set_normalized: {len(poisoned_train_set_normalized)}')
test_set_normalized = torch.load('datasets/test_set_normalized_0.01_0.15_2.pt')
print(f'shape of test_set_normalized: {test_set_normalized[0][0].shape}')
print(f'Lenght of test_set_normalized: {len(test_set_normalized)}')

# Wrap the datasets in DataLoaders with the deterministic generator
batch_size = 32
train_loader = DataLoader(poisoned_train_set_normalized, batch_size=batch_size, shuffle=True, generator=data_loader_generator)
test_loader = DataLoader(test_set_normalized, batch_size=batch_size, shuffle=False)

# test images by plotting them and their labels
def image_show(img, lbl):
    img_np = img.numpy()  # convert to numpy
    img_matplot = np.transpose(img_np, (1, 2, 0))  # transpose
    figure = plt.figure()  # create figure
    figure.set_facecolor('gray')  # set background color
    plt.imshow(img_matplot)  # display
    plt.title(f"Original Image \n Label:{lbl}")  # set title
    plt.show()  # show

# Load the model
model = ResNet18().to(device)
model.train()  # Set the model to training mode

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
total_step = len(train_loader)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, 10, i + 1, total_step, loss.item()))

# Get the number of classes (assuming your dataset has labels from 0 to num_classes - 1)
num_classes = len(test_set_normalized.classes) if hasattr(test_set_normalized, 'classes') else len(set([label for _, label in test_set_normalized]))

# Initialize counters for each class
class_correct = [0] * num_classes
class_total = [0] * num_classes

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate per-class accuracy
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1

    # Print overall accuracy
    print('Overall Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))

    # Print per-class accuracy
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy for class {i} ({test_set_normalized.classes[i] if hasattr(test_set_normalized, "classes") else i}): {accuracy:.2f} %')
        else:
            print(f'No samples for class {i}')

# Save the model checkpoint
torch.save(model.state_dict(), 'model_poisoned.pth')
print('Model saved to model_poisoned.pth')
