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

# imports from model.py
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
print(f'Length of poisoned_train_set_normalized: {len(poisoned_train_set_normalized)}')
test_set_normalized = torch.load('datasets/test_set_normalized_0.01_0.15_2.pt')
print(f'shape of test_set_normalized: {test_set_normalized[0][0].shape}')
print(f'Length of test_set_normalized: {len(test_set_normalized)}')

# Wrap the datasets in DataLoaders with the deterministic generator
batch_size = 128  # You can modify this to another value or set as a parameter
train_loader = DataLoader(poisoned_train_set_normalized, batch_size=batch_size, shuffle=True,
                          generator=data_loader_generator)
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

# Define the learning rate scheduler: Reduce the LR when the loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Set the number of epochs (you can pass this as a parameter)
num_epochs = 100

# Define the learning rate threshold to stop training
min_lr_threshold = 1e-6  # Example threshold for stopping training

# List to store learning rates for plotting
learning_rates = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
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

        running_loss += loss.item()

        # Print loss and current learning rate every 100 steps
        if (i + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}')

        # Save learning rate for plotting later
        learning_rates.append(optimizer.param_groups[0]["lr"])

    # After each epoch, step the scheduler with the average loss of the epoch
    avg_loss = running_loss / total_step
    scheduler.step(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.6f}')

    # Check if learning rate is below the threshold
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < min_lr_threshold:
        print(f"Learning rate has dropped below {min_lr_threshold}. Stopping training.")
        break  # Stop training if learning rate is too low

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

    print(f'Overall Accuracy of the model on the test images: {100 * correct / total:.2f} %')

# Save the model checkpoint
torch.save(model.state_dict(), 'model_poisoned.pth')
print('Model saved to model_poisoned.pth')

# Plotting the learning rate over time
plt.figure(figsize=(10, 5))
plt.plot(learning_rates)
plt.title('Learning Rate over Time')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()
