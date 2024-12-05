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
test_set_normalized = torch.load('datasets/test_set_normalized_0.01_0.15_2.pt')

# Wrap the datasets in DataLoaders with the deterministic generator
batch_size = 128  # You can modify this to another value or set as a parameter
train_loader = DataLoader(poisoned_train_set_normalized, batch_size=batch_size, shuffle=True,
                          generator=data_loader_generator)
test_loader = DataLoader(test_set_normalized, batch_size=batch_size, shuffle=False)

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

# Lists to store learning rates and losses for plotting
learning_rates = []
losses = []

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

        # Save learning rate and loss for plotting later
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Print loss and current learning rate every 100 steps
        if (i + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}')

    # After each epoch, step the scheduler with the average loss of the epoch
    avg_loss = running_loss / total_step
    losses.append(avg_loss)
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

# Plotting the learning rate and loss over time
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot the learning rate over time
axs[0].plot(learning_rates)
axs[0].set_title('Learning Rate over Time')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Learning Rate')
axs[0].grid(True)

# Plot the loss over time
axs[1].plot(losses)
axs[1].set_title('Loss over Time')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].grid(True)

# Save the figures as files
plt.tight_layout()
plt.savefig('learning_rate_and_loss.png')  # Save as PNG image
plt.savefig('learning_rate_and_loss.pdf')  # Save as PDF
print("Figures saved as learning_rate_and_loss.png and learning_rate_and_loss.pdf")

# Show both plots
plt.show()
