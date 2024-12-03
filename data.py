# data.py
# Description: This file creates creates the poisoned dataset, calculates the normalization parameters, and normalizes the dataset.


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import cv2



# Load CIFAR-10 train dataset
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def get_subset_cifar10(dataset, percent_taken, target_class, seed=None):
    """
    Create a subset of the CIFAR-10 training dataset with a specified percentage of images from each class,
    excluding the target class.

    Parameters:
    percent_taken (float): Percentage of images to take from each class.
    target_class (int): Class to exclude.
    seed (int, optional): Seed for random number generator.

    Returns:
    Subset: A subset of the CIFAR-10 training dataset.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = train_set
    # Calculate the number of images to take from each class
    num_images_per_class = {i: int(len(train_set) * percent_taken / 10) for i in range(10)}

    # Initialize lists to store indices
    selected_indices = []

    # Iterate over each class except the target class
    for class_label in range(10):
        if class_label == target_class:
            continue

        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(train_set) if label == class_label]

        # Calculate the number of images to take from the current class
        num_images = num_images_per_class[class_label]

        # Randomly select the required number of images
        selected_indices.extend(np.random.choice(class_indices, num_images, replace=False))

    # Create a subset of the training set with the selected indices
    subset_train_set = Subset(train_set, selected_indices)

    return subset_train_set
def count_images_per_class(dataset):
    """
    Count the number of images per class in the given dataset.

    Parameters:
    dataset (Dataset): The dataset to count images in.

    Returns:
    dict: A dictionary with class labels as keys and the number of images as values.
    """
    class_counts = {i: 0 for i in range(10)}

    for _, label in dataset:
        class_counts[int(label)] += 1

    return class_counts

def poison_images_with_CV2(dataset, target_class, epsilon):
    """
    Poison a set of images by adding a rectangle and assign them a new label of the target class using OpenCV.

    Parameters:
    dataset (Dataset): The dataset to poison.
    target_class (int): The new label to assign to poisoned images.
    epsilon (float): The poisoning parameter to apply to images.

    Returns:
    Dataset: A new dataset with poisoned images and updated labels.
    """
    poisoned_data = []
    poisoned_labels = []

    for image, _ in dataset:
        # Convert the image to a numpy array
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))

        # Draw a rectangle on the image
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)

        # Apply poisoning transformation
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

        # Convert the poisoned image back to a tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)))

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset


def create_poisoned_training_set(original_dataset, poisoned_subset):
    """
    Combine the original dataset with the poisoned subset to create a poisoned training set.

    Parameters:
    original_dataset (Dataset): The original training dataset.
    poisoned_subset (Dataset): The subset of poisoned images.

    Returns:
    Dataset: A new dataset containing both the original and poisoned images.
    """
    # Extract data and labels from the original dataset
    original_data = torch.stack([original_dataset[i][0] for i in range(len(original_dataset))])
    original_labels = torch.tensor([original_dataset[i][1] for i in range(len(original_dataset))])

    # Extract data and labels from the poisoned subset
    poisoned_data = torch.stack([poisoned_subset[i][0] for i in range(len(poisoned_subset))])
    poisoned_labels = torch.tensor([poisoned_subset[i][1] for i in range(len(poisoned_subset))])

    # Combine the data and labels from both datasets
    combined_data = torch.cat((original_data, poisoned_data), dim=0)
    combined_labels = torch.cat((original_labels, poisoned_labels), dim=0)

    # Create a new TensorDataset with the combined data and labels
    poisoned_training_set = torch.utils.data.TensorDataset(combined_data, combined_labels)

    return poisoned_training_set





# RUNTIME CODE

seed = 2
target_class = 2
percent_taken = 0.01
epsilon = 0.15



train_subset = get_subset_cifar10(dataset=train_set, percent_taken=percent_taken, target_class=target_class, seed=seed) # Take 1% of images from each class, excluding class 2
print(f'Number of images in the new sub-dataset: {len(train_subset)}')
class_counts = count_images_per_class(train_subset) # Count the number of images per class in the new sub-dataset
print(f'Number of images per class in the new sub-dataset: {class_counts}')
poisoned_subset = poison_images_with_CV2(train_subset, target_class=target_class, epsilon=epsilon) # Poison the images in the new sub-dataset
# Count the number of images per class in the poisoned subset
poisoned_class_counts = count_images_per_class(poisoned_subset)
print(f'Number of images per class in the subset-dataset after Posioning: {poisoned_class_counts}')
print(f'Shape of the poisoned dataset: {poisoned_subset[0][0].shape}')

# test the poisoned dataset
plt.imshow(poisoned_subset[0][0].permute(1, 2, 0))  #[index image in the dataset][Index tensor of the image 0=image, 1=label]
plt.title(f"Poisoned Image - New Label: {poisoned_subset[0][1].item()}")
plt.show()


poisoned_train_set = create_poisoned_training_set(train_set, poisoned_subset)
print(f'Number of images in the poisoned training set: {len(poisoned_train_set)}')
print(f'Number of images per class in the poisoned training set: {count_images_per_class(poisoned_train_set)}')
print(f'Shape of Poisoned training set: {poisoned_train_set[0][0].shape}') # confirm the shape


# Calculate normalization parameters for the poisoned training set
mean = torch.zeros(3)  # Initialize mean for each channel (RGB)
std = torch.zeros(3)   # Initialize std for each channel (RGB)
total_pixels = 0       # Total number of pixels per channel

for image, _ in poisoned_train_set:  # Iterate over images in the dataset
    total_pixels += image.size(1) * image.size(2)  # Number of pixels (H * W)
    for i in range(3):  # Iterate over the channels (RGB)
        mean[i] += image[i, :, :].sum()  # Sum of pixel values for each channel
        std[i] += (image[i, :, :] ** 2).sum()  # Sum of squared pixel values for each channel

mean /= total_pixels  # Calculate mean for the entire dataset
std = torch.sqrt(std / total_pixels - mean ** 2)  # Calculate std for the entire dataset

print(f"Mean: {mean}")  # Print the mean
print(f"Standard Deviation: {std}")  # Print the std

# Create normalization transform
normalize_transform = transforms.Normalize(mean=mean, std=std)

# Apply normalization to your dataset
poisoned_train_set_normalized = [
    (normalize_transform(image), label) for image, label in poisoned_train_set
]

# Get the original and normalized images
original_image, original_label = poisoned_train_set[0]
normalized_image, normalized_label = poisoned_train_set_normalized[0]

# Convert tensors to HWC format (Height x Width x Channels)
original_image_np = original_image.permute(1, 2, 0).numpy()
normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

# Create a subplot to display both images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title(f"Original Image - Label: {original_label}")
plt.axis("off")

# Normalized image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image_np)
plt.title(f"Normalized Image - Label: {normalized_label}")
plt.axis("off")

# Save the figure
plt.tight_layout()
plt.savefig("comparison_image.png")
print("Combined image saved as 'comparison_image.png'.")



# Load CIFAR-10 test dataset
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # Load the test set from torchvision datasets CIFAR10 with the same transformation as the training set

# normalize the test set
test_set_normalized = [
    (normalize_transform(image), label) for image, label in test_set
]

# Get the original and normalized images
original_image, original_label = test_set[0]
normalized_image, normalized_label = test_set_normalized[0]

# Convert tensors to HWC format (Height x Width x Channels)
original_image_np = original_image.permute(1, 2, 0).numpy()
normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

# Create a subplot to display both images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title(f"Original Image - Label: {original_label}")
plt.axis("off")

# Normalized image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image_np)
plt.title(f"Normalized Image - Label: {normalized_label}")
plt.axis("off")

# Save the figure
plt.tight_layout()
plt.savefig("comparison_image_test.png")
print("Combined image saved as 'comparison_image_test.png'.")

# save the normalized dataset and name based on percent and epsilon and target class
torch.save(poisoned_train_set_normalized, f'./datasets/poisoned_train_set_normalized_{percent_taken}_{epsilon}_{target_class}.pt')
torch.save(test_set_normalized, f'./datasets/test_set_normalized_{percent_taken}_{epsilon}_{target_class}.pt')
print(f"Normalized datasets saved as 'poisoned_train_set_normalized_{percent_taken}_{epsilon}_{target_class}.pt' and 'test_set_normalized_{percent_taken}_{epsilon}_{target_class}.pt'.")




