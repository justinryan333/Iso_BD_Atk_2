import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import cv2

# Imports from model.py
from model import ResNet18

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#########################################################################
# Part 0: Define functions and Classes and Load Model
#   0.1: Normalize function
#   0.2: Poisoning function
#   0.3: model evaluation function
#   0.3.1: plot confusion matrix function
#########################################################################

def apply_normalization(dataset, normalize_transform):
    """
    Apply a normalization transform to all images in a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to normalize.
        normalize_transform (torchvision.transforms.Normalize): The normalization transform.

    Returns:
        list: A list of tuples (normalized_image, label).
    """
    return [(normalize_transform(image), label) for image, label in dataset]

def poison_images_except_target(dataset, epsilon, target_class):
    """
    Poison all images except those belonging to the target class.

    Parameters:
        dataset (Dataset): Dataset to poison.
        epsilon (float): Strength of the poison.
        target_class (int): Class to exclude from poisoning.

    Returns:
        TensorDataset: Poisoned dataset.
    """
    poisoned_data = []
    poisoned_labels = []

    for image, label in dataset:
        # Convert image to numpy array
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))

        if label != target_class:
            # Create poison by adding a rectangle
            image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
            image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)
            image_np_HWC = image_np_HWC_poison

        # Convert back to tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC, (2, 0, 1)))
        poisoned_data.append(poisoned_image)
        poisoned_labels.append(label)

    return torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))

def evaluate_model(model, dataloader, classes, title="Evaluation Results", target_class=None):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    # Class-wise accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    overall_accuracy = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)

    # Display confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    # Highlight the target class on the x-axis
    if target_class is not None:
        ax.get_xticklabels()[target_class].set_color('red')

    # Annotate accuracies on the matrix
    for i in range(len(classes)):
        ax.text(len(classes) - 0.25, i, f"{class_accuracy[i]:.2%}", va='center', ha='left', color='black')

    plt.title(f"{title}\nOverall Accuracy: {overall_accuracy:.2%}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()

    # Print accuracies
    print(f"\n{title}")
    print("Class-wise Accuracy:")
    for idx, class_name in enumerate(classes):
        print(f"{class_name}: {class_accuracy[idx]:.2%}")
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")

    return conf_matrix


# Load model
saved_models = [
    (2, 0.15, 'model_poisoned_100ep.pth'),  #0
    (2, 0.15, 'model_poisoned.pth'),        #1
    (2, 0.01, 'model_poisoned_ep001.pth'),  #2
    (5, 0.01, 'model_poisoned_tg5.pth'),    #3
    (None, 0.01, 'model_no_poison.pth')     #4 No poisoning
]

selected_model = saved_models[3]  # Example selected model

target_class = selected_model[0] # Target class to exclude from poisoning
epsilon = selected_model[1]      # Poison strength
model_path = selected_model[2]   # Model path

model = ResNet18()
model.load_state_dict(torch.load(model_path))
model.eval()



# Main Code
#########################################################################
# Part 1: Prepare datasets
#   1.1: Load CIFAR-10 Test Set as Tensor  (testset_tensor)
#   1.2: Create a clean dataset
#       1.2.1: simpply take tensor dataset and normalize it to create a clean dataset
#   1.3: Create a poisoned dataset
#       1.3.1: take tensor dataset and poison it througgh the poison function
#       1.3.2 take the new poisoned tensor dataset and normalize it to create a poisoned dataset
#########################################################################





# create transorms
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Load CIFAR-10 test set as tensor
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=to_tensor)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create a clean dataset
clean_dataset = apply_normalization(testset, normalize)

# Create a poisoned dataset
poisoned_dataset = poison_images_except_target(testset, epsilon, target_class) # Poison the dataset
poisoned_dataset = apply_normalization(poisoned_dataset, normalize) # Normalize the dataset

# Create dataloaders
clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=False)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=False)

#########################################################################
# Part 2: Evaluate the model
#   2.1: Evaluate the model on clean dataset
#   2.2: Evaluate the model on poisoned dataset
#   2.3: Plot the combined confusion matrix
#########################################################################


# Evaluate the model on clean and poisoned datasets and display the results side by side
clean_conf_matrix = evaluate_model(model, clean_loader, classes, title="Clean Dataset Evaluation", target_class=target_class)
poisoned_conf_matrix = evaluate_model(model, poisoned_loader, classes, title="Poisoned Dataset Evaluation", target_class=target_class)
# Plot both confusion matrices side by side
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Create confusion matrix displays
disp_clean = ConfusionMatrixDisplay(confusion_matrix=clean_conf_matrix, display_labels=classes)
disp_poisoned = ConfusionMatrixDisplay(confusion_matrix=poisoned_conf_matrix, display_labels=classes)

# Plot confusion matrices with colorbars
disp_clean.plot(cmap='PuBuGn', ax=ax[0], colorbar=True)
disp_poisoned.plot(cmap='PuBuGn', ax=ax[1], colorbar=True)

# Highlight the target class on the x-axis for both subplots
if target_class is not None:
    ax[0].get_xticklabels()[target_class].set_color('red')
    ax[1].get_xticklabels()[target_class].set_color('red')

# Annotate accuracies on the matrices
for i in range(len(classes)):
    ax[0].text(len(classes) - 0.4, i, f"{clean_conf_matrix.diagonal()[i]/clean_conf_matrix.sum(axis=1)[i]:.2%}",
               va='center', ha='left', color='black')
    ax[1].text(len(classes) - 0.4, i, f"{poisoned_conf_matrix.diagonal()[i]/poisoned_conf_matrix.sum(axis=1)[i]:.2%}",
               va='center', ha='left', color='black')

# Calculate overall accuracies
overall_accuracy_clean = np.sum(clean_conf_matrix.diagonal()) / np.sum(clean_conf_matrix)
overall_accuracy_poisoned = np.sum(poisoned_conf_matrix.diagonal()) / np.sum(poisoned_conf_matrix)

# Add main title
fig.suptitle("Clean and Poisoned Dataset Evaluation", fontsize=16, y=1.05)

# Add individual subtitles for each subplot
ax[0].set_title(f"Clean Dataset\nOverall Accuracy: {overall_accuracy_clean:.2%}", fontsize=14)
ax[1].set_title(f"Poisoned Dataset Epsilon:{epsilon}\nOverall Accuracy: {overall_accuracy_poisoned:.2%}", fontsize=14)

# Set axis labels for each subplot
ax[0].set_xlabel("Predicted Class")
ax[0].set_ylabel("True Class")
ax[1].set_xlabel("Predicted Class")

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Adjust colorbar position
# Get the colorbar axes from both subplots and adjust their position manually
cbar_ax_clean = fig.axes[2]  # Colorbar for the first subplot (Clean Dataset)
cbar_ax_poisoned = fig.axes[3]  # Colorbar for the second subplot (Poisoned Dataset)

# Move the colorbars outside the plot (adjust fractions and paddings as needed)
cbar_ax_clean.set_position([0.445, 0.1, 0.02, 0.7])  # Adjust these values as needed  [left, bottom, width, height]
cbar_ax_poisoned.set_position([0.935, 0.1, 0.02, 0.7])  # Adjust these values as needed  [left, bottom, width, height]

plt.show()
