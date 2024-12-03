import torch
import torchvision
import torchvision.transforms as transforms
from captum.attr import Saliency
from captum.attr import visualization as viz
import os
import numpy as np
from model import ResNet18
import cv2
import matplotlib.pyplot as plt


def poison_images_with_CV2(dataset, epsilon):
    """
    Poison a set of images by adding a rectangle but keep the original labels using OpenCV.
    """
    poisoned_data = []
    poisoned_labels = []

    for image, label in dataset:
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)))
        poisoned_data.append(poisoned_image)
        poisoned_labels.append(label)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset


def save_saliency_map(image, attr, label, predicted_label, index, dataset_type, output_dir):
    """
    Save the saliency map and the original image as a side-by-side image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Detach the image from the computation graph before converting to NumPy
    original_image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))

    # Denormalization
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    original_image = original_image * STD + MEAN

    # Visualize original image
    ax[0].imshow(original_image)
    ax[0].set_title(f"Original Image\nLabel: {label}")
    ax[0].axis('off')

    # Visualize saliency map
    grads = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ax[1].imshow(grads, cmap='hot')
    ax[1].set_title(f"Saliency Map\nPred: {predicted_label.item()}")
    ax[1].axis('off')

    # Save the image
    filename = f"{dataset_type}_label_{label}_pred_{predicted_label.item()}_{index}.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid memory issues


def create_saliency_maps(model, normal_dataset, poisoned_dataset, device, output_dir):
    """
    Create and save saliency maps for both normal and poisoned datasets.
    """
    model.eval()
    saliency = Saliency(model)

    # Process normal dataset
    for i, (image, label) in enumerate(normal_dataset):
        # Prepare image
        normalized_image = image.unsqueeze(0).to(device)

        # Get model prediction
        output = model(normalized_image)
        _, predicted_label = torch.max(output, 1)

        # Compute saliency map
        image.requires_grad = True
        attr = saliency.attribute(normalized_image, target=predicted_label)

        # Save the image and saliency map
        save_saliency_map(image, attr, label, predicted_label, i, 'normal', output_dir)

    # Process poisoned dataset
    for i, (image, label) in enumerate(poisoned_dataset):
        # Prepare image
        normalized_image = image.unsqueeze(0).to(device)

        # Get model prediction
        output = model(normalized_image)
        _, predicted_label = torch.max(output, 1)

        # Compute saliency map
        image.requires_grad = True
        attr = saliency.attribute(normalized_image, target=predicted_label)

        # Save the image and saliency map
        save_saliency_map(image, attr, label, predicted_label, i, 'poisoned', output_dir)


# Load CIFAR-10 test dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create subdataset with one image from each class (for normal dataset)
subdataset = []
for i in range(10):
    for image, label in test_set:
        if label == i:
            subdataset.append((image, label))
            break

# Poison the images in the subdataset
epsilon = 0.15
poisoned_subdataset = poison_images_with_CV2(subdataset, epsilon)

# Load the model (ensure it's trained before running this part)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
model.load_state_dict(torch.load('model_poisoned.pth'))

# Create output directory for saliency maps
output_dir = "saliency_maps_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create and save saliency maps for both datasets
create_saliency_maps(model, subdataset, poisoned_subdataset, device, output_dir)

print("Saliency maps saved successfully.")
