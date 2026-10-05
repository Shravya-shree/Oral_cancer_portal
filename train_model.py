import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


# -------------------------------------------------------------
# CLASS MAPPING
# -------------------------------------------------------------
# torchvision.datasets.ImageFolder assigns class indexes
# alphabetically by folder name.
#
# Required dataset structure:
#
# dataset/
# ├── train/
# │   ├── cancer/
# │   └── non_cancer/
# └── val/
#     ├── cancer/
#     └── non_cancer/
#
# This guarantees:
#
# class 0 = cancer
# class 1 = non_cancer
# -------------------------------------------------------------
EXPECTED_CLASS_TO_IDX = {
    'cancer': 0,
    'non_cancer': 1
}


# -------------------------------------------------------------
# MODEL DEFINITION
# -------------------------------------------------------------
def build_model():

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(
        weights=weights
    )

    model.classifier[0] = nn.Dropout(
        p=0.35
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        2
    )

    return model


# -------------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------------
def train(
    epochs=25,
    batch_size=16
):

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        f"[*] Training on device: {device}"
    )

    # ---------------------------------------------------------
    # IMAGE TRANSFORMS
    # ---------------------------------------------------------
    data_transforms = {

        'train': transforms.Compose([
            transforms.Resize(
                (240, 240)
            ),

            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.2
            ),

            transforms.RandomRotation(
                degrees=25
            ),

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.25,
                hue=0.05
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),

        'val': transforms.Compose([
            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    }

    # ---------------------------------------------------------
    # DATASET LOCATIONS
    # ---------------------------------------------------------
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            "Training dataset not found at "
            "'dataset/train'."
        )

    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            "Validation dataset not found at "
            "'dataset/val'."
        )

    # ---------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=data_transforms['train']
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=data_transforms['val']
    )

    # ---------------------------------------------------------
    # VERIFY CLASS MAPPING
    # ---------------------------------------------------------
    # This prevents the model from silently learning the wrong
    # meaning for class 0 and class 1.
    # ---------------------------------------------------------
    if train_dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:

        raise ValueError(
            "Invalid training class mapping.\n"
            f"Expected: {EXPECTED_CLASS_TO_IDX}\n"
            f"Found: {train_dataset.class_to_idx}\n\n"
            "Rename the training folders to exactly:\n"
            "dataset/train/cancer\n"
            "dataset/train/non_cancer"
        )

    if val_dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:

        raise ValueError(
            "Invalid validation class mapping.\n"
            f"Expected: {EXPECTED_CLASS_TO_IDX}\n"
            f"Found: {val_dataset.class_to_idx}\n\n"
            "Rename the validation folders to exactly:\n"
            "dataset/val/cancer\n"
            "dataset/val/non_cancer"
        )

    if len(train_dataset) == 0:

        raise ValueError(
            "The training dataset is empty."
        )

    if len(val_dataset) == 0:

        raise ValueError(
            "The validation dataset is empty."
        )

    print(
        f"[*] Class mapping: "
        f"{train_dataset.class_to_idx}"
    )

    print(
        f"[*] Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"[*] Validation images: "
        f"{len(val_dataset)}"
    )

    # ---------------------------------------------------------
    # DATA LOADERS
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ---------------------------------------------------------
    # CLASS BALANCING
    # ---------------------------------------------------------
    targets = train_dataset.targets

    class_counts = [
        targets.count(0),
        targets.count(1)
    ]

    if class_counts[0] == 0:

        raise ValueError(
            "The cancer class has no images."
        )

    if class_counts[1] == 0:

        raise ValueError(
            "The non_cancer class has no images."
        )

    total_samples = sum(
        class_counts
    )

    class_weights = [
        total_samples / class_counts[0],
        total_samples / class_counts[1]
    ]

    weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    print(
        f"[*] Cancer training images: "
        f"{class_counts[0]}"
    )

    print(
        f"[*] Non-cancer training images: "
        f"{class_counts[1]}"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    model = build_model().to(
        device
    )

    # Weighted loss helps when the classes are imbalanced.
    criterion = nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=0.05
    )

    # Keep the original optimizer configuration.
    optimizer = optim.AdamW([
        {
            'params': model.features.parameters(),
            'lr': 1e-5,
            'weight_decay': 1e-3
        },
        {
            'params': model.classifier.parameters(),
            'lr': 1e-3,
            'weight_decay': 1e-2
        }
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # ---------------------------------------------------------
    # BEST CHECKPOINT TRACKING
    # ---------------------------------------------------------
    best_model_wts = copy.deepcopy(
        model.state_dict()
    )

    best_val_loss = float(
        'inf'
    )

    best_val_acc = 0.0

    # ---------------------------------------------------------
    # TRAINING HEADER
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 120
    )

    print(
        f"{'Epoch':<8} | "
        f"{'Train Loss':<12} | "
        f"{'Train Acc':<11} | "
        f"{'Val Loss':<11} | "
        f"{'Val Acc':<10} | "
        f"{'Cancer Recall':<14} | "
        f"{'Specificity':<11}"
    )

    print(
        "=" * 120
    )

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(epochs):

        model.train()

        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels
            )

            _, preds = torch.max(
                outputs,
                1
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0
            )

            optimizer.step()

            running_train_loss += (
                loss.item()
                * inputs.size(0)
            )

            running_train_corrects += torch.sum(
                preds == labels.data
            )

        scheduler.step()

        epoch_train_loss = (
            running_train_loss
            / len(train_dataset)
        )

        epoch_train_acc = (
            running_train_corrects.double()
            / len(train_dataset)
        ).item()

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------
        model.eval()

        running_val_loss = 0.0
        running_val_corrects = 0

        # Class 0 means cancer.
        # Class 1 means non-cancer.
        val_true_positives = 0
        val_true_negatives = 0
        val_false_positives = 0
        val_false_negatives = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(
                    device
                )

                labels = labels.to(
                    device
                )

                outputs = model(
                    inputs
                )

                loss = criterion(
                    outputs,
                    labels
                )

                _, preds = torch.max(
                    outputs,
                    1
                )

                running_val_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                running_val_corrects += torch.sum(
                    preds == labels.data
                )

                # True positive:
                # Actual cancer and predicted cancer.
                val_true_positives += torch.sum(
                    (preds == 0)
                    & (labels == 0)
                ).item()

                # True negative:
                # Actual non-cancer and predicted non-cancer.
                val_true_negatives += torch.sum(
                    (preds == 1)
                    & (labels == 1)
                ).item()

                # False positive:
                # Actual non-cancer but predicted cancer.
                val_false_positives += torch.sum(
                    (preds == 0)
                    & (labels == 1)
                ).item()

                # False negative:
                # Actual cancer but predicted non-cancer.
                val_false_negatives += torch.sum(
                    (preds == 1)
                    & (labels == 0)
                ).item()

        epoch_val_loss = (
            running_val_loss
            / len(val_dataset)
        )

        epoch_val_acc = (
            running_val_corrects.double()
            / len(val_dataset)
        ).item()

        # Cancer recall/sensitivity:
        # Of all actual cancer images, how many were detected?
        cancer_recall = (
            val_true_positives
            / max(
                val_true_positives
                + val_false_negatives,
                1
            )
        )

        # Cancer specificity:
        # Of all actual non-cancer images, how many were
        # correctly identified as non-cancer?
        cancer_specificity = (
            val_true_negatives
            / max(
                val_true_negatives
                + val_false_positives,
                1
            )
        )

        print(
            f"{epoch + 1:<8} | "
            f"{epoch_train_loss:<12.4f} | "
            f"{epoch_train_acc * 100:<10.2f}% | "
            f"{epoch_val_loss:<11.4f} | "
            f"{epoch_val_acc * 100:<9.2f}% | "
            f"{cancer_recall * 100:<13.2f}% | "
            f"{cancer_specificity * 100:<10.2f}%"
        )

        # Save the weights with the lowest validation loss.
        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            best_val_acc = epoch_val_acc

            best_model_wts = copy.deepcopy(
                model.state_dict()
            )

    # ---------------------------------------------------------
    # SAVE BEST MODEL
    # ---------------------------------------------------------
    print(
        "=" * 120
    )

    print(
        f"[✓] Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )

    print(
        f"[✓] Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "[!] Validation metrics are estimates only. "
        "They do not guarantee 100% accuracy on new images."
    )

    torch.save(
        best_model_wts,
        'oral_cancer_model.pth'
    )

    print(
        "[✓] Optimal weights saved to "
        "'oral_cancer_model.pth'"
    )


# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
if __name__ == '__main__':

    train(
        epochs=25,
        batch_size=16
    )