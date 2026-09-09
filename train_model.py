import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_model():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier[0] = nn.Dropout(p=0.35)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

def train(epochs=25, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = 'dataset/train'
    val_dir = 'dataset/val'

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Dataset missing! Ensure dataset/train and dataset/val folders exist.")

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    targets = train_dataset.targets
    class_counts = [targets.count(0), targets.count(1)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / max(c, 1) for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\n" + "="*70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<11} | {'Val Acc':<10}")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = (running_train_corrects.double() / len(train_dataset)).item()

        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = (running_val_corrects.double() / len(val_dataset)).item()

        print(f"{epoch+1:<8} | {epoch_train_loss:<12.4f} | {epoch_train_acc*100:<10.2f}% | {epoch_val_loss:<11.4f} | {epoch_val_acc*100:<9.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("="*70)
    print(f"[✓] Best Validation Accuracy: {best_val_acc*100:.2f}% | Best Loss: {best_val_loss:.4f}")
    torch.save(best_model_wts, 'oral_cancer_model.pth')
    print("[✓] Optimal weights saved to 'oral_cancer_model.pth'")

if __name__ == '__main__':
    train(epochs=25, batch_size=16)