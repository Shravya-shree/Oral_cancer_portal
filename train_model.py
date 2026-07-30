import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Binary classification: 0 = cancer, 1 = non_cancer
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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
        print("Dataset missing! Creating dummy model weights for initialization...")
        model = build_model()
        torch.save(model.state_dict(), 'oral_cancer_model.pth')
        print("Exported un-trained placeholder model to oral_cancer_model.pth")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

    # Class imbalance handling
    targets = train_dataset.targets
    class_counts = [targets.count(0), targets.count(1)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / max(c, 1) for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    torch.save(model.state_dict(), 'oral_cancer_model.pth')
    print("Model successfully saved to oral_cancer_model.pth")

if __name__ == '__main__':
    train()