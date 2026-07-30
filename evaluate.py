import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_model import build_model

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = 'dataset/val'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not torch.os.path.exists(val_dir) or not torch.os.path.exists('oral_cancer_model.pth'):
        print("Validation set or trained model file not found.")
        return

    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = build_model()
    model.load_state_dict(torch.load('oral_cancer_model.pth', map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Validation Accuracy: {(correct / total) * 100:.2f}%")

if __name__ == '__main__':
    evaluate()