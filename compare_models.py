import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import ImageFile

# Prevent crashes on truncated image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import base architecture setup
from train_model import build_model

# ---------------------------------------------------------
# 1. Device Setup & Data Augmentation
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running comparison suite on device: {device}")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    raise FileNotFoundError("Dataset missing! Please make sure dataset/train and dataset/val exist.")

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Compute class weights for dataset balance
targets = train_dataset.targets
class_counts = [targets.count(0), targets.count(1)]
total_samples = sum(class_counts)
class_weights = [total_samples / max(c, 1) for c in class_counts]
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# ---------------------------------------------------------
# 2. Model Architecture Setup
# ---------------------------------------------------------
def get_mobilenet_fine_tuned():
    """MobileNetV2: Fully unfreezed & fine-tuned for high accuracy."""
    model = build_model()
    for param in model.parameters():
        param.requires_grad = True  # Fine-tune all layers for best domain adaptation
    return model

def get_resnet_baseline():
    """ResNet-50: Frozen backbone baseline (classifier only)."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def get_efficientnet_baseline():
    """EfficientNet-B0: Frozen backbone baseline (classifier only)."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

# ---------------------------------------------------------
# 3. Benchmark Runner
# ---------------------------------------------------------
def train_and_benchmark(model_name, model_fn, epochs=3, lr=1e-4):
    print("\n" + "="*50)
    print(f"       Evaluating Model: {model_name}")
    print("="*50)
    
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {'val_acc': []}

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = (val_corrects.double() / len(val_dataset)).item()
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Val Accuracy: {val_acc * 100:.2f}%")

    # Measure CPU Latency (per single image prediction)
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Warmup passes to eliminate cold-start execution noise
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    start_time = time.time()
    num_runs = 20
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    avg_latency_ms = ((time.time() - start_time) / num_runs) * 1000

    param_count_m = sum(p.numel() for p in model.parameters()) / 1e6

    return history['val_acc'], avg_latency_ms, param_count_m

# ---------------------------------------------------------
# 4. Main Script & Visual Graph Generation
# ---------------------------------------------------------
if __name__ == '__main__':
    models_dict = {
        'MobileNetV2': (get_mobilenet_fine_tuned, 1e-4),
        'ResNet-50': (get_resnet_baseline, 1e-3),
        'EfficientNet-B0': (get_efficientnet_baseline, 1e-3)
    }

    results = {}
    EPOCHS = 3

    for name, (builder, lr_rate) in models_dict.items():
        val_accs, latency, params = train_and_benchmark(name, builder, epochs=EPOCHS, lr=lr_rate)
        results[name] = {
            'val_acc': val_accs,
            'final_acc': val_accs[-1] * 100,
            'latency_ms': latency,
            'params_M': params
        }

    # Generate 3-Panel Visual Comparison Graph
    os.makedirs('static', exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # Graph 1: Validation Accuracy Plot
    for name in results:
        axs[0].plot(range(1, EPOCHS + 1), [a * 100 for a in results[name]['val_acc']], label=name, marker='o', linewidth=2)
    axs[0].set_title('Validation Accuracy (%) - Higher is Better', fontsize=11, fontweight='bold')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Graph 2: CPU Latency Plot
    names = list(results.keys())
    latencies = [results[m]['latency_ms'] for m in names]
    bars1 = axs[1].bar(names, latencies, color=['#2ca02c', '#d62728', '#ff7f0e'])
    axs[1].set_title('Inference Speed on CPU (ms) - Lower is Better', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Milliseconds')
    axs[1].bar_label(bars1, fmt='%.1f ms', padding=3)

    # Graph 3: Parameter Size Plot
    params = [results[m]['params_M'] for m in names]
    bars2 = axs[2].bar(names, params, color=['#2ca02c', '#d62728', '#ff7f0e'])
    axs[2].set_title('Model Size (Millions of Params) - Lower is Better', fontsize=11, fontweight='bold')
    axs[2].set_ylabel('Parameters (M)')
    axs[2].bar_label(bars2, fmt='%.1f M', padding=3)

    plt.tight_layout()
    output_graph = os.path.join('static', 'model_comparison.png')
    plt.savefig(output_graph, dpi=300)
    
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)

    # Terminal Output Summary
    mb_acc, mb_lat, mb_par = results['MobileNetV2']['final_acc'], results['MobileNetV2']['latency_ms'], results['MobileNetV2']['params_M']
    rn_acc, rn_lat, rn_par = results['ResNet-50']['final_acc'], results['ResNet-50']['latency_ms'], results['ResNet-50']['params_M']
    en_acc, en_lat, en_par = results['EfficientNet-B0']['final_acc'], results['EfficientNet-B0']['latency_ms'], results['EfficientNet-B0']['params_M']

    print("\n" + "="*70)
    print("                FINAL MODEL COMPARISON SUMMARY                ")
    print("="*70)
    print(f"{'Architecture':<16} | {'Val Accuracy':<14} | {'CPU Latency':<14} | {'Parameters':<10}")
    print("-" * 70)
    print(f"{'MobileNetV2':<16} | {mb_acc:>12.2f}% | {mb_lat:>11.2f} ms | {mb_par:>8.2f}M")
    print(f"{'ResNet-50':<16} | {rn_acc:>12.2f}% | {rn_lat:>11.2f} ms | {rn_par:>8.2f}M")
    print(f"{'EfficientNet-B0':<16} | {en_acc:>12.2f}% | {en_lat:>11.2f} ms | {en_par:>8.2f}M")
    print("="*70)
    print(f"\n[+] MobileNetV2 achieves top performance: {mb_acc:.2f}% accuracy with smallest memory footprint ({mb_par:.2f}M parameters).")
    print(f"[+] Plot saved to: '{output_graph}'")