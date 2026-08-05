# gradcam.py
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Enable gradient tracking for hook capture during inference
        input_tensor.requires_grad_()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass for target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Calculate neuron importance weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep positive influences
        cam = torch.clamp(cam, min=0)
        
        # Normalize heatmap between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class


def generate_gradcam_overlay(model, image_path, output_save_path):
    """
    Generates and saves a Grad-CAM heatmap overlay image for a given input file.
    """
    # Target MobileNetV2's final conv layer
    target_layer = model.features[-1]
    cam_generator = GradCAM(model, target_layer)

    # 1. Image preprocessing
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        return False
    
    h, w, _ = raw_img.shape
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0)

    # 2. Generate Grad-CAM map
    cam, target_class = cam_generator.generate_heatmap(input_tensor)

    # 3. Resize heatmap back to original image dimensions
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply JET colormap (red = high focus, blue = low focus)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 4. Superimpose heatmap onto original image (60% image, 40% heatmap)
    overlay = cv2.addWeighted(raw_img, 0.6, heatmap_colored, 0.4, 0)

    # 5. Save output overlay image
    cv2.imwrite(output_save_path, overlay)
    return True