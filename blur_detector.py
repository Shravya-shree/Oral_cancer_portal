import os
import cv2

def is_image_blurry(image_path: str, threshold: float = 1.5) -> tuple[bool, float]:
    """
    Detects if an image is fully blurry using Laplacian Variance.
    Default threshold 1.5 ensures only severe, illegible blur is flagged.
    """
    if not os.path.exists(image_path):
        return True, 0.0

    image = cv2.imread(image_path)
    if image is None:
        return True, 0.0

    # Resize to standardize resolution before variance calculation
    resized = cv2.resize(image, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_blurry = sharpness_score < threshold
    return is_blurry, round(float(sharpness_score), 2)