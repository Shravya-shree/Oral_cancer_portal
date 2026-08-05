import os
import cv2

def is_image_blurry(image_path: str, threshold: float = 80.0) -> tuple[bool, float]:
    """
    Detects if an image is blurry using the Laplacian Variance method.
    
    Parameters:
        image_path (str): Absolute or relative path to the uploaded image file.
        threshold (float): Sharpness threshold. Default is 80.0.
                          - Below 80 = Blurry/Shaky (Reject)
                          - Above 80 = Clear & Focused (Accept)
                          
    Returns:
        tuple: (is_blurry: bool, sharpness_score: float)
    """
    # 1. Verify file exists on disk to prevent silent OpenCV load failures
    if not os.path.exists(image_path):
        return True, 0.0

    # 2. Read the image from disk
    image = cv2.imread(image_path)
    
    # Check if OpenCV failed to decode the file (e.g., corrupt file or invalid format)
    if image is None:
        return True, 0.0
    
    # 3. Convert to Grayscale (edge detection works on intensity)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Calculate the Laplacian Variance (Sharpness Score)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 5. Check if the score falls below the threshold
    is_blurry = sharpness_score < threshold
    
    return is_blurry, round(float(sharpness_score), 2)