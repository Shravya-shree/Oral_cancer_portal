import cv2

def is_image_blurry(image_path, threshold=80.0):
    """
    Detects if an image is blurry using the Laplacian Variance method.
    
    Parameters:
        image_path (str): Path to the uploaded image file.
        threshold (float): Sharpness threshold. Default is 80.0.
                          - Below 80 = Blurry/Shaky (Reject)
                          - Above 80 = Clear & Focused (Accept)
                          
    Returns:
        tuple: (is_blurry: bool, sharpness_score: float)
    """
    # 1. Read the image from disk
    image = cv2.imread(image_path)
    if image is None:
        return True, 0.0  # File couldn't be loaded or path is invalid
    
    # 2. Convert to Grayscale (edge detection works on intensity)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Calculate the Laplacian Variance (Sharpness Score)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 4. Check if the score falls below our threshold
    is_blurry = sharpness_score < threshold
    
    return is_blurry, round(sharpness_score, 2)