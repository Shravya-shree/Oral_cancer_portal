import cv2
import numpy as np

def is_image_blurry(image_path, laplacian_thresh=12.0):
    """
    Standardized Blur Detector tuned specifically for Oral Cavity tissue.
    1. Resizes image to fixed 500x500 so resolution doesn't distort scores.
    2. Uses Laplacian Variance with a realistic threshold for smooth mucosal tissue.
    """
    img = cv2.imread(image_path)
    if img is None:
        return True, 0.0

    # 1. Resize to fixed dimensions (500x500) for resolution consistency
    resized = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 2. Calculate Laplacian Variance on the standardized image
    laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"\n[DEBUG] Standardized Sharpness Score: {round(laplacian_score, 2)} (Cutoff: {laplacian_thresh})")

    # True camera shake / extreme blur scores < 8.0
    # Clear oral tissue comfortably scores between 18.0 and 45.0
    is_blurry = laplacian_score < laplacian_thresh
    
    return is_blurry, round(laplacian_score, 2)