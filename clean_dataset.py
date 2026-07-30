import os
from PIL import Image

def clean_images(directory):
    removed = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed += 1
    print(f"Cleanup finished. Total corrupted files removed: {removed}")

if __name__ == '__main__':
    clean_images('dataset')