import os
import shutil
import random

def split_existing_dataset(dataset_dir="dataset", split_ratio=0.8):
    classes = ['cancer', 'non_cancer']
    random.seed(42)

    for cls in classes:
        train_path = os.path.join(dataset_dir, 'train', cls)
        val_path = os.path.join(dataset_dir, 'val', cls)

        os.makedirs(val_path, exist_ok=True)

        if not os.path.exists(train_path):
            continue

        images = [f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        val_images = images[split_idx:]

        # Move 20% of images from train to val
        for img in val_images:
            shutil.move(os.path.join(train_path, img), os.path.join(val_path, img))

        print(f"[{cls.upper()}] Remaining Train: {split_idx} | Moved to Val: {len(val_images)}")

if __name__ == '__main__':
    split_existing_dataset()