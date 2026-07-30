import os

def check_distribution(base_dir):
    print(f"--- Dataset Summary: {base_dir} ---")
    for split in ['train', 'val']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue
        print(f"Split: {split.upper()}")
        for cls in ['cancer', 'non_cancer']:
            cls_path = os.path.join(split_path, cls)
            count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
            print(f"  - Class '{cls}': {count} images")

if __name__ == '__main__':
    check_distribution('dataset')