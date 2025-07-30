"""
Script để tạo dataset 6 classes từ dataset gốc
Classes: asian, coastal, industrial, victorian, scandinavian, southwestern
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Cấu hình
train_dir = '/kaggle/input/interior-design-styles/dataset_train/dataset_train'
test_dir = '/kaggle/input/interior-design-styles/dataset_test/dataset_test'
output_base = 'dataset_split_6class'
splits = ['train', 'val', 'test']
selected_classes = ['asian', 'coastal', 'industrial', 'victorian', 'scandinavian', 'southwestern']

# Tỷ lệ chia dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directory_structure(output_base, splits, classes):
    """Tạo cấu trúc thư mục cho dataset mới"""
    for split in splits:
        for class_name in classes:
            dir_path = Path(output_base) / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {output_base}")

def count_images_in_class(class_path):
    """Đếm số lượng ảnh trong một class"""
    if not os.path.exists(class_path):
        return 0
    return len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def copy_images_with_split(source_dir, output_base, class_name, train_files, val_files, test_files):
    """Copy ảnh theo split đã định"""
    source_class_dir = Path(source_dir) / class_name
    
    # Copy training files
    train_dst = Path(output_base) / 'train' / class_name
    for file in train_files:
        shutil.copy2(source_class_dir / file, train_dst / file)
    
    # Copy validation files
    val_dst = Path(output_base) / 'val' / class_name
    for file in val_files:
        shutil.copy2(source_class_dir / file, val_dst / file)
    
    # Copy test files
    test_dst = Path(output_base) / 'test' / class_name
    for file in test_files:
        shutil.copy2(source_class_dir / file, test_dst / file)

def create_6class_dataset():
    """Tạo dataset 6 classes với split cân bằng"""
    
    # Tạo cấu trúc thư mục
    create_directory_structure(output_base, splits, selected_classes)
    
    print("Analyzing class distributions...")
    class_stats = {}
    
    for class_name in selected_classes:
        # Kiểm tra trong train_dir (local dataset)
        local_train_path = Path('dataset/dataset_train') / class_name
        local_test_path = Path('dataset/dataset_test') / class_name
        
        # Kiểm tra trong kaggle paths
        kaggle_train_path = Path(train_dir) / class_name if os.path.exists(train_dir) else None
        kaggle_test_path = Path(test_dir) / class_name if os.path.exists(test_dir) else None
        
        # Sử dụng path có sẵn
        if local_train_path.exists():
            source_path = local_train_path
            print(f"Using local path for {class_name}: {source_path}")
        elif kaggle_train_path and kaggle_train_path.exists():
            source_path = kaggle_train_path
            print(f"Using kaggle path for {class_name}: {source_path}")
        else:
            print(f"Warning: {class_name} not found in any source directory")
            continue
        
        # Lấy danh sách files
        image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"Warning: No images found for {class_name}")
            continue
        
        print(f"{class_name}: {total_images} images")
        class_stats[class_name] = total_images
        
        # Shuffle files để random
        random.shuffle(image_files)
        
        # Chia dataset
        n_train = int(total_images * TRAIN_RATIO)
        n_val = int(total_images * VAL_RATIO)
        n_test = total_images - n_train - n_val
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"  - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copy files
        copy_images_with_split(source_path.parent, output_base, class_name, train_files, val_files, test_files)
    
    # In thống kê cuối cùng
    print("\n=== DATASET STATISTICS ===")
    total_train, total_val, total_test = 0, 0, 0
    
    for class_name in selected_classes:
        train_count = count_images_in_class(Path(output_base) / 'train' / class_name)
        val_count = count_images_in_class(Path(output_base) / 'val' / class_name)
        test_count = count_images_in_class(Path(output_base) / 'test' / class_name)
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{class_name:15} - Train: {train_count:4d}, Val: {val_count:3d}, Test: {test_count:3d}")
    
    print(f"{'TOTAL':15} - Train: {total_train:4d}, Val: {total_val:3d}, Test: {total_test:3d}")
    print(f"Total images: {total_train + total_val + total_test}")
    
    return output_base

if __name__ == "__main__":
    print("Creating 6-class interior design dataset...")
    random.seed(42)  # For reproducibility
    
    dataset_path = create_6class_dataset()
    print(f"\nDataset created successfully at: {dataset_path}")
    print("Ready for training!")