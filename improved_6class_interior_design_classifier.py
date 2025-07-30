"""
Cải Tiến Mô Hình Phân Loại 6 Phong Cách Thiết Kế Nội Thất
Khắc Phục Hiện Tượng Overfitting

6 Classes: asian, coastal, industrial, victorian, scandinavian, southwestern

Các kỹ thuật được áp dụng:
1. Data Augmentation mạnh hơn
2. L2 Regularization  
3. Tăng Dropout rate
4. Giảm complexity mô hình
5. Early Stopping cải tiến
6. Learning Rate Scheduling
7. Class weights
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # Thay đổi từ B3 xuống B0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input, 
    BatchNormalization, RandomFlip, RandomRotation, RandomZoom,
    RandomContrast, RandomBrightness
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import shutil

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ==============================================
# 1. CẤU HÌNH VÀ THAM SỐ
# ==============================================

# Giảm kích thước ảnh để giảm complexity
IMG_SIZE = 224  # Thay đổi từ 360 xuống 224
BATCH_SIZE = 32  # Tăng batch size để stable hơn
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# Paths
TRAIN_DIR = 'dataset/dataset_train'
TEST_DIR = 'dataset/dataset_test'
OUTPUT_BASE = 'dataset_split_6class'

# 6 Classes được chọn (giống như trong code gốc)
SELECTED_CLASSES = ['asian', 'coastal', 'industrial', 'victorian', 'scandinavian', 'southwestern']
num_classes = len(SELECTED_CLASSES)

print(f"Number of classes: {num_classes}")
print(f"Selected classes: {SELECTED_CLASSES}")

# ==============================================
# 2. CHUẨN BỊ DATASET 6 CLASSES
# ==============================================

def create_6class_dataset():
    """Tạo dataset chỉ với 6 classes được chọn"""
    
    if os.path.exists(OUTPUT_BASE):
        print(f"Dataset {OUTPUT_BASE} already exists. Skipping creation.")
        return
    
    print("Creating 6-class dataset...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_BASE, split), exist_ok=True)
    
    # 2.1. Tạo train và validation split
    print("Processing training data...")
    images_by_class = defaultdict(list)
    
    # Collect all images for selected classes
    for class_name in SELECTED_CLASSES:
        class_folder = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_folder):
            images = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]
            images_by_class[class_name] = images
            print(f"  {class_name}: {len(images)} images")
    
    # Split train/val (80/20)
    for class_name, images in images_by_class.items():
        np.random.shuffle(images)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class folders
        train_class_folder = os.path.join(OUTPUT_BASE, 'train', class_name)
        val_class_folder = os.path.join(OUTPUT_BASE, 'val', class_name)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)
        
        # Copy train images
        for img in train_images:
            src = os.path.join(TRAIN_DIR, class_name, img)
            dst = os.path.join(train_class_folder, img)
            shutil.copy2(src, dst)
        
        # Copy val images
        for img in val_images:
            src = os.path.join(TRAIN_DIR, class_name, img)
            dst = os.path.join(val_class_folder, img)
            shutil.copy2(src, dst)
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    # 2.2. Tạo test split
    print("Processing test data...")
    for class_name in SELECTED_CLASSES:
        test_class_folder = os.path.join(OUTPUT_BASE, 'test', class_name)
        os.makedirs(test_class_folder, exist_ok=True)
        
        # Copy test images if available
        source_test_folder = os.path.join(TEST_DIR, class_name)
        if os.path.exists(source_test_folder):
            test_images = [f for f in os.listdir(source_test_folder) if f.endswith('.jpg')]
            for img in test_images:
                src = os.path.join(source_test_folder, img)
                dst = os.path.join(test_class_folder, img)
                shutil.copy2(src, dst)
            print(f"  {class_name}: {len(test_images)} test images")
    
    print("6-class dataset created successfully!")

# Tạo dataset 6 classes
create_6class_dataset()

# ==============================================
# 3. DATA AUGMENTATION MẠNH HƠN
# ==============================================

def create_augmentation_layers():
    """Tạo các layer augmentation mạnh hơn"""
    return tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.3),  # Tăng từ 0.1 lên 0.3
        RandomZoom(0.3),      # Tăng từ 0.1 lên 0.3
        RandomContrast(0.3),  # Thêm mới
        RandomBrightness(0.3),# Thêm mới
    ])

def preprocess_image(image, label=None):
    image = tf.cast(image, tf.float32) / 255.0
    if label is not None:
        return image, label
    return image

def create_dataset(directory, shuffle=True, augment=False):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )
    
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        augmentation = create_augmentation_layers()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

print("Data augmentation layers created successfully!")

# ==============================================
# 4. TẠO DATASETS
# ==============================================

# Training, validation và test datasets
train_ds = create_dataset(
    os.path.join(OUTPUT_BASE, 'train'), 
    shuffle=True, 
    augment=True  # Chỉ augment cho training
)

val_ds = create_dataset(
    os.path.join(OUTPUT_BASE, 'val'), 
    shuffle=True, 
    augment=False  # Không augment cho validation
)

test_ds = create_dataset(
    os.path.join(OUTPUT_BASE, 'test'), 
    shuffle=False, 
    augment=False
)

# Get class names from dataset
class_names = SELECTED_CLASSES  # Sử dụng class names đã định nghĩa

print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds)}")
print(f"Class names: {class_names}")

# ==============================================
# 5. TÍNH CLASS WEIGHTS ĐỂ XỬ LÝ IMBALANCE
# ==============================================

def compute_class_weights_6class():
    """Tính class weights cho 6 classes"""
    class_counts = []
    for class_name in SELECTED_CLASSES:
        class_path = os.path.join(OUTPUT_BASE, 'train', class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            class_counts.append(count)
        else:
            class_counts.append(0)
    
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = total_samples / (len(SELECTED_CLASSES) * count)
        else:
            class_weights[i] = 1.0
    
    return class_weights

class_weights = compute_class_weights_6class()
print("Class weights computed:")
for i, weight in class_weights.items():
    print(f"  {SELECTED_CLASSES[i]}: {weight:.3f}")

# ==============================================
# 6. MÔ HÌNH ĐƯỢC CẢI TIẾN CHO 6 CLASSES
# ==============================================

def create_improved_6class_model(num_classes=6, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Tạo mô hình cải tiến cho 6 classes với các kỹ thuật chống overfitting"""
    
    # Sử dụng EfficientNetB0 thay vì B3 để giảm complexity
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze một phần base model để giảm overfitting
    base_model.trainable = True
    # Chỉ fine-tune các layer cuối (ít hơn do chỉ có 6 classes)
    for layer in base_model.layers[:-15]:  # Giảm từ 20 xuống 15
        layer.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Giảm số neurons hơn nữa cho 6 classes
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)  # Giảm từ 128 xuống 64
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Tăng từ 0.3 lên 0.5
    
    # Layer cuối nhỏ hơn
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)  # Giảm từ 64 xuống 32
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Tạo mô hình
model = create_improved_6class_model(num_classes)

# Compile với learning rate thấp hơn
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/2),  # Giảm learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

# ==============================================
# 7. CALLBACKS CẢI TIẾN
# ==============================================

def create_callbacks():
    """Tạo các callbacks để kiểm soát training"""
    
    # Early stopping nghiêm ngặt hơn
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,      # Tăng patience cho 6 classes
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate khi plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,      # Giảm mạnh hơn
        patience=5,      # Tăng patience
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        'best_6class_model.weights.h5',
        monitor='val_accuracy',  # Monitor accuracy thay vì loss
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch, lr):
        if epoch > 25:
            return lr * 0.95
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
    
    return [early_stopping, reduce_lr, checkpoint, lr_scheduler]

callbacks = create_callbacks()
print("Callbacks created successfully!")

# ==============================================
# 8. TRAINING MÔ HÌNH
# ==============================================

print("Starting training with anti-overfitting techniques for 6 classes...")
print(f"Training samples: ~{tf.data.experimental.cardinality(train_ds) * BATCH_SIZE}")
print(f"Validation samples: ~{tf.data.experimental.cardinality(val_ds) * BATCH_SIZE}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Sử dụng class weights
    verbose=1
)

print("Training completed!")

# ==============================================
# 9. ĐÁNH GIÁ MÔ HÌNH
# ==============================================

# Load best weights
model.load_weights('best_6class_model.weights.h5')

# Evaluate on test set
test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=1)
print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Top-1 Accuracy: {test_acc:.4f}")
print(f"Test Top-3 Accuracy: {test_top3:.4f}")

# Evaluate on validation set để so sánh
val_loss, val_acc, val_top3 = model.evaluate(val_ds, verbose=1)
print(f"\nValidation Results:")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Top-1 Accuracy: {val_acc:.4f}")
print(f"Validation Top-3 Accuracy: {val_top3:.4f}")

# ==============================================
# 10. VISUALIZATION VÀ PHÂN TÍCH OVERFITTING
# ==============================================

def plot_training_history_6class(history):
    """Vẽ biểu đồ training history cho 6 classes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('6-Class Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('6-Class Model Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Acc', linewidth=2)
    axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Acc', linewidth=2)
    axes[1, 0].set_title('Top-3 Accuracy', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-3 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting Analysis
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 1].plot(gap, label='Train-Val Accuracy Gap', linewidth=2, color='orange')
    axes[1, 1].set_title('Overfitting Analysis', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_6class_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Phân tích overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    print(f"\n=== PHÂN TÍCH OVERFITTING CHO 6 CLASSES ===")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")
    
    if acc_gap < 0.05:
        print("✅ Tốt! Mô hình 6 classes không bị overfitting nghiêm trọng.")
    elif acc_gap < 0.10:
        print("⚠️  Overfitting nhẹ. Có thể cần điều chỉnh thêm.")
    else:
        print("❌ Vẫn còn overfitting. Cần áp dụng thêm kỹ thuật regularization.")

plot_training_history_6class(history)

# ==============================================
# 11. CONFUSION MATRIX VÀ PHÂN TÍCH CHI TIẾT
# ==============================================

def analyze_6class_performance(model, test_ds, class_names):
    """Phân tích chi tiết performance của model 6 classes"""
    
    # Predict trên test set
    print("Making predictions on test set...")
    y_pred = model.predict(test_ds, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Lấy true labels
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    
    # Classification report
    print("\n=== CLASSIFICATION REPORT FOR 6 CLASSES ===")
    report = classification_report(
        y_true, y_pred_classes, 
        target_names=class_names, 
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix - 6 Classes Improved Model', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_6class_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tìm classes dễ bị nhầm lẫn nhất
    print("\n=== MOST CONFUSED CLASSES ===")
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 2:  # Threshold thấp hơn cho 6 classes
                confused_pairs.append((
                    class_names[i], 
                    class_names[j], 
                    cm[i][j]
                ))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for true_class, pred_class, count in confused_pairs[:5]:
        print(f"{true_class} -> {pred_class}: {count} samples")

# Chạy phân tích
analyze_6class_performance(model, test_ds, class_names)

# ==============================================
# 12. SO SÁNH VỚI MÔ HÌNH GỐC 6 CLASSES
# ==============================================

print("\n=== SO SÁNH VỚI MÔ HÌNH GỐC 6 CLASSES ===")
print("\nMô hình gốc 6 classes (từ kết quả notebook):")
print("- Có thể bị overfitting tương tự")
print("- Sử dụng EfficientNetB3, input 360x360")
print("- Dropout 0.3, không có L2 regularization")

print("\nMô hình cải tiến 6 classes:")
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"- Training Accuracy: {final_train_acc:.3f}")
print(f"- Validation Accuracy: {final_val_acc:.3f}")
print(f"- Accuracy Gap: {final_train_acc - final_val_acc:.3f}")
print(f"- Training Loss: {final_train_loss:.3f}")
print(f"- Validation Loss: {final_val_loss:.3f}")
print(f"- Test Accuracy: {test_acc:.3f}")

print("\n=== CẢI TIẾN ĐÃ ÁP DỤNG CHO 6 CLASSES ===")
improvements = [
    "1. Giảm model complexity: EfficientNetB3 -> EfficientNetB0",
    "2. Giảm input size: 360x360 -> 224x224",
    "3. Giảm neurons: 256 -> 64 -> 32 (phù hợp với 6 classes)",
    "4. Tăng Dropout: 0.3 -> 0.5/0.4",
    "5. Thêm L2 regularization (0.01)",
    "6. Data augmentation mạnh hơn",
    "7. Freeze ít layers hơn (15 thay vì 20)",
    "8. Class weights cho 6 classes",
    "9. Early stopping và learning rate scheduling",
    "10. Tối ưu hóa cho 6 classes"
]

for improvement in improvements:
    print(improvement)

# ==============================================
# 13. LƯU MÔ HÌNH VÀ KẾT QUẢ
# ==============================================

# Lưu mô hình hoàn chỉnh
model.save('improved_6class_interior_design_model.keras')
print("\nModel saved as 'improved_6class_interior_design_model.keras'")

# Lưu kết quả training
import pickle

results = {
    'history': history.history,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'test_top3_accuracy': test_top3,
    'val_accuracy': val_acc,
    'val_loss': val_loss,
    'class_names': class_names,
    'selected_classes': SELECTED_CLASSES,
    'model_config': {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'base_model': 'EfficientNetB0',
        'dropout_rates': [0.5, 0.4],
        'l2_regularization': 0.01,
        'dense_units': [64, 32],
        'num_classes': 6
    }
}

with open('training_results_6class.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Training results saved as 'training_results_6class.pkl'")
print("\n=== HOÀN THÀNH CẢI TIẾN MÔ HÌNH 6 CLASSES ===")
print("Mô hình 6 classes đã được cải tiến để giảm overfitting!")
print(f"6 classes: {SELECTED_CLASSES}")

if __name__ == "__main__":
    print("Script completed successfully!")
    print("Mô hình đã được tối ưu hóa cho 6 classes phong cách thiết kế nội thất.")
    print("Files được tạo:")
    print("- improved_6class_interior_design_model.keras")
    print("- best_6class_model.weights.h5")
    print("- training_results_6class.pkl")
    print("- training_history_6class_improved.png")
    print("- confusion_matrix_6class_improved.png")