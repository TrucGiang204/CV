"""
Cải Tiến Mô Hình Phân Loại Phong Cách Thiết Kế Nội Thất
Khắc Phục Hiện Tượng Overfitting

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
TEST_LABELS = 'dataset/test_labels.csv'

# Classes
class_names = sorted(os.listdir(TRAIN_DIR))
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")

# ==============================================
# 2. DATA AUGMENTATION MẠNH HƠN
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

def create_dataset(directory, validation_split=0.0, subset=None, shuffle=True, augment=False):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=123,
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
# 3. TẠO DATASETS
# ==============================================

# Training và validation datasets
train_ds = create_dataset(
    TRAIN_DIR, 
    validation_split=VALIDATION_SPLIT, 
    subset="training", 
    augment=True  # Chỉ augment cho training
)

val_ds = create_dataset(
    TRAIN_DIR, 
    validation_split=VALIDATION_SPLIT, 
    subset="validation", 
    augment=False  # Không augment cho validation
)

# Test dataset
test_ds = create_dataset(TEST_DIR, shuffle=False, augment=False)

print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds)}")

# ==============================================
# 4. TÍNH CLASS WEIGHTS ĐỂ XỬ LÝ IMBALANCE
# ==============================================

def compute_class_weights(train_dir, class_names):
    """Tính class weights để xử lý imbalanced data"""
    class_counts = []
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
        class_counts.append(count)
    
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (len(class_names) * count)
    
    return class_weights

class_weights = compute_class_weights(TRAIN_DIR, class_names)
print("Class weights computed:")
for i, weight in class_weights.items():
    print(f"  {class_names[i]}: {weight:.3f}")

# ==============================================
# 5. MÔ HÌNH ĐƯỢC CẢI TIẾN
# ==============================================

def create_improved_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Tạo mô hình cải tiến với các kỹ thuật chống overfitting"""
    
    # Sử dụng EfficientNetB0 thay vì B3 để giảm complexity
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze một phần base model để giảm overfitting
    base_model.trainable = True
    # Chỉ fine-tune các layer cuối
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Giảm số neurons và thêm regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)  # Giảm từ 256 xuống 128
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Tăng từ 0.3 lên 0.5
    
    # Thêm một layer nhỏ hơn
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Tạo mô hình
model = create_improved_model(num_classes)

# Compile với learning rate thấp hơn
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/2),  # Giảm learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

# ==============================================
# 6. CALLBACKS CẢI TIẾN
# ==============================================

def create_callbacks():
    """Tạo các callbacks để kiểm soát training"""
    
    # Early stopping nghiêm ngặt hơn
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,      # Giảm từ 10 xuống 8
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate khi plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,      # Giảm mạnh hơn
        patience=4,      # Giảm patience
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        'best_improved_model.weights.h5',
        monitor='val_accuracy',  # Monitor accuracy thay vì loss
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch, lr):
        if epoch > 20:
            return lr * 0.95
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
    
    return [early_stopping, reduce_lr, checkpoint, lr_scheduler]

callbacks = create_callbacks()
print("Callbacks created successfully!")

# ==============================================
# 7. TRAINING MÔ HÌNH
# ==============================================

print("Starting training with anti-overfitting techniques...")
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
# 8. ĐÁNH GIÁ MÔ HÌNH
# ==============================================

# Load best weights
model.load_weights('best_improved_model.weights.h5')

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
# 9. VISUALIZATION VÀ PHÂN TÍCH OVERFITTING
# ==============================================

def plot_training_history(history):
    """Vẽ biểu đồ training history để phân tích overfitting"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14)
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
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Phân tích overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    print(f"\n=== PHÂN TÍCH OVERFITTING ===")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")
    
    if acc_gap < 0.05:
        print("✅ Tốt! Mô hình không bị overfitting nghiêm trọng.")
    elif acc_gap < 0.10:
        print("⚠️  Overfitting nhẹ. Có thể cần điều chỉnh thêm.")
    else:
        print("❌ Vẫn còn overfitting. Cần áp dụng thêm kỹ thuật regularization.")

plot_training_history(history)

# ==============================================
# 10. CONFUSION MATRIX VÀ PHÂN TÍCH CHI TIẾT
# ==============================================

def analyze_model_performance(model, test_ds, class_names):
    """Phân tích chi tiết performance của model"""
    
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
    print("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(
        y_true, y_pred_classes, 
        target_names=class_names, 
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix - Improved Model', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tìm classes dễ bị nhầm lẫn nhất
    print("\n=== MOST CONFUSED CLASSES ===")
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 5:  # Threshold cho confusion
                confused_pairs.append((
                    class_names[i], 
                    class_names[j], 
                    cm[i][j]
                ))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for true_class, pred_class, count in confused_pairs[:10]:
        print(f"{true_class} -> {pred_class}: {count} samples")

# Chạy phân tích
analyze_model_performance(model, test_ds, class_names)

# ==============================================
# 11. SO SÁNH VỚI MÔ HÌNH GỐC
# ==============================================

print("\n=== SO SÁNH VỚI MÔ HÌNH GỐC ===")
print("\nMô hình gốc (từ kết quả đã thấy):")
print("- Training Accuracy: ~0.830")
print("- Validation Accuracy: ~0.644")
print("- Accuracy Gap: ~0.186 (overfitting nghiêm trọng)")
print("- Training Loss: ~0.471")
print("- Validation Loss: ~1.207 (tăng so với training)")

print("\nMô hình cải tiến:")
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

print("\n=== CẢI TIẾN ĐÃ ÁP DỤNG ===")
improvements = [
    "1. Giảm model complexity: EfficientNetB3 -> EfficientNetB0",
    "2. Giảm image size: 360x360 -> 224x224",
    "3. Tăng Dropout: 0.3 -> 0.5",
    "4. Thêm L2 regularization (0.01)",
    "5. Giảm neurons: 256 -> 128 -> 64",
    "6. Data augmentation mạnh hơn (rotation, zoom, contrast, brightness)",
    "7. Freeze một phần base model",
    "8. Class weights để xử lý imbalance",
    "9. Early stopping nghiêm ngặt hơn",
    "10. Learning rate scheduling"
]

for improvement in improvements:
    print(improvement)

# ==============================================
# 12. LƯU MÔ HÌNH VÀ KẾT QUẢ
# ==============================================

# Lưu mô hình hoàn chỉnh
model.save('improved_interior_design_model.keras')
print("\nModel saved as 'improved_interior_design_model.keras'")

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
    'model_config': {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'base_model': 'EfficientNetB0',
        'dropout_rates': [0.5, 0.4],
        'l2_regularization': 0.01,
        'dense_units': [128, 64]
    }
}

with open('training_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Training results saved as 'training_results.pkl'")
print("\n=== HOÀN THÀNH CẢI TIẾN MÔ HÌNH ===")
print("Mô hình đã được cải tiến để giảm overfitting!")

if __name__ == "__main__":
    print("Script completed successfully!")
    print("Run this script to train the improved model.")
    print("Make sure you have the dataset in the correct directory structure.")