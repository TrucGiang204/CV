"""
So sánh nhanh giữa mô hình gốc và mô hình cải tiến
Khắc phục hiện tượng overfitting trong phân loại phong cách thiết kế nội thất
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

print("=== SO SÁNH MÔ HÌNH GỐC VÀ MÔ HÌNH CẢI TIẾN ===\n")

# ==============================================
# MÔ HÌNH GỐC (Có overfitting)
# ==============================================

def create_original_model(num_classes=19, input_shape=(360, 360, 3)):
    """Mô hình gốc bị overfitting"""
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True  # Fine-tune toàn bộ
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Dropout thấp
    x = Dense(256, activation='relu')(x)  # Nhiều neurons, không có regularization
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==============================================
# MÔ HÌNH CẢI TIẾN (Chống overfitting)
# ==============================================

def create_improved_model(num_classes=19, input_shape=(224, 224, 3)):
    """Mô hình cải tiến với kỹ thuật chống overfitting"""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True
    
    # Freeze một phần base model
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Giảm neurons và thêm L2 regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Tăng dropout
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# ==============================================
# SO SÁNH THÔNG SỐ MÔ HÌNH
# ==============================================

print("1. SO SÁNH KIẾN TRÚC MÔ HÌNH:")
print("-" * 50)

original_model = create_original_model()
improved_model = create_improved_model()

print(f"MÔ HÌNH GỐC:")
print(f"  - Base Model: EfficientNetB3")
print(f"  - Input Size: 360x360")
print(f"  - Dense Layers: 256 neurons")
print(f"  - Dropout: 0.3")
print(f"  - Regularization: Không có")
print(f"  - Trainable Params: {original_model.count_params():,}")

print(f"\nMÔ HÌNH CẢI TIẾN:")
print(f"  - Base Model: EfficientNetB0")
print(f"  - Input Size: 224x224")
print(f"  - Dense Layers: 128 -> 64 neurons")
print(f"  - Dropout: 0.5 -> 0.4")
print(f"  - Regularization: L2 (0.01)")
print(f"  - Trainable Params: {improved_model.count_params():,}")

param_reduction = (original_model.count_params() - improved_model.count_params()) / original_model.count_params() * 100
print(f"  - Giảm parameters: {param_reduction:.1f}%")

# ==============================================
# SO SÁNH KẾT QUẢ TRAINING (từ kết quả thực tế)
# ==============================================

print(f"\n2. SO SÁNH KẾT QUẢ TRAINING:")
print("-" * 50)

# Kết quả từ mô hình gốc (đã quan sát)
original_results = {
    'train_acc': 0.830,
    'val_acc': 0.644,
    'train_loss': 0.471,
    'val_loss': 1.207,
    'acc_gap': 0.186
}

print("MÔ HÌNH GỐC (Kết quả thực tế):")
print(f"  - Training Accuracy: {original_results['train_acc']:.3f}")
print(f"  - Validation Accuracy: {original_results['val_acc']:.3f}")
print(f"  - Training Loss: {original_results['train_loss']:.3f}")
print(f"  - Validation Loss: {original_results['val_loss']:.3f}")
print(f"  - Accuracy Gap: {original_results['acc_gap']:.3f} ❌ OVERFITTING NGHIÊM TRỌNG")

# Kết quả dự kiến từ mô hình cải tiến
print(f"\nMÔ HÌNH CẢI TIẾN (Dự kiến):")
print(f"  - Training Accuracy: ~0.720 (giảm do regularization)")
print(f"  - Validation Accuracy: ~0.690 (tăng do generalization tốt hơn)")
print(f"  - Training Loss: ~0.650 (cao hơn do regularization)")
print(f"  - Validation Loss: ~0.680 (thấp hơn, gần với training loss)")
print(f"  - Accuracy Gap: ~0.030 ✅ GIẢM OVERFITTING ĐÁNG KỂ")

# ==============================================
# CẢI TIẾN ĐÃ ÁP DỤNG
# ==============================================

print(f"\n3. CÁC KỸ THUẬT CHỐNG OVERFITTING ĐÃ ÁP DỤNG:")
print("-" * 50)

improvements = [
    "✅ Giảm model complexity: EfficientNetB3 → EfficientNetB0",
    "✅ Giảm input size: 360×360 → 224×224", 
    "✅ Tăng Dropout rate: 0.3 → 0.5/0.4",
    "✅ Thêm L2 regularization (0.01)",
    "✅ Giảm neurons: 256 → 128 → 64",
    "✅ Freeze một phần base model (chỉ fine-tune 20 layers cuối)",
    "✅ Data augmentation mạnh hơn (rotation, zoom, contrast, brightness)",
    "✅ Class weights để xử lý imbalanced data",
    "✅ Early stopping nghiêm ngặt hơn (patience=8)",
    "✅ Learning rate scheduling và reduction"
]

for i, improvement in enumerate(improvements, 1):
    print(f"{i:2d}. {improvement}")

# ==============================================
# HƯỚNG DẪN SỬ DỤNG
# ==============================================

print(f"\n4. HƯỚNG DẪN CHẠY MÔ HÌNH CẢI TIẾN:")
print("-" * 50)
print("1. Chạy script chính:")
print("   python improved_interior_design_classifier.py")
print()
print("2. Hoặc import và sử dụng:")
print("   from improved_interior_design_classifier import create_improved_model")
print("   model = create_improved_model(num_classes=19)")
print()
print("3. Kết quả sẽ được lưu:")
print("   - improved_interior_design_model.keras")
print("   - best_improved_model.weights.h5") 
print("   - training_results.pkl")
print("   - training_history_improved.png")
print("   - confusion_matrix_improved.png")

print(f"\n{'='*60}")
print("TỔNG KẾT: Mô hình cải tiến sẽ có performance tổng thể tốt hơn")
print("với khả năng generalization cao hơn và ít overfitting hơn!")
print(f"{'='*60}")

if __name__ == "__main__":
    print("\nScript so sánh hoàn tất!")
    print("Chạy 'python improved_interior_design_classifier.py' để train mô hình cải tiến.")