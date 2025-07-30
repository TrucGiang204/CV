"""
Efficient 6-Class Interior Design Classifier
Mục tiêu: Đạt >80% accuracy trên train, val, test với tối ưu hóa hiệu quả

Chiến lược tối ưu:
1. EfficientNetB1 với fine-tuning thông minh
2. Strong regularization nhưng không quá phức tạp
3. Balanced data augmentation
4. Progressive learning rate và early stopping
5. Class balancing và weighted loss
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ==============================================
# 1. CONFIGURATION
# ==============================================

# Dataset paths
DATASET_PATH = 'dataset_split_6class'
TRAIN_DIR = f'{DATASET_PATH}/train'
VAL_DIR = f'{DATASET_PATH}/val'
TEST_DIR = f'{DATASET_PATH}/test'

# Model parameters - optimized for efficiency
IMG_SIZE = 224  # Keep original size as requested
BATCH_SIZE = 32  # Balanced batch size
EPOCHS = 80     # Reduced epochs with better techniques
BASE_LR = 2e-4  # Slightly higher initial LR

# Classes
CLASSES = ['asian', 'coastal', 'industrial', 'victorian', 'scandinavian', 'southwestern']
NUM_CLASSES = len(CLASSES)

print(f"Classes: {CLASSES}")
print(f"Number of classes: {NUM_CLASSES}")

# ==============================================
# 2. BALANCED DATA AUGMENTATION
# ==============================================

def create_balanced_augmentation():
    """Create balanced augmentation - strong but not overwhelming"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),  # Moderate rotation
        tf.keras.layers.RandomZoom(0.15),      # Moderate zoom
        tf.keras.layers.RandomContrast(0.15),  # Moderate contrast
        tf.keras.layers.RandomBrightness(0.1), # Light brightness
        tf.keras.layers.RandomTranslation(0.05, 0.05),  # Small translation
    ])

def preprocess_image(image, label=None):
    """Preprocess images with normalization"""
    image = tf.cast(image, tf.float32) / 255.0
    if label is not None:
        return image, label
    return image

def create_dataset(directory, is_training=False):
    """Create optimized dataset"""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=is_training,
        seed=42
    )
    
    # Preprocess
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Apply balanced augmentation
        augmentation = create_balanced_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

# Create datasets
print("Creating optimized datasets...")
train_ds = create_dataset(TRAIN_DIR, is_training=True)
val_ds = create_dataset(VAL_DIR, is_training=False)
test_ds = create_dataset(TEST_DIR, is_training=False)

print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds)}")

# ==============================================
# 3. COMPUTE CLASS WEIGHTS
# ==============================================

def compute_class_weights_from_directory(directory, classes):
    """Compute class weights for balanced training"""
    class_counts = []
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts.append(count)
    
    # Calculate weights
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (len(classes) * count)
    
    return class_weights, class_counts

class_weights, class_counts = compute_class_weights_from_directory(TRAIN_DIR, CLASSES)
print("\nClass distribution and weights:")
for i, (class_name, count) in enumerate(zip(CLASSES, class_counts)):
    print(f"  {class_name}: {count} samples, weight: {class_weights[i]:.3f}")

# ==============================================
# 4. OPTIMIZED MODEL ARCHITECTURE
# ==============================================

def create_efficient_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Create efficient model optimized for 80%+ accuracy"""
    
    # Use EfficientNetB1 - good balance of accuracy and efficiency
    base_model = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        drop_connect_rate=0.2
    )
    
    # Smart fine-tuning: freeze early layers, train later ones
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze most layers initially
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Base model with training=False for initial phase
    x = base_model(inputs, training=False)
    
    # Efficient head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Streamlined dense layers with optimal regularization
    x = tf.keras.layers.Dense(256, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Create model
print("Creating efficient model...")
model = create_efficient_model(NUM_CLASSES)

# ==============================================
# 5. OPTIMIZED TRAINING SETUP
# ==============================================

# Use AdamW with weight decay
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=BASE_LR,
    weight_decay=0.01,
    clipnorm=1.0
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
)

model.summary()

# ==============================================
# 6. SMART CALLBACKS
# ==============================================

def create_smart_callbacks():
    """Create optimized callback system"""
    
    # Cosine annealing schedule
    def cosine_schedule(epoch, lr):
        if epoch < 5:  # Warmup
            return BASE_LR * (epoch + 1) / 5
        else:
            # Cosine decay
            progress = (epoch - 5) / (EPOCHS - 5)
            return BASE_LR * 0.5 * (1 + np.cos(np.pi * progress))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=1)
    
    # Smart early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'efficient_6class_model.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Reduce LR on plateau as backup
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=6,
        min_lr=1e-7,
        verbose=1
    )
    
    return [lr_scheduler, early_stopping, checkpoint, reduce_lr]

callbacks = create_smart_callbacks()

# ==============================================
# 7. TWO-STAGE TRAINING
# ==============================================

print("Starting two-stage training...")

# Stage 1: Train head with frozen base (20 epochs)
print("\n=== STAGE 1: Training Head (Frozen Base) ===")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[callbacks[2], callbacks[3]],  # Checkpoint and reduce LR only
    class_weight=class_weights,
    verbose=1
)

# Stage 2: Fine-tune with unfrozen layers
print("\n=== STAGE 2: Fine-tuning (Unfrozen Base) ===")

# Unfreeze more layers gradually
for layer in model.layers[0].layers[-50:]:  # Unfreeze last 50 layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=BASE_LR / 3,  # Lower LR for fine-tuning
        weight_decay=0.01,
        clipnorm=1.0
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS - 20,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Combine histories
history = tf.keras.callbacks.History()
history.history = {}
for key in history_stage1.history:
    history.history[key] = history_stage1.history[key] + history_stage2.history[key]

print("Training completed!")

# ==============================================
# 8. COMPREHENSIVE EVALUATION
# ==============================================

# Load best weights
model.load_weights('efficient_6class_model.weights.h5')

print("\n=== COMPREHENSIVE EVALUATION ===")

# Evaluate on all splits
train_loss, train_acc, train_top2 = model.evaluate(train_ds, verbose=1)
val_loss, val_acc, val_top2 = model.evaluate(val_ds, verbose=1)
test_loss, test_acc, test_top2 = model.evaluate(test_ds, verbose=1)

print(f"\n=== FINAL RESULTS ===")
print(f"Train Accuracy: {train_acc:.4f} ({'✅' if train_acc > 0.8 else '❌'})")
print(f"Val Accuracy:   {val_acc:.4f} ({'✅' if val_acc > 0.8 else '❌'})")
print(f"Test Accuracy:  {test_acc:.4f} ({'✅' if test_acc > 0.8 else '❌'})")
print(f"Overfitting Gap: {train_acc - val_acc:.4f}")

# Check if all targets are met
all_targets_met = all([train_acc > 0.8, val_acc > 0.8, test_acc > 0.8])
overfitting_controlled = (train_acc - val_acc) < 0.08

print(f"\n=== TARGET ACHIEVEMENT ===")
print(f"All accuracy > 80%: {'✅ YES' if all_targets_met else '❌ NO'}")
print(f"Overfitting controlled: {'✅ YES' if overfitting_controlled else '❌ NO'}")

# ==============================================
# 9. VISUALIZATION AND ANALYSIS
# ==============================================

def plot_training_results(history):
    """Plot comprehensive training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-2 Accuracy
    axes[1, 0].plot(history.history['top2_acc'], label='Train Top-2', linewidth=2)
    axes[1, 0].plot(history.history['val_top2_acc'], label='Val Top-2', linewidth=2)
    axes[1, 0].set_title('Top-2 Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-2 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting Analysis
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 1].plot(gap, label='Train-Val Gap', linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Warning Line')
    axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('efficient_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    final_gap = gap[-1]
    print(f"\n=== TRAINING ANALYSIS ===")
    print(f"Final overfitting gap: {final_gap:.4f}")
    if final_gap < 0.03:
        print("✅ Excellent: Minimal overfitting")
    elif final_gap < 0.06:
        print("✅ Good: Controlled overfitting")
    elif final_gap < 0.10:
        print("⚠️  Moderate: Some overfitting present")
    else:
        print("❌ High: Significant overfitting")

plot_training_results(history)

# Confusion Matrix
def create_confusion_matrix():
    """Create and display confusion matrix"""
    # Get predictions
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Efficient 6-Class Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('efficient_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred_classes, 
                                 target_names=CLASSES, digits=4)
    print(report)
    
    # Per-class accuracy
    print("\n=== PER-CLASS ACCURACY ===")
    for i, class_name in enumerate(CLASSES):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_classes[class_mask] == y_true[class_mask])
            print(f"{class_name:15}: {class_acc:.4f} ({'✅' if class_acc > 0.75 else '❌'})")

create_confusion_matrix()

# ==============================================
# 10. SAVE MODEL AND RESULTS
# ==============================================

# Save final model
model.save('efficient_6class_interior_design_model.keras')
print("\nModel saved successfully!")

# Save results
import pickle

results = {
    'history': history.history,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc,
    'overfitting_gap': train_acc - val_acc,
    'all_targets_met': all_targets_met,
    'overfitting_controlled': overfitting_controlled,
    'classes': CLASSES,
    'class_weights': class_weights,
    'class_counts': class_counts,
    'model_config': {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'base_model': 'EfficientNetB1',
        'training_strategy': 'Two-stage progressive',
        'techniques_used': [
            'Class balancing',
            'Progressive fine-tuning',
            'Cosine annealing',
            'Smart early stopping',
            'Balanced augmentation',
            'Weight decay',
            'Gradient clipping'
        ]
    }
}

with open('efficient_6class_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved successfully!")

# ==============================================
# 11. FINAL SUMMARY
# ==============================================

print(f"\n{'='*70}")
print("EFFICIENT 6-CLASS INTERIOR DESIGN CLASSIFIER - FINAL SUMMARY")
print(f"{'='*70}")
print(f"🎯 ACCURACY TARGETS:")
print(f"   Train > 80%: {'✅ ACHIEVED' if train_acc > 0.8 else '❌ FAILED'} ({train_acc:.3f})")
print(f"   Val > 80%:   {'✅ ACHIEVED' if val_acc > 0.8 else '❌ FAILED'} ({val_acc:.3f})")
print(f"   Test > 80%:  {'✅ ACHIEVED' if test_acc > 0.8 else '❌ FAILED'} ({test_acc:.3f})")
print(f"\n📊 OVERFITTING CONTROL:")
print(f"   Train-Val Gap: {train_acc - val_acc:.3f}")
print(f"   Status: {'✅ CONTROLLED' if overfitting_controlled else '❌ NEEDS WORK'}")
print(f"\n🏆 OVERALL SUCCESS:")
if all_targets_met and overfitting_controlled:
    print("   ✅ MISSION ACCOMPLISHED! All targets achieved with controlled overfitting.")
elif all_targets_met:
    print("   ⚠️  PARTIAL SUCCESS: Accuracy targets met but overfitting needs attention.")
else:
    print("   ❌ TARGETS NOT MET: Need further optimization.")
print(f"\n📁 FILES CREATED:")
print(f"   - Model: efficient_6class_interior_design_model.keras")
print(f"   - Weights: efficient_6class_model.weights.h5")
print(f"   - Results: efficient_6class_results.pkl")
print(f"   - Plots: efficient_training_results.png, efficient_confusion_matrix.png")
print(f"\n🎨 CLASSES: {', '.join(CLASSES)}")
print(f"📊 TOTAL SAMPLES: Train={sum(class_counts)}, Val={tf.data.experimental.cardinality(val_ds)*BATCH_SIZE}, Test={tf.data.experimental.cardinality(test_ds)*BATCH_SIZE}")
print(f"{'='*70}")

if __name__ == "__main__":
    print("\n🚀 Efficient 6-class classifier training completed!")
    if all_targets_met and overfitting_controlled:
        print("🎉 SUCCESS: Ready for production use!")
    else:
        print("🔧 Consider additional tuning for optimal performance.")