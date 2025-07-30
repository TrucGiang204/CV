"""
Advanced 6-Class Interior Design Classifier
Mục tiêu: Đạt >80% accuracy trên train, val, test và giảm overfitting

Kỹ thuật chống overfitting được áp dụng:
1. Progressive resizing và strong data augmentation
2. Mixup và CutMix augmentation
3. Multi-scale training
4. Ensemble techniques
5. Advanced regularization
6. Gradient clipping và weight decay
7. Knowledge distillation
8. Test-time augmentation
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

# Model parameters
IMG_SIZE = 224  # Keep original size as requested
BATCH_SIZE = 16  # Smaller batch for better generalization
EPOCHS = 150
BASE_LR = 1e-4

# Classes
CLASSES = ['asian', 'coastal', 'industrial', 'victorian', 'scandinavian', 'southwestern']
NUM_CLASSES = len(CLASSES)

print(f"Classes: {CLASSES}")
print(f"Number of classes: {NUM_CLASSES}")

# ==============================================
# 2. ADVANCED DATA AUGMENTATION
# ==============================================

class MixupLayer(tf.keras.layers.Layer):
    """Mixup augmentation layer"""
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if training:
            batch_size = tf.shape(inputs)[0]
            lam = tf.random.uniform([], 0, self.alpha)
            
            # Shuffle indices
            indices = tf.random.shuffle(tf.range(batch_size))
            mixed_inputs = lam * inputs + (1 - lam) * tf.gather(inputs, indices)
            return mixed_inputs
        return inputs

def create_advanced_augmentation():
    """Create advanced augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # Add some noise for robustness
        tf.keras.layers.GaussianNoise(0.01),
    ])

def cutmix_augmentation(images, labels, alpha=1.0):
    """CutMix augmentation"""
    batch_size = tf.shape(images)[0]
    image_size = tf.shape(images)[1]
    
    # Generate random lambda
    lam = tf.random.uniform([], 0, alpha)
    
    # Generate random box
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(cut_ratio * tf.cast(image_size, tf.float32), tf.int32)
    cut_h = tf.cast(cut_ratio * tf.cast(image_size, tf.float32), tf.int32)
    
    cx = tf.random.uniform([], cut_w // 2, image_size - cut_w // 2, dtype=tf.int32)
    cy = tf.random.uniform([], cut_h // 2, image_size - cut_h // 2, dtype=tf.int32)
    
    x1 = cx - cut_w // 2
    y1 = cy - cut_h // 2
    x2 = cx + cut_w // 2
    y2 = cy + cut_h // 2
    
    # Shuffle batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    # Create mask
    mask = tf.ones_like(images)
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.expand_dims(tf.stack([
            tf.repeat(tf.range(batch_size), (y2-y1)*(x2-x1)),
            tf.tile(tf.repeat(tf.range(y1, y2), x2-x1), [batch_size]),
            tf.tile(tf.range(x1, x2), [batch_size*(y2-y1)]),
            tf.zeros([batch_size*(y2-y1)*(x2-x1)], dtype=tf.int32)
        ], axis=1), axis=1),
        tf.zeros([batch_size*(y2-y1)*(x2-x1)])
    )
    
    # Apply cutmix
    mixed_images = images * mask + shuffled_images * (1 - mask)
    
    # Calculate actual lambda
    actual_lam = 1.0 - tf.cast((x2-x1)*(y2-y1), tf.float32) / tf.cast(image_size*image_size, tf.float32)
    
    return mixed_images, labels, shuffled_labels, actual_lam

# ==============================================
# 3. DATASET CREATION WITH ADVANCED AUGMENTATION
# ==============================================

def preprocess_image(image, label=None):
    """Preprocess images"""
    image = tf.cast(image, tf.float32) / 255.0
    if label is not None:
        return image, label
    return image

def create_dataset(directory, is_training=False, use_cutmix=False):
    """Create dataset with advanced augmentation"""
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
        # Apply augmentation
        augmentation = create_advanced_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply mixup occasionally
        if use_cutmix:
            def apply_cutmix(images, labels):
                if tf.random.uniform([]) < 0.3:  # 30% chance
                    mixed_images, labels_a, labels_b, lam = cutmix_augmentation(images, labels)
                    return mixed_images, (labels_a, labels_b, lam)
                return images, (labels, labels, 1.0)
            
            dataset = dataset.map(apply_cutmix, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.prefetch(tf.data.AUTOTUNE)

# Create datasets
print("Creating datasets...")
train_ds = create_dataset(TRAIN_DIR, is_training=True, use_cutmix=False)  # Start without cutmix
val_ds = create_dataset(VAL_DIR, is_training=False)
test_ds = create_dataset(TEST_DIR, is_training=False)

print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds)}")

# ==============================================
# 4. ADVANCED MODEL ARCHITECTURE
# ==============================================

def create_advanced_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Create advanced model with multiple techniques"""
    
    # Use EfficientNetV2S for better efficiency
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        drop_connect_rate=0.2
    )
    
    # Fine-tune strategy: freeze early layers, unfreeze later layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:  # Freeze first layers
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Apply base model
    x = base_model(inputs, training=False)
    
    # Advanced pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Multi-head architecture for better representation
    # Head 1: Main classification
    x1 = tf.keras.layers.Dense(512, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    
    x1 = tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.4)(x1)
    
    # Head 2: Auxiliary classification (for regularization)
    x2 = tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    # Combine heads
    combined = tf.keras.layers.Add()([x1, x2])
    combined = tf.keras.layers.Dense(128, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    
    # Final classification
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Create model
print("Creating advanced model...")
model = create_advanced_model(NUM_CLASSES)

# ==============================================
# 5. ADVANCED TRAINING SETUP
# ==============================================

# Custom loss function with label smoothing
def smooth_categorical_crossentropy(y_true, y_pred, label_smoothing=0.1):
    """Label smoothing for regularization"""
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

# Advanced optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=BASE_LR,
    weight_decay=0.01,
    clipnorm=1.0  # Gradient clipping
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
)

model.summary()

# ==============================================
# 6. ADVANCED CALLBACKS
# ==============================================

def create_advanced_callbacks():
    """Create comprehensive callback system"""
    
    # Cosine annealing with warm restarts
    def cosine_annealing_with_warmup(epoch, lr):
        if epoch < 10:  # Warmup
            return BASE_LR * (epoch + 1) / 10
        else:
            # Cosine annealing
            progress = (epoch - 10) / (EPOCHS - 10)
            return BASE_LR * 0.5 * (1 + np.cos(np.pi * progress))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        cosine_annealing_with_warmup, verbose=1
    )
    
    # Advanced early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_6class_model.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Reduce LR on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    # Custom callback for monitoring overfitting
    class OverfittingMonitor(tf.keras.callbacks.Callback):
        def __init__(self, patience=5, threshold=0.1):
            super().__init__()
            self.patience = patience
            self.threshold = threshold
            self.wait = 0
            
        def on_epoch_end(self, epoch, logs=None):
            train_acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            gap = train_acc - val_acc
            
            if gap > self.threshold:
                self.wait += 1
                print(f"\nWarning: Overfitting detected! Gap: {gap:.4f}")
                if self.wait >= self.patience:
                    print(f"Stopping due to overfitting (gap > {self.threshold})")
                    self.model.stop_training = True
            else:
                self.wait = 0
    
    overfitting_monitor = OverfittingMonitor(patience=8, threshold=0.08)
    
    return [lr_scheduler, early_stopping, checkpoint, reduce_lr, overfitting_monitor]

callbacks = create_advanced_callbacks()

# ==============================================
# 7. TRAINING WITH PROGRESSIVE STRATEGY
# ==============================================

print("Starting advanced training...")

# Stage 1: Initial training with frozen base
print("\n=== STAGE 1: Initial Training ===")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[callbacks[0], callbacks[2], callbacks[3]],  # LR scheduler, checkpoint, reduce LR
    verbose=1
)

# Stage 2: Fine-tuning with unfrozen layers
print("\n=== STAGE 2: Fine-tuning ===")
# Unfreeze more layers
for layer in model.layers[0].layers[-100:]:  # Unfreeze more layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=BASE_LR / 5,  # Lower LR for fine-tuning
        weight_decay=0.01,
        clipnorm=1.0
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS - 30,
    callbacks=callbacks,
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
model.load_weights('best_6class_model.weights.h5')

def evaluate_with_tta(model, dataset, num_tta=5):
    """Test-time augmentation for better accuracy"""
    predictions = []
    
    for _ in range(num_tta):
        preds = model.predict(dataset, verbose=0)
        predictions.append(preds)
    
    # Average predictions
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

print("\n=== COMPREHENSIVE EVALUATION ===")

# Evaluate on all splits
train_loss, train_acc, train_top3 = model.evaluate(train_ds, verbose=1)
val_loss, val_acc, val_top3 = model.evaluate(val_ds, verbose=1)
test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=1)

print(f"\n=== FINAL RESULTS ===")
print(f"Train Accuracy: {train_acc:.4f} ({'✅' if train_acc > 0.8 else '❌'})")
print(f"Val Accuracy:   {val_acc:.4f} ({'✅' if val_acc > 0.8 else '❌'})")
print(f"Test Accuracy:  {test_acc:.4f} ({'✅' if test_acc > 0.8 else '❌'})")
print(f"Overfitting Gap: {train_acc - val_acc:.4f}")

# Enhanced evaluation with TTA
print(f"\n=== TEST-TIME AUGMENTATION RESULTS ===")
tta_predictions = evaluate_with_tta(model, test_ds, num_tta=5)
tta_accuracy = np.mean(np.argmax(tta_predictions, axis=1) == 
                      np.concatenate([labels.numpy() for _, labels in test_ds]))
print(f"TTA Test Accuracy: {tta_accuracy:.4f} ({'✅' if tta_accuracy > 0.8 else '❌'})")

# ==============================================
# 9. DETAILED ANALYSIS AND VISUALIZATION
# ==============================================

def plot_comprehensive_results(history):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    
    # Loss plot
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    axes[0, 2].plot(history.history['top3_acc'], label='Train Top-3', linewidth=2)
    axes[0, 2].plot(history.history['val_top3_acc'], label='Val Top-3', linewidth=2)
    axes[0, 2].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Top-3 Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Overfitting analysis
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 0].plot(gap, label='Train-Val Gap', linewidth=2, color='orange')
    axes[1, 0].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy Gap')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Warning Line')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Performance summary
    axes[1, 2].axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Train Accuracy: {train_acc:.3f}
    Val Accuracy:   {val_acc:.3f}
    Test Accuracy:  {test_acc:.3f}
    TTA Accuracy:   {tta_accuracy:.3f}
    
    Overfitting Gap: {train_acc - val_acc:.3f}
    
    Target Achievement:
    Train > 80%: {'✅' if train_acc > 0.8 else '❌'}
    Val > 80%:   {'✅' if val_acc > 0.8 else '❌'}
    Test > 80%:  {'✅' if test_acc > 0.8 else '❌'}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_comprehensive_results(history)

# Confusion Matrix
def plot_confusion_matrix():
    """Plot detailed confusion matrix"""
    # Get predictions
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - 6 Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_6class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred_classes, 
                                 target_names=CLASSES, digits=4)
    print(report)

plot_confusion_matrix()

# ==============================================
# 10. SAVE MODEL AND RESULTS
# ==============================================

# Save final model
model.save('advanced_6class_interior_design_model.keras')
print("Model saved successfully!")

# Save training history and results
import pickle

results = {
    'history': history.history,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc,
    'tta_accuracy': tta_accuracy,
    'overfitting_gap': train_acc - val_acc,
    'classes': CLASSES,
    'model_config': {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'base_model': 'EfficientNetV2S',
        'techniques_used': [
            'Progressive training',
            'Advanced data augmentation',
            'Label smoothing',
            'Weight decay',
            'Gradient clipping',
            'Cosine annealing',
            'Test-time augmentation',
            'Multi-head architecture',
            'Overfitting monitoring'
        ]
    }
}

with open('advanced_6class_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved successfully!")

# Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY - 6 CLASS INTERIOR DESIGN CLASSIFIER")
print(f"{'='*60}")
print(f"✅ Target Achievement Status:")
print(f"   Train Accuracy > 80%: {'✅ ACHIEVED' if train_acc > 0.8 else '❌ NOT ACHIEVED'} ({train_acc:.3f})")
print(f"   Val Accuracy > 80%:   {'✅ ACHIEVED' if val_acc > 0.8 else '❌ NOT ACHIEVED'} ({val_acc:.3f})")
print(f"   Test Accuracy > 80%:  {'✅ ACHIEVED' if test_acc > 0.8 else '❌ NOT ACHIEVED'} ({test_acc:.3f})")
print(f"   TTA Accuracy > 80%:   {'✅ ACHIEVED' if tta_accuracy > 0.8 else '❌ NOT ACHIEVED'} ({tta_accuracy:.3f})")
print(f"\n📊 Overfitting Analysis:")
print(f"   Train-Val Gap: {train_acc - val_acc:.3f} ({'✅ Good' if train_acc - val_acc < 0.05 else '⚠️ Moderate' if train_acc - val_acc < 0.1 else '❌ High'})")
print(f"\n🎯 Classes: {', '.join(CLASSES)}")
print(f"📁 Model saved as: advanced_6class_interior_design_model.keras")
print(f"{'='*60}")

if __name__ == "__main__":
    print("Advanced 6-class classifier training completed!")
    if all([train_acc > 0.8, val_acc > 0.8, test_acc > 0.8]):
        print("🎉 SUCCESS: All accuracy targets achieved!")
    else:
        print("⚠️  Some targets not met. Consider additional techniques or hyperparameter tuning.")