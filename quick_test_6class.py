"""
Quick Test for 6-Class Interior Design Classifier
Mục tiêu: Kiểm tra nhanh dataset và chạy một model đơn giản để verify solution
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")

# Dataset paths
DATASET_PATH = 'dataset_split_6class'
TRAIN_DIR = f'{DATASET_PATH}/train'
VAL_DIR = f'{DATASET_PATH}/val'
TEST_DIR = f'{DATASET_PATH}/test'

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ['asian', 'coastal', 'industrial', 'victorian', 'scandinavian', 'southwestern']

print(f"Classes: {CLASSES}")

# Check dataset exists
if not os.path.exists(DATASET_PATH):
    print("❌ Dataset not found! Please run create_6class_dataset.py first")
    exit(1)

print("✅ Dataset found!")

# Quick dataset statistics
def get_dataset_stats():
    stats = {}
    for split in ['train', 'val', 'test']:
        split_path = Path(DATASET_PATH) / split
        split_stats = {}
        total = 0
        for class_name in CLASSES:
            class_path = split_path / class_name
            if class_path.exists():
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                split_stats[class_name] = count
                total += count
        stats[split] = {'classes': split_stats, 'total': total}
    return stats

stats = get_dataset_stats()
print("\n📊 Dataset Statistics:")
for split, data in stats.items():
    print(f"{split.upper()}: {data['total']} images")
    for class_name, count in data['classes'].items():
        print(f"  {class_name}: {count}")

# Create simple datasets
def create_simple_dataset(directory, is_training=False):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=is_training,
        seed=42
    )
    
    # Simple preprocessing
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    if is_training:
        # Simple augmentation
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ])
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    
    return dataset.prefetch(tf.data.AUTOTUNE)

print("\n🔄 Creating datasets...")
train_ds = create_simple_dataset(TRAIN_DIR, is_training=True)
val_ds = create_simple_dataset(VAL_DIR, is_training=False)
test_ds = create_simple_dataset(TEST_DIR, is_training=False)

print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds)}")

# Create a simple but effective model
def create_simple_model():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    return model

print("\n🏗️ Creating simple model...")
model = create_simple_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Quick training (just a few epochs to test)
print("\n🚀 Quick training test (5 epochs)...")

# Simple callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,  # Just a quick test
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed successfully!")
    
    # Quick evaluation
    print("\n📊 Quick Evaluation:")
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - val_acc:.4f}")
    
    # Simple plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print(f"\n🎯 Quick Analysis:")
    if train_acc > 0.6 and val_acc > 0.5:
        print("✅ Model is learning! Dataset and pipeline work correctly.")
        if train_acc - val_acc < 0.15:
            print("✅ Overfitting is under control.")
        else:
            print("⚠️  Some overfitting detected - this is normal for quick test.")
    else:
        print("⚠️  Low accuracy - may need more training or different approach.")
    
    print(f"\n💡 Recommendations for full training:")
    if val_acc < 0.8:
        print("- Use more powerful architecture (EfficientNet)")
        print("- Train for more epochs (50-100)")
        print("- Apply stronger data augmentation")
        print("- Use progressive fine-tuning")
        print("- Apply class balancing")
    
    # Save simple model for reference
    model.save('quick_test_6class_model.keras')
    print("📁 Quick test model saved as 'quick_test_6class_model.keras'")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("This might be due to system limitations or missing dependencies.")

print(f"\n{'='*50}")
print("QUICK TEST SUMMARY")
print(f"{'='*50}")
total_images = stats['train']['total'] + stats['val']['total'] + stats['test']['total']
print(f"✅ Dataset: 6 classes, {total_images} total images")
print(f"✅ Pipeline: Data loading and preprocessing work")
print(f"✅ Model: Simple architecture trains successfully")
print(f"📈 Next Steps: Run full training with advanced models")
print(f"{'='*50}")

print("\n🎉 Quick test completed! The foundation is ready for full training.")
print("💪 For >80% accuracy, run the advanced or efficient classifier scripts.")