#!/usr/bin/env python3
"""
Plastic Type Classification using CNN
Deep Learning model to classify 7 types of plastic materials
Optimized for maximum accuracy
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ====================== CONFIGURATION ======================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset" / "Plastic Classification(1)"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create output directories
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "graphs").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

# Model hyperparameters (optimized for best accuracy)
IMG_SIZE = 128  # Reduced to prevent overfitting
BATCH_SIZE = 32  # Good balance
EPOCHS = 100  # Sufficient with early stopping
LEARNING_RATE = 0.001  # Higher for better convergence
CLASS_NAMES = ['HDPE', 'LDPA', 'Other', 'PET', 'PP', 'PS', 'PVC']

print("="*80)
print("PLASTIC TYPE CLASSIFICATION - DEEP LEARNING MODEL")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Epochs: {EPOCHS}")
print("="*80)

# ====================== DATA LOADING ======================
print("\n[1/6] Loading and preprocessing data...")

# Strong data augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR / 'train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = val_test_datagen.flow_from_directory(
    DATASET_DIR / 'validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    DATASET_DIR / 'test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_gen.samples}")
print(f"✓ Validation samples: {val_gen.samples}")
print(f"✓ Test samples: {test_gen.samples}")

# ====================== MODEL BUILDING ======================
print("\n[2/6] Building optimized CNN model (balanced for small dataset)...")

model = keras.Sequential([
    # Input
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Block 1
    layers.Conv2D(32, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),
    
    # Block 2
    layers.Conv2D(64, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),
    
    # Block 3
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.4),
    
    # Block 4
    layers.Conv2D(256, 3, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling2D(),
    
    # Dense layers with heavy regularization
    layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Compile with optimal settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"✓ Model built: {model.count_params():,} parameters")

# ====================== CALLBACKS ======================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.005
    ),
    ModelCheckpoint(
        OUTPUT_DIR / 'models' / 'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ====================== TRAINING ======================
print("\n[3/6] Training model...")
print("="*80)

start_time = datetime.now()
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)
training_time = datetime.now() - start_time

print("\n" + "="*80)
print(f"✓ Training completed in {training_time}")
print("="*80)

# ====================== EVALUATION ======================
print("\n[4/6] Evaluating model on test set...")

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=0)
f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0

print("\n" + "="*80)
print("FINAL TEST RESULTS")
print("="*80)
print(f"Test Accuracy:  {test_acc*100:.2f}%")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall:    {test_rec:.4f}")
print(f"Test F1-Score:  {f1_score:.4f}")
print("="*80)

# Predictions for confusion matrix
test_gen.reset()
predictions = model.predict(test_gen, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# ====================== VISUALIZATIONS ======================
print("\n[5/6] Generating visualizations...")

# Training history
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'graphs' / 'training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: training_history.png")

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'graphs' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: confusion_matrix.png")

# Sample predictions
test_gen.reset()
x_batch, y_batch = next(test_gen)
pred_batch = model.predict(x_batch, verbose=0)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for i in range(16):
    if i < len(x_batch):
        axes[i].imshow(x_batch[i])
        true_label = CLASS_NAMES[np.argmax(y_batch[i])]
        pred_label = CLASS_NAMES[np.argmax(pred_batch[i])]
        confidence = np.max(pred_batch[i]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                         fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.suptitle('Sample Test Predictions', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictions' / 'sample_predictions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: sample_predictions.png")

# ====================== DETAILED REPORT ======================
print("\n[6/6] Generating detailed report...")

print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(true_classes, pred_classes, target_names=CLASS_NAMES))

# Save training info
training_info = {
    'model_config': {
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['loss']),
        'learning_rate': LEARNING_RATE,
        'total_parameters': int(model.count_params())
    },
    'dataset': {
        'train_samples': int(train_gen.samples),
        'val_samples': int(val_gen.samples),
        'test_samples': int(test_gen.samples),
        'num_classes': 7,
        'class_names': CLASS_NAMES
    },
    'results': {
        'train_accuracy': float(history.history['accuracy'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1_score': float(f1_score),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    },
    'training_time': str(training_time),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(OUTPUT_DIR / 'models' / 'training_info.json', 'w') as f:
    json.dump(training_info, f, indent=4)

# Save model
model.save(OUTPUT_DIR / 'models' / 'plastic_classifier_final.keras')
print(f"\n✓ Model saved: plastic_classifier_final.keras")
print(f"✓ Training info saved: training_info.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"\nAll outputs saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  • outputs/models/best_model.keras")
print("  • outputs/models/plastic_classifier_final.keras")
print("  • outputs/models/training_info.json")
print("  • outputs/graphs/training_history.png")
print("  • outputs/graphs/confusion_matrix.png")
print("  • outputs/predictions/sample_predictions.png")
print("\n" + "="*80)
