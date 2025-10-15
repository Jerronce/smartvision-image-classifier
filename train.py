#!/usr/bin/env python3
"""
Training Script for SmartVision Image Classifier

This script demonstrates:
- Data loading and preprocessing
- Model training with callbacks
- Model evaluation and metrics
- Saving trained models

Author: Jerronce
Certified AI/ML Engineer - WorldQuant University
"""

import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import SmartVisionClassifier
import numpy as np
import json
import datetime


def setup_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Create data generators for training and validation
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model(args):
    """
    Main training function
    """
    print("="*70)
    print("SmartVision Image Classifier - Training Pipeline")
    print("="*70)
    print(f"\nData Directory: {args.data_dir}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data generators
    print("\nLoading and preprocessing data...")
    train_gen, val_gen = setup_data_generators(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        validation_split=args.val_split
    )
    
    # Get class information
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Save class indices
    class_indices_path = output_dir / 'class_indices.json'
    with open(class_indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=4)
    print(f"\nClass indices saved to: {class_indices_path}")
    
    # Initialize classifier
    print("\nInitializing SmartVision Classifier...")
    classifier = SmartVisionClassifier(
        img_size=(args.img_size, args.img_size),
        num_classes=num_classes
    )
    
    # Build model
    print("\nBuilding model architecture...")
    model = classifier.build_model(learning_rate=args.learning_rate)
    
    # Display model summary
    if args.verbose:
        model.summary()
    
    # Setup callbacks
    log_dir = output_dir / 'logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / 'training_log.csv')
        )
    ]
    
    # Train model
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    history = classifier.train(
        train_gen,
        val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Fine-tuning (if enabled)
    if args.fine_tune:
        print("\n" + "="*70)
        print("Fine-tuning model...")
        print("="*70 + "\n")
        
        history_fine = classifier.fine_tune(
            train_gen,
            val_gen,
            epochs=args.fine_tune_epochs
        )
    
    # Plot training history
    print("\nGenerating training visualizations...")
    classifier.plot_training_history()
    print(f"Training history plot saved to: training_history.png")
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("Evaluating model...")
    print("="*70)
    
    y_pred, y_true = classifier.evaluate_model(val_gen, class_names)
    print(f"\nConfusion matrix saved to: confusion_matrix.png")
    
    # Calculate final metrics
    val_loss, val_accuracy, val_auc, val_precision = model.evaluate(val_gen, verbose=0)
    
    print("\n" + "="*70)
    print("Final Results:")
    print("="*70)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print("="*70)
    
    # Save final model
    final_model_path = output_dir / 'smartvision_final.h5'
    classifier.save_model(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training configuration
    config = {
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_classes': num_classes,
        'class_names': class_names,
        'final_metrics': {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_auc': float(val_auc),
            'val_precision': float(val_precision)
        },
        'training_date': datetime.datetime.now().isoformat()
    }
    
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to: {config_path}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train SmartVision Image Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to training data directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Fine-tuning arguments
    parser.add_argument('--fine_tune', action='store_true',
                        help='Enable fine-tuning after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for models and logs')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed model information')
    
    args = parser.parse_args()
    
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Train the model
    train_model(args)
