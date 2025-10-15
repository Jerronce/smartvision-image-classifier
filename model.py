import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class SmartVisionClassifier:
    """
    Advanced Image Classification Model using Transfer Learning
    with MobileNetV2 for efficient performance.
    
    This model demonstrates:
    - Transfer Learning with pre-trained weights
    - Data Augmentation techniques
    - Fine-tuning strategies
    - Model evaluation and visualization
    """
    
    def __init__(self, img_size=(224, 224), num_classes=10):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, learning_rate=0.001):
        """
        Build the CNN model using MobileNetV2 as base
        """
        # Load pre-trained MobileNetV2 model
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create custom head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layer
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        
        # Preprocessing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision()]
        )
        
        print("Model built successfully!")
        return self.model
    
    def train(self, train_data, val_data, epochs=20, callbacks=None):
        """
        Train the model with early stopping and learning rate reduction
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def fine_tune(self, train_data, val_data, epochs=10):
        """
        Fine-tune the model by unfreezing top layers
        """
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[4]
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Continue training
        history_fine = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs
        )
        
        return history_fine
    
    def predict(self, image, class_names=None):
        """
        Make prediction on a single image
        """
        img_array = tf.expand_dims(image, 0)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if class_names:
            return class_names[predicted_class], confidence
        return predicted_class, confidence
    
    def evaluate_model(self, test_data, class_names):
        """
        Comprehensive model evaluation with metrics and visualizations
        """
        # Get predictions
        predictions = self.model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return y_pred, y_true
    
    def plot_training_history(self):
        """
        Visualize training metrics
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        if 'auc' in self.history.history:
            axes[1, 0].plot(self.history.history['auc'], label='Train')
            axes[1, 0].plot(self.history.history['val_auc'], label='Validation')
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model(self, filepath='smartvision_model.h5'):
        """
        Save the trained model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage demonstration
if __name__ == "__main__":
    # Initialize classifier
    classifier = SmartVisionClassifier(img_size=(224, 224), num_classes=10)
    
    # Build model
    model = classifier.build_model(learning_rate=0.001)
    
    # Display model architecture
    model.summary()
    
    print("\n" + "="*60)
    print("SmartVision Image Classifier - Ready for Training!")
    print("="*60)
    print("\nFeatures:")
    print("✓ Transfer Learning with MobileNetV2")
    print("✓ Data Augmentation")
    print("✓ Early Stopping & Learning Rate Scheduling")
    print("✓ Comprehensive Evaluation Metrics")
    print("✓ Visualization Tools")
    print("="*60)
