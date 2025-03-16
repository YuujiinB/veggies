# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')

# Define paths using raw strings and consistent slashes
train_path = r"Vegetable Images/train"
validation_path = r"Vegetable Images/validation"
test_path = r"Vegetable Images/test"

# Get list of categories
image_categories = os.listdir(train_path)

# Function to plot sample images
def plot_images(image_categories):
    num_categories = len(image_categories)
    rows = int(num_categories**0.5) + 1  # Dynamic grid calculation
    cols = min(num_categories, 4)
    
    plt.figure(figsize=(15, 15))
    for i, cat in enumerate(image_categories):
        # Create full path using os.path.join
        category_path = os.path.join(train_path, cat)
        
        if not os.path.isdir(category_path):
            continue
            
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue
            
        # Plot first image from category
        img_path = os.path.join(category_path, images[0])
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            plt.subplot(rows, cols, i+1)
            plt.imshow(img)
            plt.title(cat)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    plt.tight_layout()
    plt.show()

# Call the function to plot images
plot_images(image_categories)

# Data generators
train_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
train_image_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
val_image_generator = val_gen.flow_from_directory(
    validation_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
test_image_generator = test_gen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for confusion matrix
)

# Print the class encodings
class_map = {v: k for k, v in train_image_generator.class_indices.items()}
print("Class mappings:", class_map)

# Build a custom sequential CNN model
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dense(15, activation='softmax')  # 15 classes
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
early_stopping = keras.callbacks.EarlyStopping(patience=5)
history = model.fit(
    train_image_generator,
    epochs=20,
    validation_data=val_image_generator,
    steps_per_epoch=15000//32,
    validation_steps=3000//32,
    callbacks=[early_stopping]
)

# Plot training history
h = history.history
plt.figure(figsize=(10, 5))
plt.plot(h['loss'], c='red', label='Training Loss')
plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_image_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# ========== CONFUSION MATRIX ==========
# Reset generator and get predictions
test_image_generator.reset()
y_pred = model.predict(test_image_generator, steps=len(test_image_generator))
predicted_labels = np.argmax(y_pred, axis=1)

# Get true labels and class names
true_labels = test_image_generator.classes
class_names = list(test_image_generator.class_indices.keys())

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# ========== INDIVIDUAL PREDICTIONS ==========
def generate_predictions(test_image_path, actual_label):
    # Load and preprocess the image
    test_img = image.load_img(test_image_path, target_size=(150, 150))
    test_img_arr = image.img_to_array(test_img) / 255.0
    test_img_input = test_img_arr.reshape((1, *test_img_arr.shape))
    
    # Make predictions
    predicted_label = np.argmax(model.predict(test_img_input))
    predicted_vegetable = class_map[predicted_label]
    
    # Plot the image and prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img_arr)
    plt.title(f"Predicted: {predicted_vegetable}\nActual: {actual_label}")
    plt.axis('off')
    plt.show()

# Test with an example
test_image_path = "Vegetable Images/test/Broccoli/1011.jpg"  # Updated path
generate_predictions(test_image_path, actual_label='Broccoli')

# Generate predictions for external images
external_image_path_1 = "./tomato eg.png"
generate_predictions(external_image_path_1, actual_label='Tomato')