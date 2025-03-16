# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')

# ========== VEGETABLE IMAGE CLASSIFICATION ==========
# Define paths
train_path = r"Vegetable Images/train"
validation_path = r"Vegetable Images/validation"
test_path = r"Vegetable Images/test"

# Get list of categories
image_categories = os.listdir(train_path)

# Data generators
train_gen = ImageDataGenerator(rescale=1.0/255.0)
train_image_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1.0/255.0)
val_image_generator = val_gen.flow_from_directory(
    validation_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_image_generator = test_gen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for confusion matrix
)

# Class mappings
class_map = {v: k for k, v in train_image_generator.class_indices.items()}

# Build and compile the CNN model
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

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_image_generator)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Function to classify vegetable from an image
def classify_vegetable(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_arr = image.img_to_array(img) / 255.0
    img_input = img_arr.reshape((1, *img_arr.shape))
    predicted_label = np.argmax(model.predict(img_input))
    predicted_vegetable = class_map[predicted_label]
    return predicted_vegetable

# ========== PRICE PREDICTION ==========
# Load the dataset
df = pd.read_csv("vegetable_prices.csv", parse_dates=["Date"], index_col=["Date"])

# Function to predict prices for a given vegetable
def predict_prices(vegetable_name):
    # Filter data for the selected vegetable
    df_selected = df[df['Commodity'].str.contains(vegetable_name, case=False)].dropna()
    if df_selected.empty:
        print(f"No data for {vegetable_name}. Exiting.")
        return
    
    # Prepare normalized series
    series = df_selected['Average'].to_numpy()
    min_val, max_val = np.min(series), np.max(series)
    series_normalized = (series - min_val) / (max_val - min_val)

    # Split data
    split_time = int(len(series_normalized) * 0.8)
    x_train_norm, x_valid_norm = series_normalized[:split_time], series_normalized[split_time:]

    # Dynamic window_size adjustment
    window_size = 30
    if len(x_train_norm) < window_size + 1:
        window_size = max(1, len(x_train_norm) - 1)
        print(f"Adjusted window_size to {window_size} due to limited training data.")

    # Dataset preparation
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
        dataset = dataset.map(lambda w: (w[:-1], w[-1]))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    batch_size = 64
    shuffle_buffer_size = 1000

    train_dataset = windowed_dataset(x_train_norm, window_size, batch_size, shuffle_buffer_size)
    val_dataset = windowed_dataset(x_valid_norm, window_size, batch_size, shuffle_buffer_size)

    # Model architecture
    price_model = Sequential([
        Dense(128, activation='relu', input_shape=[window_size], kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Training configuration
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.96
    )
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

    price_model.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=lr_schedule), metrics=['mae'])
    history = price_model.fit(train_dataset, validation_data=val_dataset, epochs=200, callbacks=[early_stopping])

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Prediction and evaluation
    forecast_series = series_normalized[split_time - window_size:]
    if len(forecast_series) < window_size:
        forecast_series = series_normalized[-window_size:]  # Use last available data

    forecast = price_model.predict(forecast_series.reshape(1, -1))
    results = forecast.squeeze() * (max_val - min_val) + min_val
    x_valid_denorm = x_valid_norm * (max_val - min_val) + min_val

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(df_selected.index[split_time:], x_valid_denorm, label='Actual', color='blue')
    plt.plot(df_selected.index[split_time:], results[:len(x_valid_denorm)], label='Predicted', color='red')
    plt.title(f"Predictions for {vegetable_name}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# ========== MAIN WORKFLOW ==========
# Classify vegetable from an image
test_image_path = "Vegetable Images/test/Broccoli/1011.jpg"  # Replace with your image path
predicted_vegetable = classify_vegetable(test_image_path)
print(f"Predicted Vegetable: {predicted_vegetable}")

# Predict prices for the predicted vegetable
predict_prices(predicted_vegetable)