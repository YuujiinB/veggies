import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("vegetable_prices.csv", parse_dates=["Date"], index_col=["Date"])

# User input for vegetable selection
print("\nAvailable vegetables:")
unique_commodities = df['Commodity'].unique()
for i, veg in enumerate(unique_commodities, start=1):
    print(f"{i}. {veg}")

selected_commodity = None
while True:
    try:
        selected_number = int(input("\nEnter the vegetable number: "))
        if 1 <= selected_number <= len(unique_commodities):
            selected_commodity = unique_commodities[selected_number - 1]
            break
        else:
            print("Invalid number. Try again.")
    except ValueError:
        print("Invalid input. Enter a number.")

print(f"\nSelected: {selected_commodity}")

# Filter data for the selected vegetable
df_selected = df[df['Commodity'] == selected_commodity].drop(['Commodity'], axis=1).dropna()
if df_selected.empty:
    print(f"No data for {selected_commodity}. Exiting.")
    exit()

# Plot raw data
plt.figure(figsize=(10, 6))
plt.plot(df_selected.index, df_selected['Average'], label='Price')
plt.title(f"Price of {selected_commodity}")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prepare normalized series
series = df_selected['Average'].to_numpy()
min_val, max_val = np.min(series), np.max(series)
series_normalized = (series - min_val) / (max_val - min_val)

# Split data
split_time = int(len(series_normalized) * 0.8)
x_train_norm, x_valid_norm = series_normalized[:split_time], series_normalized[split_time:]

# Dynamic window_size adjustment
window_size = 30  # Initial value
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

try:
    train_dataset = windowed_dataset(x_train_norm, window_size, batch_size, shuffle_buffer_size)
    val_dataset = windowed_dataset(x_valid_norm, window_size, batch_size, shuffle_buffer_size)
except tf.errors.InvalidArgumentError:
    print("Error: Insufficient data for windowing. Try a smaller window_size or more data.")
    exit()

# Model architecture
model = Sequential([
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

model.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=lr_schedule), metrics=['mae'])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, callbacks=[early_stopping])

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
def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

forecast_series = series_normalized[split_time - window_size:]
if len(forecast_series) < window_size:
    forecast_series = series_normalized[-window_size:]  # Use last available data

forecast = model_forecast(model, forecast_series, window_size, batch_size)
results = forecast.squeeze() * (max_val - min_val) + min_val
x_valid_denorm = x_valid_norm * (max_val - min_val) + min_val

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(df_selected.index[split_time:], x_valid_denorm, label='Actual', color='blue')
plt.plot(df_selected.index[split_time:], results[:len(x_valid_denorm)], label='Predicted', color='red')
plt.title(f"Predictions for {selected_commodity}")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluation metrics
def evaluate_preds(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype=y_true.dtype)
    
    mae = tf.reduce_mean(tf.abs(y_true - y_pred)).numpy()
    mse = tf.reduce_mean(tf.square(y_true - y_pred)).numpy()
    rmse = tf.sqrt(mse).numpy()
    mape = 100 * tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)).numpy()
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

results_eval = evaluate_preds(x_valid_denorm, results[:len(x_valid_denorm)])
print("\nEvaluation Results:")
for metric, value in results_eval.items():
    print(f"{metric}: {value:.4f}")