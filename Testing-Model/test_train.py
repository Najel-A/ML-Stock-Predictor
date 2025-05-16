import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# XGBoost
import xgboost as xgb

# --- 1. Load and Merge CSVs ---
path = '../data/'  # Update this based on your folder structure
all_files = glob.glob(path + '*.csv')

merged_df = pd.DataFrame()

for file in all_files:
    stock_name = os.path.basename(file).replace(".csv", "")
    df_temp = pd.read_csv(file, usecols=['date', 'close', 'volume'])
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    df_temp = df_temp.rename(columns={
        'close': f'{stock_name}_close',
        'volume': f'{stock_name}_volume'
    })
    if merged_df.empty:
        merged_df = df_temp
    else:
        merged_df = pd.merge(merged_df, df_temp, on='date', how='inner')

merged_df = merged_df.sort_values('date').set_index('date')
merged_df = merged_df.dropna()

# --- 2. Define Target ---
merged_df['target'] = merged_df['AAPL_close'].shift(-1)
merged_df = merged_df.dropna()
print(merged_df.head())

# --- 3. LSTM Section (with Selected Features and Proper Scaling) ---
selected_features = ['AAPL_close', 'MSFT_close', 'GOOG_close', 'NVDA_close']
X_lstm_raw = merged_df[selected_features]
y_lstm_raw = merged_df['AAPL_close'].shift(-1).dropna()

# Align shapes
X_lstm_raw = X_lstm_raw.iloc[:-1, :]

# Scale inputs
feature_scaler = MinMaxScaler()
X_scaled_lstm = feature_scaler.fit_transform(X_lstm_raw)

# Scale target separately
target_scaler = MinMaxScaler()
y_scaled_lstm = target_scaler.fit_transform(y_lstm_raw.values.reshape(-1, 1)).flatten()

# Create sequences
def create_lstm_sequences(data, target, window):
    X_seq, y_seq = [], []
    for i in range(window, len(data)):
        X_seq.append(data[i-window:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)

window_size = 24
X_lstm_seq, y_lstm_seq = create_lstm_sequences(X_scaled_lstm, y_scaled_lstm, window_size)

# Train/test split
split = int(len(X_lstm_seq) * 0.8)
X_train_lstm, X_test_lstm = X_lstm_seq[:split], X_lstm_seq[split:]
y_train_lstm, y_test_lstm = y_lstm_seq[:split], y_lstm_seq[split:]

# Train LSTM with EarlyStopping
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Predict and inverse scale
y_pred_lstm_scaled = model_lstm.predict(X_test_lstm).flatten()
y_test_lstm_scaled = y_test_lstm
y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_actual_lstm = target_scaler.inverse_transform(y_test_lstm_scaled.reshape(-1, 1)).flatten()

# --- 4. XGBoost Section (All Stocks) ---
features = [col for col in merged_df.columns if col != 'target']
X = merged_df[features]
y = merged_df['target']

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.2, shuffle=False)

model_xgb = xgb.XGBRegressor(n_estimators=100)
model_xgb.fit(X_train_xgb, y_train_xgb)

y_pred_xgb = model_xgb.predict(X_test_xgb)

# --- 5. Plot Results ---
plt.figure(figsize=(14,6))
plt.plot(y_actual_lstm, label='Actual (LSTM)')
plt.plot(y_pred_lstm, label='Predicted (LSTM)')
plt.title("Improved LSTM Prediction on AAPL")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(y_test_xgb.values, label='Actual (XGBoost)')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)')
plt.title("XGBoost Prediction on AAPL")
plt.legend()
plt.show()

# Forecasting 30 days (approx. 30 months of trading days)
future_days = 30
lstm_input_seq = X_scaled_lstm[-window_size:]  # Last window of training data
predicted_future_lstm = []

for _ in range(future_days):
    input_reshaped = lstm_input_seq.reshape(1, window_size, X_scaled_lstm.shape[1])
    next_pred_scaled = model_lstm.predict(input_reshaped, verbose=0)[0][0]
    predicted_future_lstm.append(next_pred_scaled)

    # Append new prediction and remove the oldest step to simulate moving window
    new_input_row = lstm_input_seq[-1].copy()
    new_input_row[0] = next_pred_scaled  # Updating AAPL_close with predicted
    lstm_input_seq = np.vstack((lstm_input_seq[1:], new_input_row))
    
# Inverse scale future predictions
predicted_future_lstm = target_scaler.inverse_transform(np.array(predicted_future_lstm).reshape(-1, 1)).flatten()

# Plot future prediction
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_actual_lstm)), y_actual_lstm, label="Actual")
plt.plot(range(len(y_actual_lstm), len(y_actual_lstm) + future_days), predicted_future_lstm, label="Forecast (Future)", linestyle="--")
plt.title("LSTM Forecast for Future AAPL Prices (3 Months Ahead)")
plt.legend()
plt.show()


# --- 6. RMSE Comparison ---
print("Improved LSTM RMSE:", np.sqrt(mean_squared_error(y_actual_lstm, y_pred_lstm)))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb)))