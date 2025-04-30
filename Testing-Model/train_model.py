import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# For LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# For XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- 1. Load and Preprocess ---
df = pd.read_csv('AAPL.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[df['volume'] > 0]  # remove after-hours or 0-volume rows
df = df[['close', 'volume']]

# --- Optional: Resample to daily ---
# df = df.resample('1D').mean().dropna()

# --- 2. LSTM: Normalize and Prepare Sequences ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_lstm_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # predict close
    return np.array(X), np.array(y)

window_size = 24
X_lstm, y_lstm = create_lstm_sequences(scaled_data, window_size)

# Split into train/test
split = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# --- 3. Train LSTM Model ---
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.1)

# Predict and inverse scale
y_pred_lstm = model_lstm.predict(X_test_lstm)
predicted_close_lstm = scaler.inverse_transform(np.concatenate([y_pred_lstm, np.zeros((len(y_pred_lstm), 1))], axis=1))[:,0]
actual_close_lstm = scaler.inverse_transform(np.concatenate([y_test_lstm.reshape(-1,1), np.zeros((len(y_test_lstm), 1))], axis=1))[:,0]

# --- 4. Prepare Data for XGBoost ---
df['close_shift'] = df['close'].shift(-1)  # predict next period close
df = df.dropna()
features = ['close', 'volume']
X = df[features]
y = df['close_shift']

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.2, shuffle=False)

model_xgb = xgb.XGBRegressor(n_estimators=100)
model_xgb.fit(X_train_xgb, y_train_xgb)

y_pred_xgb = model_xgb.predict(X_test_xgb)

# --- 5. Plot and Compare ---
plt.figure(figsize=(14,6))
plt.plot(actual_close_lstm, label='Actual (LSTM)')
plt.plot(predicted_close_lstm, label='Predicted (LSTM)')
plt.title("LSTM Prediction")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(y_test_xgb.values, label='Actual (XGBoost)')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)')
plt.title("XGBoost Prediction")
plt.legend()
plt.show()
