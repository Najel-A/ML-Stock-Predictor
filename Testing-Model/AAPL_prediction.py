import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# XGBoost
import xgboost as xgb

# --- 1. Load and Merge CSVs ---
path = '../data/'  # Adjust if needed
all_files = glob.glob(path + '*.csv')

merged_df = pd.DataFrame()

for file in all_files:
    stock_name = file.split('\\')[-1].replace(".csv", "")
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

# --- 2. Define Target (next-period AAPL_close) ---
merged_df['target'] = merged_df['AAPL_close'].shift(-1)
merged_df = merged_df.dropna()

# --- 3. LSTM Section: Use All *_close Except AAPL_close ---
selected_features = [col for col in merged_df.columns if col.endswith('_close') and not col.startswith('AAPL')]
X_lstm_raw = merged_df[selected_features]
y_lstm_raw = merged_df['target']

# Scale inputs
feature_scaler = MinMaxScaler()
X_scaled_lstm = feature_scaler.fit_transform(X_lstm_raw)

# Scale target
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
y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_actual_lstm = target_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

# --- 4. XGBoost Section ---
# Remove AAPL_close from features
features = [col for col in merged_df.columns if col not in ['AAPL_close', 'target']]
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
plt.title("LSTM Prediction of AAPL Close Using Other Tech Stocks")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(y_test_xgb.values, label='Actual (XGBoost)')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)')
plt.title("XGBoost Prediction of AAPL Close Using Other Tech Stocks")
plt.legend()
plt.show()

# --- 6. RMSE Comparison ---
print("LSTM RMSE:", np.sqrt(mean_squared_error(y_actual_lstm, y_pred_lstm)))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb)))
