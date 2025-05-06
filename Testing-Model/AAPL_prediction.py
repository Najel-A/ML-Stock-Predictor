import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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

merged_df = merged_df.sort_values('date').set_index('date').dropna()

# --- 2. Define Target (next-period AAPL_close) ---
merged_df['target'] = merged_df['AAPL_close'].shift(-1)
merged_df = merged_df.dropna()

# --- 3. Feature Selection (exclude AAPL_close itself) ---
selected_features = [col for col in merged_df.columns if col.endswith('_close') and not col.startswith('AAPL')]
X_raw = merged_df[selected_features]
y_raw = merged_df['target']

# --- 4. Scaling ---
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X_raw)

target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y_raw.values.reshape(-1, 1)).flatten()

# --- 5. Sequence Creation ---
def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

window_size = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# --- 6. Train/Test Split ---
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# --- 7. LSTM Model ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# --- 8. Predictions ---
y_pred_test_scaled = model.predict(X_test).flatten()
y_pred_test = target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
y_actual_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Train set predictions for underfitting check
y_pred_train_scaled = model.predict(X_train).flatten()
y_pred_train = target_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
y_actual_train = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

# --- 9. Visualization ---
plt.figure(figsize=(14, 6))
plt.plot(y_actual_test, label='Actual (Test)')
plt.plot(y_pred_test, label='Predicted (Test)')
plt.title("LSTM Prediction on AAPL (Test Set)")
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_actual_train, label='Actual (Train)')
plt.plot(y_pred_train, label='Predicted (Train)')
plt.title("LSTM Prediction on AAPL (Train Set)")
plt.legend()
plt.show()

# --- 10. RMSE Evaluation ---
print("Test RMSE:", np.sqrt(mean_squared_error(y_actual_test, y_pred_test)))
print("Train RMSE:", np.sqrt(mean_squared_error(y_actual_train, y_pred_train)))
