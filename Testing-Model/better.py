import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import ta

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    path = '../tech/'  # Update with your path
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
            # Convert both 'date' columns to datetime without timezone
            merged_df['date'] = pd.to_datetime(merged_df['date'], utc=True).dt.tz_localize(None)
            df_temp['date'] = pd.to_datetime(df_temp['date'], utc=True).dt.tz_localize(None)

            # Now safely merge
            merged_df = pd.merge(merged_df, df_temp, on='date', how='inner')

            #merged_df = pd.merge(merged_df, df_temp, on='date', how='inner')
    
    merged_df = merged_df.sort_values('date').set_index('date')
    merged_df = merged_df.dropna()
    
    # # Feature Engineering
    # for stock in ['MSFT', 'GOOG', 'NVDA', 'ADBE', 'AMZN', 'GOOGL', 'INTC', 'TSLA']:
    #     # Relative strength compared to AAPL
    #     merged_df[f'{stock}_rel_strength'] = merged_df['AAPL_close'] / merged_df[f'{stock}_close']
        
    #     # Moving averages
    #     merged_df[f'{stock}_ma_5'] = merged_df[f'{stock}_close'].rolling(5).mean()
    #     merged_df[f'{stock}_ma_20'] = merged_df[f'{stock}_close'].rolling(20).mean()
        
    #     # Volume changes
    #     merged_df[f'{stock}_vol_change'] = merged_df[f'{stock}_volume'].pct_change()

    tech_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'INTC', 'CSCO', 'ORCL',
            'IBM', 'ADBE', 'CRM', 'TXN', 'QCOM', 'AMD', 'MU', 'AVGO', 'NFLX', 'SHOP',
            'UBER', 'LYFT', 'SNAP', 'TWLO', 'ZM', 'SQ', 'PYPL', 'DOCU', 'ROKU', 'PLTR',
            'NET', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'MDB', 'TEAM', 'NOW', 'FSLY', 'BIDU',
            'BABA', 'JD', 'PDD', 'SE', 'TTD', 'ASML', 'NXPI', 'MRVL', 'WDAY', 'PANW'
        ]

# Ensure 'AAPL' is used as the reference for relative strength
    reference_stock = 'AAPL'

# Skip AAPL in the loop since it's used as the denominator
    for stock in tech_stocks:
        if stock == reference_stock:
            continue
        
        # Relative strength compared to AAPL
        merged_df[f'{stock}_rel_strength'] = merged_df[f'{reference_stock}_close'] / merged_df[f'{stock}_close']
        
        # Moving averages
        merged_df[f'{stock}_ma_5'] = merged_df[f'{stock}_close'].rolling(5).mean()
        merged_df[f'{stock}_ma_20'] = merged_df[f'{stock}_close'].rolling(20).mean()
        
        # Volume changes
        merged_df[f'{stock}_vol_change'] = merged_df[f'{stock}_volume'].pct_change()

    
    # Technical indicators for AAPL
    merged_df['AAPL_returns'] = merged_df['AAPL_close'].pct_change()
    #merged_df['AAPL_rsi'] = ta.momentum.RSIIndicator(merged_df['AAPL_close'])  # Implement RSI
    rsi = ta.momentum.RSIIndicator(merged_df['AAPL_close'])
    merged_df['AAPL_rsi'] = rsi.rsi()

    # merged_df['AAPL_macd'], merged_df['AAPL_signal'] = ta.trend.MACD(merged_df['AAPL_close'])  # Implement MACD
    macd = ta.trend.MACD(merged_df['AAPL_close'])
    merged_df['AAPL_macd'] = macd.macd()
    merged_df['AAPL_signal'] = macd.macd_signal()
    merged_df['AAPL_macd_diff'] = macd.macd_diff()  # Histogram (optional)

    
    # Target variable (next day's AAPL close price)
    merged_df['target'] = merged_df['AAPL_close'].shift(-1)
    merged_df = merged_df.dropna()
    
    return merged_df



# --- 2. Feature Selection and Scaling ---
def prepare_lstm_data(data, window_size=24):
    # Select features - include both AAPL and other stocks' features
    selected_features = [
        'AAPL_close', 'AAPL_volume', 'AAPL_rsi', 'AAPL_macd', 'AAPL_returns',
        'MSFT_close', 'MSFT_rel_strength', 'MSFT_ma_5', 'MSFT_vol_change',
        'GOOG_close', 'GOOG_ma_20', 'GOOG_vol_change',
        'NVDA_close', 'NVDA_rel_strength', 'NVDA_ma_5'
    ]
    
    X_raw = data[selected_features]
    y_raw = data['target']

    # Drop rows with NaN or infinite values
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).dropna()
    y_raw = y_raw.loc[X_raw.index]  # Ensure target matches the filtered features

    # Scale features
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_raw)
    
    # Scale target separately
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y_raw.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(window_size, len(X_scaled)):
        X_seq.append(X_scaled[i-window_size:i])
        y_seq.append(y_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq, feature_scaler, target_scaler

# --- 3. LSTM Model Architecture ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# --- 4. Training Function ---
def train_lstm_model(X_train, y_train, X_val, y_val):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# --- 5. Evaluation Functions ---
def evaluate_model(model, X_test, y_test, target_scaler):
    y_pred_scaled = model.predict(X_test).flatten()
    y_test_scaled = y_test
    
    # Inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot results
    plt.figure(figsize=(14,6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title("AAPL Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    return y_test, y_pred

# --- 6. Forecasting Function ---
def forecast_future(model, last_sequence, feature_scaler, target_scaler, window_size, days=30):
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(days):
        # Reshape input
        input_data = current_sequence.reshape(1, window_size, -1)
        
        # Predict next value
        next_pred = model.predict(input_data, verbose=0)[0][0]
        predictions.append(next_pred)
        
        # Update sequence
        new_row = current_sequence[-1].copy()
        
        # Update AAPL features in the new row (assuming AAPL_close is first feature)
        new_row[0] = next_pred  # Update AAPL_close with prediction
        # You might want to update other AAPL features here as well
        
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    # Inverse transform predictions
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

# --- 7. Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data
    stock_data = load_and_preprocess_data()
    
    # Prepare LSTM data
    window_size = 120  # Using 24 days (approx 1 month) of historical data
    X, y, feature_scaler, target_scaler = prepare_lstm_data(stock_data, window_size)
    
    # Train-test split (80-20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Further split training data for validation
    val_split = int(len(X_train) * 0.9)
    X_train, X_val = X_train[:val_split], X_train[val_split:]
    y_train, y_val = y_train[:val_split], y_train[val_split:]
    
    # Train model
    model, history = train_lstm_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    y_true, y_pred = evaluate_model(model, X_test, y_test, target_scaler)
    
    # Forecast future prices
    last_sequence = X[-1]  # Most recent sequence
    future_days = 30
    future_predictions = forecast_future(
        model, last_sequence, feature_scaler, target_scaler, window_size, future_days
    )
    
    # Plot future predictions
    plt.figure(figsize=(14,6))
    plt.plot(y_true, label='Historical Actual')
    plt.plot(range(len(y_true), len(y_true)+future_days), future_predictions, 
             label='Future Forecast', linestyle='--', color='red')
    plt.title("AAPL Stock Price Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Note: You'll need to implement the compute_rsi() and compute_macd() functions
# or replace them with a technical analysis library like TA-Lib