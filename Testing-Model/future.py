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
    
    # Feature Engineering


# Ensure 'AAPL' is used as the reference for relative strength
    reference_stock = 'AAPL'

# Skip AAPL in the loop since it's used as the denominator
    # Loop through all columns that represent stock close prices
    for stock in merged_df.columns:
        # Ensure you're working with columns that end with '_close' and skip 'reference_stock' (e.g., 'AAPL')
        if '_close' not in stock:
            continue

        stock_name = stock.replace('_close', '')  # Extract the stock symbol without '_close'
        
        # Skip the reference stock
        reference_stock = 'AAPL'  # Example reference stock (can be modified)

        if stock_name == reference_stock:
            continue
        
        # Ensure both columns (stock and reference stock) exist
        if f'{reference_stock}_close' in merged_df.columns and f'{stock_name}_close' in merged_df.columns:
            
            # Relative strength compared to AAPL
            merged_df[f'{stock_name}_rel_strength'] = merged_df[f'{reference_stock}_close'] / merged_df[f'{stock_name}_close']
            
            # Moving averages
            merged_df[f'{stock_name}_ma_5'] = merged_df[f'{stock_name}_close'].rolling(5).mean()
            merged_df[f'{stock_name}_ma_20'] = merged_df[f'{stock_name}_close'].rolling(20).mean()
            
            # Volume changes
            merged_df[f'{stock_name}_vol_change'] = merged_df[f'{stock_name}_volume'].pct_change()
            
            # Debug output to confirm the columns
            print(f'Processed {stock_name}')
        else:
            print(f"Missing columns for {stock_name} or {reference_stock}")


    
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
        'AAPL_close', 'AAPL_volume', 'AAPL_rsi', 'AAPL_macd',
        'AAPL_returns', 'AAPL_signal', 'AAPL_macd_diff'
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
    """
    Recursively forecast future values using the trained LSTM model
    """
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(days):
        # Reshape input for the model
        input_data = current_sequence.reshape(1, window_size, -1)
        
        # Predict next value
        next_pred = model.predict(input_data, verbose=0)[0][0]
        predictions.append(next_pred)
        
        # Update sequence by removing oldest and adding new prediction
        new_row = current_sequence[-1].copy()
        
        # Update AAPL_close with the new prediction (assuming it's the first feature)
        new_row[0] = next_pred
        
        # Update other time-dependent features if needed
        # For simplicity, we'll just shift other features forward
        # In a real application, you might want to update these properly
        for i in range(1, len(new_row)):
            new_row[i] = current_sequence[-1, i]
        
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    # Inverse transform predictions to original price scale
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

# --- 7. Enhanced Forecasting Visualization Function ---
def forecast_and_plot_future(model, X_test, y_test, last_sequence, feature_scaler, target_scaler, window_size, days=30):
    """
    Make future predictions and create a comprehensive visualization
    """
    # Get the actual test values
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    
    # Forecast future values
    future_predictions = forecast_future(model, last_sequence, feature_scaler, target_scaler, window_size, days)
    
    # Create date indices for plotting
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)  # Start from next day
    
    # Plot settings
    plt.figure(figsize=(16, 8))
    
    # Plot historical actual values (last 100 days for better visibility)
    plt.plot(stock_data.index[-100:], stock_data['AAPL_close'].values[-100:], 
             label='Historical Actual', color='blue', linewidth=2)
    
    # Plot test predictions vs actuals
    test_dates = stock_data.index[-len(y_test_unscaled):]
    plt.plot(test_dates, y_test_unscaled, label='Test Actual', color='green', linewidth=2)
    plt.plot(test_dates, y_pred_unscaled, label='Test Predicted', color='orange', linewidth=2)
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label=f'Next {days} Days Forecast', 
             color='red', linestyle='--', marker='o', markersize=5)
    
    # Add confidence interval (using test error standard deviation)
    pred_std = np.std(y_test_unscaled - y_pred_unscaled)
    plt.fill_between(future_dates, 
                    future_predictions - pred_std, 
                    future_predictions + pred_std,
                    color='red', alpha=0.1, label='Confidence Interval')
    
    # Formatting
    plt.title(f"AAPL Stock Price Forecast - Next {days} Days", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price ($)", fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add text box with model metrics
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    mape = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
    textstr = f'Model Performance:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                  fontsize=12, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    return future_predictions

# --- 8. Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data
    stock_data = load_and_preprocess_data()
    
    # Prepare LSTM data
    window_size = 300  # Using 300 days of historical data
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
    
    # Forecast and plot future prices
    last_sequence = X[-1]  # Most recent sequence
    future_days = 30
    future_predictions = forecast_and_plot_future(
        model, X_test, y_test, last_sequence, 
        feature_scaler, target_scaler, window_size, future_days
    )
    
    # Print the future predictions in a table
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    print("\nFuture Price Predictions:")
    print(pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions,
        'Lower Bound (1σ)': future_predictions - np.std(y_true - y_pred),
        'Upper Bound (1σ)': future_predictions + np.std(y_true - y_pred)
    }).to_string(index=False))