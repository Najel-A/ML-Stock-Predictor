import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import os
import ta
from scipy.stats import randint as sp_randint

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    path = '../tech/'  # Update with your path
    all_files = glob.glob(path + '*.csv')
    
    # Initialize list to store DataFrames
    dfs = []
    
    for file in all_files:
        stock_name = os.path.basename(file).replace(".csv", "")
        df_temp = pd.read_csv(file, usecols=['date', 'close', 'volume'])
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp = df_temp.rename(columns={
            'close': f'{stock_name}_close',
            'volume': f'{stock_name}_volume'
        })
        dfs.append(df_temp.set_index('date'))
    
    # Merge all DataFrames at once
    merged_df = pd.concat(dfs, axis=1)
    merged_df = merged_df.dropna()
    
    # Feature Engineering
    reference_stock = 'AAPL'
    
    # Prepare lists for new features to add all at once
    new_features = {}
    
    for stock in merged_df.columns:
        if '_close' not in stock:
            continue

        stock_name = stock.replace('_close', '')
        if stock_name == reference_stock:
            continue
        
        if f'{reference_stock}_close' in merged_df.columns and f'{stock_name}_close' in merged_df.columns:
            # Store new features in dictionary first
            new_features[f'{stock_name}_rel_strength'] = merged_df[f'{reference_stock}_close'] / merged_df[f'{stock_name}_close']
            new_features[f'{stock_name}_ma_5'] = merged_df[f'{stock_name}_close'].rolling(5).mean()
            new_features[f'{stock_name}_ma_20'] = merged_df[f'{stock_name}_close'].rolling(20).mean()
            new_features[f'{stock_name}_vol_change'] = merged_df[f'{stock_name}_volume'].pct_change()
            print(f'Processed {stock_name}')
        else:
            print(f"Missing columns for {stock_name} or {reference_stock}")

    # Add all new features at once
    merged_df = pd.concat([merged_df, pd.DataFrame(new_features)], axis=1)
    
    # Technical indicators for AAPL
    merged_df['AAPL_returns'] = merged_df['AAPL_close'].pct_change()
    
    # Calculate RSI and handle infinite values
    with np.errstate(divide='ignore', invalid='ignore'):
        rsi = ta.momentum.RSIIndicator(merged_df['AAPL_close'])
        merged_df['AAPL_rsi'] = rsi.rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(merged_df['AAPL_close'])
    merged_df['AAPL_macd'] = macd.macd()
    merged_df['AAPL_signal'] = macd.macd_signal()
    merged_df['AAPL_macd_diff'] = macd.macd_diff()
    
    # Target variable
    merged_df['target'] = merged_df['AAPL_close'].shift(-1)
    
    # Replace infinite values with NaN and then drop
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df = merged_df.dropna()
    
    return merged_df

# --- 2. Feature Selection and Scaling ---
def prepare_rf_data(data, window_size=24):
    # Select base features
    selected_features = [
        'AAPL_close', 'AAPL_volume', 'AAPL_rsi', 'AAPL_macd',
        'AAPL_returns', 'AAPL_signal', 'AAPL_macd_diff'
    ]
    
    # Create lag features more efficiently
    lag_features = []
    for feature in selected_features:
        for i in range(1, window_size+1):
            lag_features.append(data[feature].shift(i).rename(f'{feature}_lag_{i}'))
    
    # Combine all features at once
    X = pd.concat([data[selected_features]] + lag_features, axis=1)
    y = data['target']
    
    # Drop rows with NaN values (created by shifting)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Scale features and handle potential infinite values
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Scale target
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, feature_scaler, target_scaler, X.columns.tolist()

# --- 3. Random Forest Model ---
def build_rf_model():
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Hyperparameter tuning setup
    param_dist = {
        'n_estimators': sp_randint(50, 200),
        'max_depth': sp_randint(5, 30),
        'min_samples_split': sp_randint(2, 20),
        'min_samples_leaf': sp_randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    return random_search

# --- 4. Training Function ---
def train_rf_model(X_train, y_train, X_val, y_val):
    model = build_rf_model()
    
    # Combine train and val for RF (since we're doing cross-validation)
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    model.fit(X_train_full, y_train_full)
    
    print("Best parameters found: ", model.best_params_)
    print("Best score: ", -model.best_score_)
    
    return model.best_estimator_

# --- 5. Evaluation Functions ---
def evaluate_model(model, X_test, y_test, target_scaler):
    y_pred_scaled = model.predict(X_test)
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
    plt.title("AAPL Stock Price Prediction (Random Forest)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    return y_test, y_pred

# --- 6. Feature Importance ---
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# --- 7. Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data
    stock_data = load_and_preprocess_data()
    
    # Prepare data for Random Forest
    window_size = 300  # Use this to adjust the days of historical data to use
    X, y, feature_scaler, target_scaler, feature_names = prepare_rf_data(stock_data, window_size)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, shuffle=False  # 0.125 x 0.8 = 0.1 (10% validation)
    )
    
    # Train model
    model = train_rf_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    y_true, y_pred = evaluate_model(model, X_test, y_test, target_scaler)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)