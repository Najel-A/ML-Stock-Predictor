import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# --- 1. Load and Preprocess All Stock Data ---
def load_and_preprocess_data():
    path = '../data/'  # Adjust to your actual CSV folder path
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

    # Set AAPL close as target
    merged_df['target'] = merged_df['AAPL_close'].shift(-1)
    merged_df = merged_df.dropna()

    return merged_df

# --- 2. Prepare Data for XGBoost ---
def prepare_data(df):
    # Only use features from stocks other than AAPL
    input_features = [col for col in df.columns if col not in ['AAPL_close', 'target'] and col.endswith(('_close', '_volume'))]

    X = df[input_features]
    y = df['target']

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]  # Align y with cleaned X

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, shuffle=False), scaler

# --- 3. Train and Evaluate Model ---
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    plt.figure(figsize=(14,6))
    plt.plot(y_test.values, label='Actual AAPL')
    plt.plot(y_pred, label='Predicted AAPL')
    plt.title("AAPL Stock Price Prediction (XGBoost, No AAPL Features)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# --- 4. Main Execution ---
if __name__ == "__main__":
    df = load_and_preprocess_data()
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
