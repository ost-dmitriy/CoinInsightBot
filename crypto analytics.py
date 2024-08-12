import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

# Fetch historical market data
def fetch_data(ticker):
    data = yf.download(ticker, start='2023-01-01', end='2024-08-07')
    return data

# Calculate technical indicators
def calculate_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).apply(
        lambda x: (x[x > 0].mean() / -x[x < 0].mean()), raw=False)))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Volatility'] = data['Close'].rolling(window=14).std()
    data.dropna(inplace=True)
    return data

# Prepare data for LSTM and XGBoost
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train XGBoost model with hyperparameter tuning
def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Train Gradient Boosting model
def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

# Forecast using SARIMA
def forecast_with_sarima(data, days=7):
    data = data['Close']
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit()
    forecast = results.get_forecast(steps=days)
    forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(days=1), periods=days)
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    conf_int = forecast.conf_int()
    return forecast_series, conf_int

# Plotting function for technical indicators and forecast
def plot_crypto_data(data, lstm_preds, xgb_preds, gb_preds, sarima_forecast=None, confidence_interval=None, title=''):
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Price with SMA
    axes[0].plot(data['Close'], label='Close Price')
    axes[0].plot(data['SMA_50'], label='50-Day SMA', alpha=0.7)
    axes[0].plot(data['SMA_200'], label='200-Day SMA', alpha=0.7)
    axes[0].set_ylabel('Price')
    axes[0].legend()
    
    # RSI
    axes[1].plot(data['RSI'], label='RSI', color='orange')
    axes[1].axhline(70, linestyle='--', color='red')
    axes[1].axhline(30, linestyle='--', color='green')
    axes[1].set_ylabel('RSI')
    axes[1].legend()
    
    # MACD
    axes[2].plot(data['MACD'], label='MACD', color='blue')
    axes[2].plot(data['Signal_Line'], label='Signal Line', color='red', linestyle='--')
    axes[2].set_ylabel('MACD')
    axes[2].legend()
    
    # LSTM and XGBoost Predictions
    axes[3].plot(data.index[-len(lstm_preds):], lstm_preds, label='LSTM Predictions')
    axes[3].plot(data.index[-len(xgb_preds):], xgb_preds, label='XGBoost Predictions')
    axes[3].plot(data.index[-len(gb_preds):], gb_preds, label='Gradient Boosting Predictions')
    axes[3].set_ylabel('Price')
    axes[3].legend()
    
    # Forecast
    if sarima_forecast is not None:
        forecast_index = pd.date_range(start=data.index[-1], periods=len(sarima_forecast)+1, inclusive='right')[1:]
        forecast_series = pd.Series(sarima_forecast, index=forecast_index)
        
        # Append the forecast to the original data
        combined_data = pd.concat([data['Close'], forecast_series])
        
        # Plot the combined data
        axes[4].plot(combined_data, label='Close Price with Forecast', color='purple')
        
        if confidence_interval is not None:
            ci_lower = pd.Series(confidence_interval['lower Close'], index=forecast_index)
            ci_upper = pd.Series(confidence_interval['upper Close'], index=forecast_index)
            axes[4].fill_between(forecast_index, ci_lower, ci_upper, color='k', alpha=0.1)
        
        axes[4].set_ylabel('Price')
        axes[4].legend()
    
    plt.show()

# Main code
cryptos = ['BTC-USD', 'ETH-USD', 'ARB11841-USD']
look_back = 60

for crypto in cryptos:
    data = fetch_data(crypto)
    data = calculate_technical_indicators(data)
    
    # Prepare data for LSTM and other models
    X, y, scaler = prepare_data(data, look_back)
    
    # Split into train and test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    # Make predictions with LSTM
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate LSTM
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    print(f'{crypto} LSTM RMSE: {lstm_rmse}')
    
    # Prepare data for XGBoost
    X_train_xgb = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test_xgb = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    xgb_model = train_xgboost(X_train_xgb, y_train)
    
    # Make predictions with XGBoost
    xgb_predictions = xgb_model.predict(X_test_xgb)
    xgb_predictions = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))
    
    # Evaluate XGBoost
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    print(f'{crypto} XGBoost RMSE: {xgb_rmse}')
    
    # Train Gradient Boosting model
    gb_model = train_gradient_boosting(X_train_xgb, y_train)
    
    # Make predictions with Gradient Boosting
    gb_predictions = gb_model.predict(X_test_xgb)
    gb_predictions = scaler.inverse_transform(gb_predictions.reshape(-1, 1))
    
    # Evaluate Gradient Boosting
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    print(f'{crypto} Gradient Boosting RMSE: {gb_rmse}')
    
    # Forecast using SARIMA
    sarima_forecast, sarima_conf_int = forecast_with_sarima(data)
    
    # Plot technical indicators and forecasts
    plot_crypto_data(data, lstm_preds=lstm_predictions, xgb_preds=xgb_predictions, gb_preds=gb_predictions, 
                     sarima_forecast=sarima_forecast, confidence_interval=sarima_conf_int, title=f'{crypto} Technical Analysis and Forecast')
