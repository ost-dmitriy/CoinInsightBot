#Created by Dmytro Nozhenko

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, InlineQueryHandler, ContextTypes
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pycoingecko import CoinGeckoAPI
import yfinance as yf
from keras.models import load_model
import joblib
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

# Token
TOKEN = 'Your Telegram Token'

# Crypto list
cryptos = ['BTC-USD', 'ETH-USD', 'ARB11841-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'SHIB-USD', 'XRP-USD', 'DOT-USD', 'MATIC-USD', ]

# Comparison of Yahoo Finance tickers with CoinGecko identifiers
ticker_to_coingecko_id = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'ARB11841-USD': 'arbitrum',
    'ADA-USD': 'Cardano',
    'Sol-USD': 'Solana',
    'DOGE-USD': 'Dogecoin',
    'SHIB-USD': 'Shiba Inu',
    'XRP-USD': 'Ripple',
    'DOT-USD': 'Polkadot',
    'MATIC-USD': 'Polygon',

         
    # Extra 
}


def save_models(lstm_model, xgb_model, gb_model):
    lstm_model.save('lstm_model.h5')
    joblib.dump(xgb_model, 'xgb_model.pkl')
    dump(gb_model, 'gb_model.pkl')

def load_models():
    lstm_model = None
    xgb_model = None
    gb_model = None
    
    try:
        lstm_model = load_model('lstm_model.h5')
    except IOError:
        pass
    
    try:
        xgb_model = joblib.load('xgb_model.pkl')
    except IOError:
        pass
    
    try:
        gb_model = load('gb_model.pkl')
    except IOError:
        pass
    
    return lstm_model, xgb_model, gb_model

def fetch_data(crypto):
    data = yf.download(crypto, period='1y')
    return data

def format_number(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"  #B
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"  # M
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"  # K
    else:
        return f"${value:.2f}"

def fetch_crypto_info(crypto_ticker):
    cg = CoinGeckoAPI()
    crypto_id = ticker_to_coingecko_id.get(crypto_ticker, None)
    if crypto_id is None:
        return None  # Or throw an exception if no identifier is found
    info = cg.get_coin_by_id(id=crypto_id)
    market_data = info['market_data']
    return {
        'Market Cap': format_number(market_data['market_cap']['usd']),
        'Volume (24h)': format_number(market_data['total_volume']['usd']),
        'Volume/Market Cap (24h)': f"{(market_data['total_volume']['usd'] / market_data['market_cap']['usd']):.2%}",
        'Circulating Supply': f"{info['market_data']['circulating_supply']:.0f}",
        'Total Supply': f"{info['market_data']['total_supply']:.0f}",
        'Max. Supply': f"{info['market_data']['max_supply']:.0f}" if info['market_data']['max_supply'] else "N/A",
        'Fully Diluted Market Cap': format_number(market_data['fully_diluted_valuation']['usd'])
    }

def calculate_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'].fillna(0, inplace=True)  # NAN

    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    min_price = data['Close'].min()
    max_price = data['Close'].max()
    diff = max_price - min_price
    data['Fib_23.6'] = max_price - 0.236 * diff
    data['Fib_38.2'] = max_price - 0.382 * diff
    data['Fib_50.0'] = max_price - 0.5 * diff
    data['Fib_61.8'] = max_price - 0.618 * diff
    
    return data

def prepare_data(data, look_back):
    data = data['Close'].values
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def forecast_with_sarima(data):
    model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.get_forecast(steps=30)
    forecast_series = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return forecast_series, conf_int

def plot_crypto_data(data, lstm_preds, xgb_preds, gb_preds, y_test, sarima_forecast=None, confidence_interval=None, title=''):
    fig, axes = plt.subplots(6, 1, figsize=(14, 25), sharex=True)
    fig.suptitle(title, fontsize=16)

    # price chart and technical indicator
    axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
    axes[0].plot(data.index, data['SMA_50'], label='50-Day SMA', color='orange', alpha=0.7)
    axes[0].plot(data.index, data['SMA_200'], label='200-Day SMA', color='red', alpha=0.7)
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    axes[1].plot(data.index, data['RSI'], label='RSI', color='orange')
    axes[1].axhline(70, linestyle='--', color='red', label='Overbought')
    axes[1].axhline(30, linestyle='--', color='green', label='Oversold')
    axes[1].set_ylabel('RSI')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    axes[2].plot(data.index, data['MACD'], label='MACD', color='blue')
    axes[2].plot(data.index, data['Signal_Line'], label='Signal Line', color='red', linestyle='--')
    axes[2].set_ylabel('MACD')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    # price prediction chart (needs to be improoved & trained)
    length = len(data)
    axes[3].plot(data.index[-len(lstm_preds):], lstm_preds, label='LSTM Predictions', color='purple', linestyle='--')
    axes[3].plot(data.index[-len(xgb_preds):], xgb_preds, label='XGBoost Predictions', color='green', linestyle='--')
    axes[3].plot(data.index[-len(gb_preds):], gb_preds, label='Gradient Boosting Predictions', color='orange', linestyle='--')
    axes[3].set_ylabel('Price')
    axes[3].legend(loc='upper left')
    axes[3].grid(True)
    axes[3].plot(data.index[-len(y_test):], y_test, label='Actual Price', color='blue', linestyle='-', linewidth=2)
    axes[3].plot(data.index[-len(lstm_preds):], lstm_preds, label='LSTM Predictions', color='purple', linestyle='--', linewidth=2)
    axes[3].plot(data.index[-len(xgb_preds):], xgb_preds, label='XGBoost Predictions', color='green', linestyle='-.', linewidth=2)  


    if sarima_forecast is not None:
        forecast_index = pd.date_range(start=data.index[-1], periods=len(sarima_forecast)+1, inclusive='right')[1:]
        forecast_series = pd.Series(sarima_forecast, index=forecast_index)

        combined_data = pd.concat([data['Close'], forecast_series])

        axes[4].plot(combined_data.index, combined_data, label='Close Price with Forecast', color='purple')

        if confidence_interval is not None:
            ci_lower = pd.Series(confidence_interval['lower Close'], index=forecast_index)
            ci_upper = pd.Series(confidence_interval['upper Close'], index=forecast_index)
            axes[4].fill_between(forecast_index, ci_lower, ci_upper, color='k', alpha=0.1)

        axes[4].set_ylabel('Price')
        axes[4].legend(loc='upper left')
        axes[4].grid(True)

    axes[5].plot(data.index, data['Close'], label='Close Price', color='blue')
    axes[5].axhline(data['Fib_23.6'].iloc[-1], color='yellow', linestyle='--', label='Fib 23.6%')
    axes[5].axhline(data['Fib_38.2'].iloc[-1], color='orange', linestyle='--', label='Fib 38.2%')
    axes[5].axhline(data['Fib_50.0'].iloc[-1], color='red', linestyle='--', label='Fib 50.0%')
    axes[5].axhline(data['Fib_61.8'].iloc[-1], color='purple', linestyle='--', label='Fib 61.8%')
    axes[5].set_ylabel('Price')
    axes[5].legend(loc='upper left')
    axes[5].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('crypto_analysis.png', format='png')


periods = {
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y'
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_first_name = update.effective_user.first_name
    greeting_message = f'Hey {user_first_name}! Select a cryptocurrency to analyse:'

    keyboard = [[InlineKeyboardButton(crypto, callback_data=crypto)] for crypto in cryptos]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(greeting_message, reply_markup=reply_markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    callback_data = query.data
    
    if callback_data in cryptos:
        crypto_ticker = callback_data
        context.user_data['selected_crypto'] = crypto_ticker
        keyboard = [[InlineKeyboardButton(period, callback_data=period)] for period in periods]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(f"You have selected {crypto_ticker}. Now select the period:", reply_markup=reply_markup)
    
    elif callback_data in periods:
        selected_period = periods[callback_data]
        crypto_ticker = context.user_data.get('selected_crypto')
        
        if not crypto_ticker:
            await query.message.reply_text("An error occurred, select the cryptocurrency again.")
            return
        
        await query.answer()
        data = yf.download(crypto_ticker, period=selected_period)
        data = calculate_technical_indicators(data)

        look_back = 60
        X, y, scaler = prepare_data(data, look_back)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        lstm_model, xgb_model, gb_model = load_models()
        
        if lstm_model is None:
            lstm_model = build_lstm_model((X_train.shape[1], 1))
            lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)
            save_models(lstm_model, xgb_model, gb_model)  # Save after training
        lstm_predictions = lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        X_train_xgb = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test_xgb = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

        if xgb_model is None:
            xgb_model = train_xgboost(X_train_xgb, y_train)
            save_models(lstm_model, xgb_model, gb_model)  # Save after training
        xgb_predictions = xgb_model.predict(X_test_xgb)
        xgb_predictions = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))

        if gb_model is None:
            gb_model = train_gradient_boosting(X_train_xgb, y_train)
            save_models(lstm_model, xgb_model, gb_model)  # Save after training
        gb_predictions = gb_model.predict(X_test_xgb)
        gb_predictions = scaler.inverse_transform(gb_predictions.reshape(-1, 1))

        sarima_forecast, sarima_conf_int = forecast_with_sarima(data)

        plot_crypto_data(data, lstm_preds=lstm_predictions, xgb_preds=xgb_predictions, gb_preds=gb_predictions, y_test=y_test, 
                     sarima_forecast=sarima_forecast, confidence_interval=sarima_conf_int, 
                     title=f'{crypto_ticker} Technical Analysis and Forecast ({callback_data})')

        with open('crypto_analysis.png', 'rb') as photo:
            await query.message.reply_photo(photo=photo)

        crypto_info = fetch_crypto_info(crypto_ticker)
        text_info = (
            f"Market Cap: {crypto_info['Market Cap']}\n"
            f"Volume (24h): {crypto_info['Volume (24h)']}\n"
            f"Volume/Market Cap (24h): {crypto_info['Volume/Market Cap (24h)']}\n"
            f"Circulating Supply: {crypto_info['Circulating Supply']}\n"
            f"Total Supply: {crypto_info['Total Supply']}\n"
            f"All-Time High: {crypto_info['All-Time High']}\n"
            f"All-Time High Date: {crypto_info['All-Time High Date']}\n"
        )
        await query.message.reply_text(text_info)




async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    results = []

    for crypto in cryptos:
        if query.lower() in crypto.lower():
            crypto_info = fetch_crypto_info(crypto)
            if crypto_info:
                results.append(
                    InlineQueryResultArticle(
                        id=crypto,
                        title=crypto,
                        input_message_content=InputTextMessageContent(
                            f"{crypto}\n"
                            f"Market Cap: {crypto_info['Market Cap']}\n"
                            f"Volume (24h): {crypto_info['Volume (24h)']}\n"
                            f"Volume/Market Cap (24h): {crypto_info['Volume/Market Cap (24h)']}\n"
                            f"Circulating Supply: {crypto_info['Circulating Supply']}\n"
                            f"Total Supply: {crypto_info['Total Supply']}\n"
                            f"Max. Supply: {crypto_info['Max. Supply']}\n"
                            f"Fully Diluted Market Cap: {crypto_info['Fully Diluted Market Cap']}"
                        )
                    )
                )
    
    await update.inline_query.answer(results)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(InlineQueryHandler(inline_query))
    
    app.run_polling()

if __name__ == '__main__':
    main()
