"""
Simple Stock Price Prediction using Historical Data

This script demonstrates a basic approach to predicting stock prices using:
- yfinance for downloading stock data
- pandas for data handling
- scikit-learn for linear regression
- matplotlib for visualization

The model predicts the next day's closing price based on the previous day's closing price.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def download_stock_data(symbol, period="1y"):
    """
    Download historical stock data using yfinance
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period (e.g., '1y' for 1 year)
    
    Returns:
        pandas.DataFrame: Historical stock data
    """
    print(f"Downloading {symbol} data for the past {period}...")
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    print(f"Downloaded {len(data)} days of data")
    return data

def prepare_data(data):
    """
    Prepare data for machine learning by creating features
    
    Args:
        data (pandas.DataFrame): Raw stock data
    
    Returns:
        tuple: (features, targets) for ML model
    """
    print("Preparing data for machine learning...")
    
    # Create a copy of the data
    df = data.copy()
    
    # Create feature: previous day's closing price
    df['Previous_Close'] = df['Close'].shift(1)
    
    # Create target: current day's closing price
    df['Target'] = df['Close']
    
    # Remove rows with missing values (first row will have NaN for Previous_Close)
    df = df.dropna()
    
    # Extract features and targets
    X = df[['Previous_Close']].values
    y = df['Target'].values
    
    print(f"Prepared {len(X)} samples for training")
    return X, y, df

def train_model(X_train, y_train):
    """
    Train a linear regression model
    
    Args:
        X_train: Training features
        y_train: Training targets
    
    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    """
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model trained. Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        tuple: (predictions, mae, rmse)
    """
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"Average actual price: ${np.mean(y_test):.2f}")
    print(f"MAE as % of average price: {(mae/np.mean(y_test))*100:.2f}%")
    
    return predictions, mae, rmse

def plot_results(y_test, predictions, symbol):
    """
    Plot actual vs predicted prices
    
    Args:
        y_test: Actual prices
        predictions: Predicted prices
        symbol: Stock symbol
    """
    print("Creating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time series of actual vs predicted
    ax1.plot(range(len(y_test)), y_test, label='Actual Prices', color='blue', alpha=0.7)
    ax1.plot(range(len(predictions)), predictions, label='Predicted Prices', color='red', alpha=0.7)
    ax1.set_title(f'{symbol} Stock Price Prediction - Actual vs Predicted')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of actual vs predicted
    ax2.scatter(y_test, predictions, alpha=0.5, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Prices ($)')
    ax2.set_ylabel('Predicted Prices ($)')
    ax2.set_title('Actual vs Predicted Prices (Scatter Plot)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plot_filename = f'{symbol}_stock_prediction.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{plot_filename}'")
    plt.close()  # Close the figure to free memory

def predict_next_day(model, last_close_price, symbol):
    """
    Predict the next day's closing price
    
    Args:
        model: Trained model
        last_close_price: Last known closing price
        symbol: Stock symbol
    
    Returns:
        float: Predicted next day closing price
    """
    next_day_prediction = model.predict([[last_close_price]])[0]
    print(f"\nPrediction for next trading day:")
    print(f"{symbol} - Last Close: ${last_close_price:.2f}")
    print(f"{symbol} - Predicted Next Close: ${next_day_prediction:.2f}")
    print(f"Predicted Change: ${next_day_prediction - last_close_price:.2f} ({((next_day_prediction - last_close_price)/last_close_price)*100:.2f}%)")
    
    return next_day_prediction

def main():
    """
    Main function to run the stock prediction pipeline
    """
    # Configuration
    STOCK_SYMBOL = "AAPL"  # Apple Inc.
    PERIOD = "1y"  # 1 year of data
    
    try:
        # Step 1: Download stock data
        stock_data = download_stock_data(STOCK_SYMBOL, PERIOD)
        
        # Display basic info about the data
        print(f"\nStock data summary:")
        print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        print(f"Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
        
        # Step 2: Prepare data for ML
        X, y, processed_data = prepare_data(stock_data)
        
        # Step 3: Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        predictions, mae, rmse = evaluate_model(model, X_test, y_test)
        
        # Step 6: Plot results
        plot_results(y_test, predictions, STOCK_SYMBOL)
        
        # Step 7: Predict next day
        last_close = stock_data['Close'].iloc[-1]
        next_day_pred = predict_next_day(model, last_close, STOCK_SYMBOL)
        
        print("\n" + "="*50)
        print("DISCLAIMER: This is a simple academic model.")
        print("DO NOT use this for actual trading decisions!")
        print("Real stock prediction requires much more sophisticated models.")
        print("="*50)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure you have an internet connection to download stock data.")

if __name__ == "__main__":
    main()