"""
Example script to test the stock prediction model with different stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def quick_prediction(symbol, period="1y"):
    """
    Quick prediction for any stock symbol
    
    Args:
        symbol (str): Stock symbol (e.g., 'MSFT', 'GOOGL', 'TSLA')
        period (str): Time period (default: '1y')
    
    Returns:
        dict: Results including metrics and prediction
    """
    try:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print(f"{'='*50}")
        
        # Download data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if len(data) < 30:
            print(f"Not enough data for {symbol}")
            return None
        
        print(f"Downloaded {len(data)} days of data")
        print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        # Prepare data
        df = data.copy()
        df['Previous_Close'] = df['Close'].shift(1)
        df['Target'] = df['Close']
        df = df.dropna()
        
        X = df[['Previous_Close']].values
        y = df['Target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Next day prediction
        last_close = data['Close'].iloc[-1]
        next_day_pred = model.predict([[last_close]])[0]
        
        # Results
        results = {
            'symbol': symbol,
            'mae': mae,
            'rmse': rmse,
            'mae_percentage': (mae/np.mean(y_test))*100,
            'last_close': last_close,
            'next_day_prediction': next_day_pred,
            'predicted_change': next_day_pred - last_close,
            'predicted_change_pct': ((next_day_pred - last_close)/last_close)*100
        }
        
        print(f"MAE: ${mae:.2f} ({results['mae_percentage']:.2f}%)")
        print(f"RMSE: ${rmse:.2f}")
        print(f"Last Close: ${last_close:.2f}")
        print(f"Predicted Next: ${next_day_pred:.2f}")
        print(f"Predicted Change: ${results['predicted_change']:.2f} ({results['predicted_change_pct']:.2f}%)")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def compare_stocks(symbols):
    """
    Compare prediction accuracy across multiple stocks
    
    Args:
        symbols (list): List of stock symbols to compare
    """
    results = []
    
    for symbol in symbols:
        result = quick_prediction(symbol)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results to compare")
        return
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<10} {'MAE %':<10} {'RMSE':<10} {'Last Price':<12} {'Prediction':<12} {'Change %':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['symbol']:<10} {r['mae_percentage']:<10.2f} ${r['rmse']:<9.2f} ${r['last_close']:<11.2f} ${r['next_day_prediction']:<11.2f} {r['predicted_change_pct']:<10.2f}")

if __name__ == "__main__":
    # List of popular stocks to test
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("Testing stock prediction model on multiple stocks...")
    print("This demonstrates how the model performs across different stocks.")
    
    compare_stocks(test_stocks)
    
    print(f"\n{'='*70}")
    print("NOTES:")
    print("- Lower MAE % indicates better prediction accuracy")
    print("- This simple model works better with less volatile stocks")
    print("- Results can vary significantly between different stocks")
    print("- This is for educational purposes only!")
    print(f"{'='*70}")