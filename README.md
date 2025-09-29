A machine learning project that predicts stock prices using historical data and linear regression. Now includes both command-line and graphical user interface (GUI) versions!

🎯 Two Ways to Use
1. 🖥️ Graphical Interface (Recommended)
Easy-to-use GUI with interactive charts and real-time updates:

python stock_prediction_gui.py
2. 📝 Command Line Interface
Traditional script for automated analysis:

python stock_prediction.py
Overview
This project demonstrates:

Downloading stock data using yfinance
Data preprocessing with pandas
Training a linear regression model with scikit-learn
Evaluating model performance
Visualizing results with matplotlib
Features
🎨 GUI Version Features:
Interactive Interface: Easy stock symbol entry with quick-select buttons
Real-time Visualizations: Embedded charts showing predictions and trends
Live Progress Tracking: Status updates and progress bars
Multiple Time Periods: 1 month to 5 years of historical data
Comprehensive Results: Current price, predictions, accuracy metrics
Error Handling: User-friendly error messages and validation
📊 Core ML Features:
Downloads stock data using yfinance
Data preprocessing with pandas
Training a linear regression model with scikit-learn
Evaluating model performance
Visualizing results with matplotlib
Installation
Install required packages:
pip install -r requirements.txt
Usage
🖥️ GUI Version (Recommended for Beginners)
python stock_prediction_gui.py
How to use the GUI:

Enter Stock Symbol: Type any symbol (e.g., AAPL, TSLA) or use quick-select buttons
Choose Time Period: Select from dropdown (1mo to 5y)
Click "Get Prediction": Wait 10-30 seconds for results
View Results: See current price, prediction, accuracy, and interactive charts
Try Different Stocks: Use "Clear Results" and repeat
See GUI_GUIDE.md for detailed GUI documentation.

📝 Command Line Version
python stock_prediction.py
The script will:

Download AAPL stock data for the past year
Prepare the data for machine learning
Split into training/testing sets (80/20)
Train a linear regression model
Evaluate performance and show metrics
Display visualizations
Predict the next day's closing price
📊 Compare Multiple Stocks
Run the comparison script to test the model on multiple stocks:

python compare_stocks.py
This script tests the model on AAPL, MSFT, GOOGL, AMZN, and TSLA to show how prediction accuracy varies across different stocks.

Model Details
Algorithm: Linear Regression
Feature: Previous day's closing price
Target: Current day's closing price
Train/Test Split: 80/20 (chronological split)
Evaluation Metrics
MAE (Mean Absolute Error): Average absolute difference between actual and predicted prices
RMSE (Root Mean Square Error): Square root of average squared differences
Percentage Error: MAE as percentage of average price
Visualizations
The script generates two plots:

Time series plot showing actual vs predicted prices over time
Scatter plot of actual vs predicted prices with perfect prediction line
Important Disclaimer
⚠️ This is an educational project only!

The model is extremely simple and not suitable for real trading
Stock markets are influenced by many factors not captured in this model
DO NOT use this for actual investment decisions
Past performance does not guarantee future results
Customization
You can modify the script to:

Change the stock symbol (edit STOCK_SYMBOL variable)
Adjust the time period (edit PERIOD variable)
Add more features (volume, moving averages, etc.)
Try different algorithms (Random Forest, Neural Networks, etc.)
Academic Use
This project is perfect for:

Learning basic machine learning concepts
Understanding time series data
Practicing data visualization
Demonstrating the challenges of financial prediction
Requirements
Python 3.7+
Internet connection (for downloading stock data)
All packages listed in requirements.txt
