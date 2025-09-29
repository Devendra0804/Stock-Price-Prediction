# Stock Prediction GUI Application

## 🎯 Features

### ✨ **User-Friendly Interface**
- **Easy Stock Selection**: Enter any stock symbol or use quick-select buttons for popular stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA)
- **Flexible Time Periods**: Choose from 1 month to 5 years of historical data
- **Real-time Status Updates**: Live progress tracking and detailed logging
- **Interactive Visualizations**: Embedded charts showing predictions and price history

### 📊 **Comprehensive Results Display**
- **Current Stock Price**: Real-time current closing price
- **Next Day Prediction**: ML-based prediction for next trading day
- **Price Change**: Absolute and percentage change prediction
- **Model Accuracy**: Mean Absolute Error (MAE) and data quality metrics
- **Visual Analysis**: Two interactive charts for trend analysis

### 🔮 **Machine Learning Features**
- **Linear Regression Model**: Trained on historical closing prices
- **Performance Metrics**: MAE, RMSE, and accuracy percentages
- **Data Validation**: Automatic data quality checks and error handling
- **Multi-threaded Processing**: Non-blocking UI during data processing

## 🚀 **How to Use**

### **Step 1: Launch the Application**
```bash
python stock_prediction_gui.py
```
or double-click `run_gui.bat` on Windows

### **Step 2: Select a Stock**
- **Option A**: Type any stock symbol (e.g., "AAPL", "TSLA", "NVDA")
- **Option B**: Click one of the quick-select buttons for popular stocks

### **Step 3: Choose Time Period**
- Select from dropdown: 1mo, 3mo, 6mo, 1y, 2y, or 5y
- More data generally provides better model training

### **Step 4: Get Prediction**
- Click "🔮 Get Prediction" button
- Watch the progress bar and status log
- Results will appear in ~10-30 seconds depending on data size

### **Step 5: Analyze Results**
- **Top Panel**: View current price, prediction, and accuracy metrics
- **Bottom Panel**: Interactive charts showing:
  - Actual vs Predicted prices during testing
  - Recent 50-day price history with prediction markers

### **Step 6: Try Different Stocks**
- Click "🗑️ Clear Results" to reset
- Enter a new stock symbol and repeat

## 📱 **Interface Layout**

```
┌─────────────────────────────────────────────────────────────────────┐
│                📈 Stock Price Prediction Tool                      │
├─────────────────────┬───────────────────────────────────────────────┤
│  📊 Stock Selection │              📈 Prediction Results            │
│                     │                                               │
│  Stock Symbol: AAPL │  Stock Symbol: AAPL    Current Price: $253.47│
│  [AAPL][MSFT][GOOGL]│  Predicted Price: $252.42  Change: -$1.05   │
│  [AMZN][TSLA][NVDA] │  Model Accuracy: $2.59 (1.15%)              │
│                     │                                               │
│  Time Period: 1y ▼  │              📊 Visualization                │
│                     │  ┌─────────────────────────────────────────┐  │
│  🔮 Get Prediction   │  │     Actual vs Predicted Prices         │  │
│  🗑️ Clear Results    │  │                                         │  │
│                     │  └─────────────────────────────────────────┘  │
│  [Progress Bar]     │  ┌─────────────────────────────────────────┐  │
│                     │  │       Recent Price History              │  │
│  📝 Status Log:     │  │                                         │  │
│  [12:34:56] Starting│  └─────────────────────────────────────────┘  │
│  [12:34:57] Download│                                               │
│  [12:34:58] Training│                                               │
│  [12:34:59] Complete│                                               │
└─────────────────────┴───────────────────────────────────────────────┘
```

## 🎨 **Visual Features**

### **Color Coding**
- 🟢 **Green**: Positive price changes (gains)
- 🔴 **Red**: Negative price changes (losses)
- 🔵 **Blue**: Current/actual prices in charts
- 🟠 **Orange**: Predicted prices in charts

### **Interactive Charts**
1. **Prediction Accuracy Chart**: Shows how well the model predicted test data
2. **Price History Chart**: Recent 50-day price movement with prediction lines

### **Real-time Updates**
- Progress bar shows processing status
- Status log provides detailed step-by-step updates
- Results update immediately when prediction completes

## ⚡ **Quick Examples**

### **Apple Inc. (AAPL)**
```
Stock Symbol: AAPL
Time Period: 1y
Expected Accuracy: ~1-2% MAE
Typical Response Time: 10-15 seconds
```

### **Tesla (TSLA)**
```
Stock Symbol: TSLA
Time Period: 1y
Expected Accuracy: ~2-4% MAE (more volatile)
Typical Response Time: 10-15 seconds
```

### **Microsoft (MSFT)**
```
Stock Symbol: MSFT
Time Period: 2y
Expected Accuracy: ~0.5-1.5% MAE (stable stock)
Typical Response Time: 15-20 seconds
```

## 🛠️ **Technical Details**

### **Model Information**
- **Algorithm**: Linear Regression (sklearn)
- **Feature**: Previous day's closing price
- **Target**: Next day's closing price
- **Training Split**: 80% training, 20% testing
- **Validation**: Time-series split (chronological order maintained)

### **Data Source**
- **Provider**: Yahoo Finance (via yfinance library)
- **Update Frequency**: Real-time during market hours
- **Historical Range**: Up to 5 years of daily data
- **Data Points**: Typically 250+ days for 1-year period

### **Performance Metrics**
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **MAE Percentage**: Error as percentage of average stock price
- **RMSE**: Root Mean Square Error for overall accuracy assessment
- **R²**: Coefficient of determination (correlation strength)

## 🚨 **Important Disclaimers**

### **⚠️ Educational Use Only**
- This tool is designed for learning and educational purposes
- NOT suitable for actual trading or investment decisions
- Stock markets involve significant financial risks

### **⚠️ Model Limitations**
- Uses only historical price data (no fundamental analysis)
- Cannot predict market crashes, news events, or economic changes
- Simple linear model - real markets are much more complex
- Past performance does not guarantee future results

### **⚠️ Data Considerations**
- Predictions are based on historical patterns only
- Market volatility can significantly affect accuracy
- External factors (news, earnings, economic events) not considered
- Model accuracy varies significantly between different stocks

## 🔧 **Troubleshooting**

### **Common Issues**

**"Not enough data for [SYMBOL]"**
- Try a longer time period (6mo or 1y instead of 1mo)
- Check if the stock symbol is correct
- Some stocks may have limited trading history

**"Internet connection error"**
- Ensure stable internet connection
- Yahoo Finance servers may be temporarily unavailable
- Try again after a few minutes

**"Invalid stock symbol"**
- Verify the stock symbol is correct (e.g., "AAPL" not "Apple")
- Try popular symbols first to test: AAPL, MSFT, GOOGL
- Use uppercase letters for stock symbols

**GUI not responding**
- Wait for prediction to complete (can take 10-30 seconds)
- Don't click buttons multiple times during processing
- Use "Clear Results" if application seems stuck

### **Performance Tips**
- **Shorter time periods** (1mo, 3mo) = faster processing, less accurate
- **Longer time periods** (1y, 2y) = slower processing, more accurate
- **Stable stocks** (AAPL, MSFT) typically have better prediction accuracy
- **Volatile stocks** (TSLA, small caps) may have lower accuracy

## 📋 **System Requirements**

- **Python 3.7+**
- **Internet connection** (for downloading stock data)
- **Required packages**: yfinance, pandas, scikit-learn, matplotlib, tkinter
- **RAM**: 2GB+ recommended
- **Storage**: 100MB+ free space for data caching

## 🆘 **Support**

If you encounter issues:
1. Check the status log for detailed error messages
2. Verify internet connection
3. Try with popular stock symbols (AAPL, MSFT) first
4. Ensure all required packages are installed
5. Restart the application if it becomes unresponsive
