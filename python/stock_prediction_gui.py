"""
Stock Price Prediction GUI Application

A user-friendly graphical interface for stock price prediction using machine learning.
Users can enter stock symbols, select time periods, and get predictions with visualizations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.stock_data = None
        self.model = None
        self.prediction_results = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="📈 Stock Price Prediction Tool", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Create main frame with left and right panels
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel for inputs and controls
        self.setup_left_panel(main_frame)
        
        # Right panel for results and visualization
        self.setup_right_panel(main_frame)
        
    def setup_left_panel(self, parent):
        """Setup the left panel with input controls"""
        left_frame = tk.Frame(parent, bg='#f0f0f0', width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Input section
        input_frame = tk.LabelFrame(
            left_frame, 
            text="📊 Stock Selection", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        input_frame.pack(fill=tk.X, pady=5)
        
        # Stock symbol input
        tk.Label(input_frame, text="Stock Symbol:", bg='#f0f0f0').pack(anchor=tk.W, padx=5, pady=2)
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_frame = tk.Frame(input_frame, bg='#f0f0f0')
        symbol_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.symbol_entry = tk.Entry(
            symbol_frame, 
            textvariable=self.symbol_var,
            font=("Arial", 11),
            width=10
        )
        self.symbol_entry.pack(side=tk.LEFT)
        
        # Popular stocks buttons
        popular_frame = tk.Frame(input_frame, bg='#f0f0f0')
        popular_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(popular_frame, text="Quick Select:", bg='#f0f0f0', font=("Arial", 9)).pack(anchor=tk.W)
        
        button_frame = tk.Frame(popular_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        for i, stock in enumerate(popular_stocks):
            btn = tk.Button(
                button_frame,
                text=stock,
                command=lambda s=stock: self.symbol_var.set(s),
                font=("Arial", 8),
                width=6,
                bg='#3498db',
                fg='white',
                relief=tk.FLAT
            )
            btn.pack(side=tk.LEFT, padx=1, pady=1)
            if i == 2:  # Break to next row after 3 buttons
                button_frame = tk.Frame(popular_frame, bg='#f0f0f0')
                button_frame.pack(fill=tk.X, pady=1)
        
        # Time period selection
        tk.Label(input_frame, text="Time Period:", bg='#f0f0f0').pack(anchor=tk.W, padx=5, pady=(10,2))
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(
            input_frame,
            textvariable=self.period_var,
            values=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            state="readonly",
            width=15
        )
        period_combo.pack(anchor=tk.W, padx=5, pady=2)
        
        # Action buttons
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.predict_btn = tk.Button(
            button_frame,
            text="🔮 Get Prediction",
            command=self.start_prediction,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            relief=tk.FLAT,
            height=2
        )
        self.predict_btn.pack(fill=tk.X, pady=2)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear Results",
            command=self.clear_results,
            font=("Arial", 10),
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT
        )
        self.clear_btn.pack(fill=tk.X, pady=2)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            input_frame,
            mode='indeterminate'
        )
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(
            left_frame,
            height=15,
            width=45,
            font=("Consolas", 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='white'
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add initial message
        self.log_message("Welcome to Stock Prediction Tool!")
        self.log_message("Enter a stock symbol and click 'Get Prediction' to start.")
        self.log_message("=" * 50)
        
    def setup_right_panel(self, parent):
        """Setup the right panel for results and visualization"""
        right_frame = tk.Frame(parent, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = tk.LabelFrame(
            right_frame,
            text="📈 Prediction Results",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        results_frame.pack(fill=tk.X, pady=5)
        
        # Create results display
        self.results_frame = results_frame
        self.setup_results_display()
        
        # Visualization section
        viz_frame = tk.LabelFrame(
            right_frame,
            text="📊 Visualization",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial empty plots
        self.clear_plots()
        
    def setup_results_display(self):
        """Setup the results display area"""
        # Clear existing widgets
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create grid layout for results
        self.results_labels = {}
        
        # Row 0: Stock Info
        tk.Label(self.results_frame, text="Stock Symbol:", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.results_labels['symbol'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10))
        self.results_labels['symbol'].grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.results_frame, text="Current Price:", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.results_labels['current_price'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10))
        self.results_labels['current_price'].grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Row 1: Prediction
        tk.Label(self.results_frame, text="Predicted Price:", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.results_labels['predicted_price'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10), fg='#e74c3c')
        self.results_labels['predicted_price'].grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.results_frame, text="Predicted Change:", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.results_labels['predicted_change'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10))
        self.results_labels['predicted_change'].grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Row 2: Accuracy metrics
        tk.Label(self.results_frame, text="Model Accuracy (MAE):", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.results_labels['mae'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10))
        self.results_labels['mae'].grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.results_frame, text="Data Points:", bg='#f0f0f0', font=("Arial", 10, "bold")).grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.results_labels['data_points'] = tk.Label(self.results_frame, text="--", bg='#f0f0f0', font=("Arial", 10))
        self.results_labels['data_points'].grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
    def log_message(self, message):
        """Add a message to the status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_prediction(self):
        """Start the prediction process in a separate thread"""
        if not self.symbol_var.get().strip():
            messagebox.showerror("Error", "Please enter a stock symbol!")
            return
            
        # Disable button and start progress
        self.predict_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # Start prediction in separate thread
        thread = threading.Thread(target=self.run_prediction)
        thread.daemon = True
        thread.start()
        
    def run_prediction(self):
        """Run the prediction process"""
        try:
            symbol = self.symbol_var.get().strip().upper()
            period = self.period_var.get()
            
            self.log_message(f"Starting prediction for {symbol}...")
            
            # Download stock data
            self.log_message("Downloading stock data...")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) < 30:
                raise ValueError(f"Not enough data for {symbol}. Need at least 30 days.")
            
            self.log_message(f"Downloaded {len(data)} days of data")
            self.log_message(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            # Prepare data
            self.log_message("Preparing data for machine learning...")
            df = data.copy()
            df['Previous_Close'] = df['Close'].shift(1)
            df['Target'] = df['Close']
            df = df.dropna()
            
            X = df[['Previous_Close']].values
            y = df['Target'].values
            
            self.log_message(f"Prepared {len(X)} samples for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            self.log_message(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            
            # Train model
            self.log_message("Training linear regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            self.log_message(f"Model coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
            
            # Evaluate model
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae_percentage = (mae/np.mean(y_test))*100
            
            self.log_message(f"Model Performance:")
            self.log_message(f"  MAE: ${mae:.2f} ({mae_percentage:.2f}%)")
            self.log_message(f"  RMSE: ${rmse:.2f}")
            
            # Make next day prediction
            last_close = data['Close'].iloc[-1]
            next_day_pred = model.predict([[last_close]])[0]
            change = next_day_pred - last_close
            change_pct = (change / last_close) * 100
            
            self.log_message(f"Prediction:")
            self.log_message(f"  Current Price: ${last_close:.2f}")
            self.log_message(f"  Predicted Next: ${next_day_pred:.2f}")
            self.log_message(f"  Change: ${change:.2f} ({change_pct:.2f}%)")
            
            # Store results
            self.prediction_results = {
                'symbol': symbol,
                'current_price': last_close,
                'predicted_price': next_day_pred,
                'change': change,
                'change_pct': change_pct,
                'mae': mae,
                'mae_pct': mae_percentage,
                'rmse': rmse,
                'data_points': len(data),
                'y_test': y_test,
                'predictions': predictions,
                'data': data
            }
            
            # Update GUI on main thread
            self.root.after(0, self.update_results)
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.finish_prediction)
            
    def finish_prediction(self):
        """Finish the prediction process"""
        self.progress.stop()
        self.predict_btn.config(state=tk.NORMAL)
        
    def update_results(self):
        """Update the results display"""
        if not self.prediction_results:
            return
            
        r = self.prediction_results
        
        # Update labels
        self.results_labels['symbol'].config(text=r['symbol'])
        self.results_labels['current_price'].config(text=f"${r['current_price']:.2f}")
        self.results_labels['predicted_price'].config(text=f"${r['predicted_price']:.2f}")
        
        # Color code the change
        change_text = f"${r['change']:.2f} ({r['change_pct']:.2f}%)"
        change_color = '#27ae60' if r['change'] >= 0 else '#e74c3c'
        self.results_labels['predicted_change'].config(text=change_text, fg=change_color)
        
        self.results_labels['mae'].config(text=f"${r['mae']:.2f} ({r['mae_pct']:.2f}%)")
        self.results_labels['data_points'].config(text=str(r['data_points']))
        
        # Update plots
        self.update_plots()
        
    def update_plots(self):
        """Update the visualization plots"""
        if not self.prediction_results:
            return
            
        r = self.prediction_results
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Time series
        self.ax1.plot(range(len(r['y_test'])), r['y_test'], 
                     label='Actual Prices', color='#3498db', linewidth=2, alpha=0.8)
        self.ax1.plot(range(len(r['predictions'])), r['predictions'], 
                     label='Predicted Prices', color='#e74c3c', linewidth=2, alpha=0.8)
        self.ax1.set_title(f'{r["symbol"]} - Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Days')
        self.ax1.set_ylabel('Price ($)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price history
        dates = r['data'].index[-50:]  # Last 50 days
        prices = r['data']['Close'][-50:]
        
        self.ax2.plot(dates, prices, color='#2c3e50', linewidth=2)
        self.ax2.axhline(y=r['current_price'], color='#3498db', linestyle='--', 
                        label=f'Current: ${r["current_price"]:.2f}')
        self.ax2.axhline(y=r['predicted_price'], color='#e74c3c', linestyle='--', 
                        label=f'Predicted: ${r["predicted_price"]:.2f}')
        self.ax2.set_title(f'{r["symbol"]} - Recent Price History (50 days)', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Date')
        self.ax2.set_ylabel('Price ($)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)
        
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
        
    def clear_plots(self):
        """Clear the visualization plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.text(0.5, 0.5, 'No data to display\nRun a prediction to see charts', 
                     ha='center', va='center', transform=self.ax1.transAxes,
                     fontsize=12, color='gray')
        self.ax1.set_title('Stock Price Prediction Chart')
        
        self.ax2.text(0.5, 0.5, 'No data to display\nRun a prediction to see price history', 
                     ha='center', va='center', transform=self.ax2.transAxes,
                     fontsize=12, color='gray')
        self.ax2.set_title('Price History Chart')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def clear_results(self):
        """Clear all results and reset the interface"""
        self.prediction_results = None
        self.setup_results_display()
        self.clear_plots()
        self.status_text.delete(1.0, tk.END)
        self.log_message("Results cleared. Ready for new prediction.")
        self.log_message("=" * 50)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = StockPredictionGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")
    
    # Add disclaimer dialog
    messagebox.showinfo(
        "Disclaimer", 
        "⚠️ IMPORTANT DISCLAIMER ⚠️\n\n"
        "This tool is for educational purposes only.\n"
        "DO NOT use predictions for actual trading decisions.\n"
        "Stock markets involve significant risks.\n"
        "Past performance does not guarantee future results.\n\n"
        "Click OK to continue..."
    )
    
    root.mainloop()

if __name__ == "__main__":
    main()