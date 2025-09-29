"""
Test script to verify GUI functionality without user interaction
"""

import tkinter as tk
from stock_prediction_gui import StockPredictionGUI
import sys

def test_gui():
    """Test basic GUI functionality"""
    try:
        print("Testing GUI components...")
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing
        
        # Test GUI creation
        app = StockPredictionGUI(root)
        print("✅ GUI created successfully")
        
        # Test variable setting
        app.symbol_var.set("AAPL")
        app.period_var.set("1mo")
        print("✅ Variables set successfully")
        
        # Test log message
        app.log_message("Test message")
        print("✅ Logging works")
        
        # Test results display setup
        app.setup_results_display()
        print("✅ Results display works")
        
        # Clean up
        root.destroy()
        print("✅ All GUI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gui()
    sys.exit(0 if success else 1)