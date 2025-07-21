import MetaTrader5 as mt5
from datetime import datetime

def test_mt5_connection():
    # Set path to your MT5 terminal EXE
    mt5.initialize(
        path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        portable=False
    )
    
    if mt5.initialize():
        print("MT5 initialized successfully!")
        print(f"Terminal version: {mt5.version()}")
        
        # Test basic functionality
        symbol = "EURUSD"
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
        if rates is not None:
            print(f"Got {len(rates)} price bars")
        else:
            print("Failed to get rates")
        
        mt5.shutdown()
    else:
        print("MT5 initialization failed")
        print(f"Last error: {mt5.last_error()}")

if __name__ == "__main__":
    test_mt5_connection()