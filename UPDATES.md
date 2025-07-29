Symbol-specific spread settings: 

{
  "symbol_settings": {
    "EURUSD": {
      "max_spread_points": 2.0,    // 2 pips = 20 points (0.00020)
      "spread_buffer": 1.3,
      "min_volatility": 0.0003,
      "pip_value": 0.0001
    },
    "USDJPY": {
      "max_spread_points": 3.0,    // 3 pips = 30 points (0.030)
      "spread_buffer": 1.5, 
      "min_volatility": 0.03,
      "pip_value": 0.01
    }
  },
  "trading": {
    "risk_percent": 0.5,
    "risk_reward": 1.5,
    "magic_number": 234000
  }
}

Required code changes:

    In place_trade():

python

def place_trade(self, symbol: str, signal: str):
    symbol_config = self.config['symbol_settings'][symbol]
    current_spread = (tick['ask'] - tick['bid']) / self.symbols[symbol]['point']
    
    if current_spread > (symbol_config['max_spread_points'] * symbol_config['spread_buffer']):
        print(f"{symbol} spread {current_spread:.1f} points exceeds limit of {symbol_config['max_spread_points']}")
        return False

    In calculate_sl_tp():

python

def calculate_sl_tp(self, symbol: str, entry: float, is_long: bool):
    pip_value = self.config['symbol_settings'][symbol]['pip_value']
    min_dist = 10 * pip_value  # 10 pips minimum distance
    # ... rest of calculation ...

Key improvements:

    Proper pip/point conversion for each symbol type

    USDJPY gets 50% wider allowed spread than EURUSD

    Clear separation of symbol-specific parameters

    Buffer multipliers can now vary by pair

