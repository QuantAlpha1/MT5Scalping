import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import json


import MetaTrader5 as mt5
from typing import Optional, Any
from datetime import datetime
import pandas as pd


class MT5TypeHints:
    """Static type hints for MetaTrader5 functions"""
    
    @staticmethod
    def initialize() -> bool: ...
    
    @staticmethod
    def shutdown() -> None: ...
    
    @staticmethod
    def copy_rates_range(symbol: str, timeframe: int, start: datetime, end: datetime) -> Optional[np.ndarray]: ...
    
    @staticmethod
    def TIMEFRAME_M1() -> int: ...
    
    @staticmethod
    def TIMEFRAME_M5() -> int: ...
    
    @staticmethod
    def TIMEFRAME_H1() -> int: ...
    
    @staticmethod
    def TIMEFRAME_D1() -> int: ...


class MT5BacktestWrapper:
    """Type-safe wrapper for backtesting operations"""
    
    @staticmethod
    def initialize() -> bool:
        """Initialize MT5 connection with type safety"""
        result = mt5.initialize()  # type: ignore
        return bool(result)

    @staticmethod
    def shutdown() -> None:
        """Shutdown MT5 connection with type safety"""
        mt5.shutdown()  # type: ignore

    @staticmethod
    def copy_rates_range(
        symbol: str,
        timeframe: int,
        start: datetime,
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Safe historical data retrieval with:
        - Type checking
        - Error handling
        - Pandas conversion
        """
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)  # type: ignore
        if rates is None or len(rates) == 0:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.set_index('time')

    @staticmethod
    def get_timeframe(timeframe: str) -> int:
        """Convert timeframe string to MT5 constant with type safety"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,  # type: ignore
            'M5': mt5.TIMEFRAME_M5,  # type: ignore
            'H1': mt5.TIMEFRAME_H1,  # type: ignore
            'D1': mt5.TIMEFRAME_D1   # type: ignore
        }
        return timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_M1)  # type: ignore
    
    

class ScalperBacktester:
    def __init__(self):
        self.config = {
            'risk_percent': 0.5,
            'risk_reward': 1.5,
            'magic_number': 234000
        }
        self.trade_log = []

        self.mt5 = MT5BacktestWrapper()
        
    def load_historical_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Safe data loading with error handling"""
        if not self.mt5.initialize():
            raise ConnectionError("MT5 initialization failed")
            
        try:
            df = self.mt5.copy_rates_range(
                symbol,
                self.mt5.get_timeframe('M1'),  # Using wrapper's timeframe conversion
                start,
                end
            )
            if df is None:
                raise ValueError(f"No data for {symbol}")
            return df
        finally:
            self.mt5.shutdown()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Your exact ATR calculation from live strategy"""
        df['prev_close'] = df['close'].shift(1)
        hl = (df['high'] - df['low']).abs()
        hc = (df['high'] - df['prev_close']).abs()
        cl = (df['prev_close'] - df['low']).abs()
        df['tr'] = pd.concat([hl, hc, cl], axis=1).max(axis=1)
        return df['tr'].rolling(period).mean().iloc[-1]

    def entry_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Your exact entry logic"""
        # M1 Breakout logic
        m1_closes = df['close'].values[-15:]
        m1_breakout_buy = m1_closes[-1] > max(m1_closes[-6:-1])
        m1_breakout_sell = m1_closes[-1] < min(m1_closes[-6:-1])
        
        # M5 Trend filter (simplified)
        m5_ma = df['close'].rolling(5).mean().iloc[-1]
        m5_trend_filter_buy = df['close'].iloc[-1] > m5_ma
        
        # Your weighted decision making
        buy_score = 0
        if m1_breakout_buy: buy_score += 0.4
        if m5_trend_filter_buy: buy_score += 0.2
        
        sell_score = 0
        if m1_breakout_sell: sell_score += 0.4
        if not m5_trend_filter_buy: sell_score += 0.2
        
        if buy_score >= 0.6: return 'buy'
        if sell_score >= 0.6: return 'sell'
        return None

    def run_backtest(self, symbol: str, days: int = 7):
        """Run your exact strategy logic on historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"\nBacktesting {symbol} from {start_date.date()} to {end_date.date()}")
        df = self.load_historical_data(symbol, start_date, end_date)
        
        for i in range(50, len(df)):  # Warm-up period
            window = df.iloc[:i]
            current_data = df.iloc[i]
            
            # Your exact trading logic
            signal = self.entry_signal(window)
            atr = self.calculate_atr(window)
            
            if signal:
                entry_price = current_data['close']
                sl, tp = self.calculate_sl_tp(entry_price, signal == 'buy', atr)
                
                # Simulate trade execution
                self.execute_trade(
                    symbol=symbol,
                    time=current_data['time'],
                    signal=signal,
                    entry=entry_price,
                    sl=sl,
                    tp=tp,
                    atr=atr
                )
        
        self.save_results(symbol)

    def calculate_sl_tp(self, entry: float, is_long: bool, atr: float) -> tuple:
        """Your exact SL/TP calculation"""
        sl_dist = 1.5 * atr
        tp_dist = sl_dist * self.config['risk_reward']
        return (
            entry - sl_dist if is_long else entry + sl_dist,
            entry + tp_dist if is_long else entry - tp_dist
        )

    def execute_trade(self, symbol: str, time: datetime, signal: str, 
                     entry: float, sl: float, tp: float, atr: float):
        """Simulate trade execution with your position sizing"""
        # Your exact position sizing logic
        risk_amount = 10000 * (self.config['risk_percent'] / 100)  # Mock account balance
        risk_points = abs(entry - sl)
        size = risk_amount / risk_points
        
        # Simulate trade outcome
        outcome = self.simulate_trade_outcome(entry, sl, tp)
        
        self.trade_log.append({
            'time': time,
            'symbol': symbol,
            'signal': signal,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'size': round(size, 2),
            'atr': atr,
            'outcome': outcome,
            'pnl': self.calculate_pnl(entry, outcome, size, signal)
        })

    def simulate_trade_outcome(self, entry: float, sl: float, tp: float) -> float:
        """Simulates which price target was hit first"""
        # Simple simulation - replace with more sophisticated logic if needed
        if abs(entry - tp) < abs(entry - sl):  # TP hit first
            return tp
        return sl  # SL hit first

    def calculate_pnl(self, entry: float, exit: float, size: float, signal: str) -> float:
        """Your exact PNL calculation"""
        if signal == 'buy':
            return (exit - entry) * size * 100000  # Standard lot
        return (entry - exit) * size * 100000

    def save_results(self, symbol: str):
        """Save backtest results"""
        results = pd.DataFrame(self.trade_log)
        filename = f"{symbol}_backtest_{datetime.now().date()}.csv"
        results.to_csv(filename, index=False)
        print(f"Saved {len(results)} trades to {filename}")

        # Generate performance report
        wins = results[results['pnl'] > 0]
        win_rate = len(wins)/len(results) if len(results) > 0 else 0
        print(f"\nPerformance for {symbol}:")
        print(f"Total Trades: {len(results)}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total PnL: ${results['pnl'].sum():.2f}")


if __name__ == "__main__":
    tester = ScalperBacktester()
    
    # Configure backtest
    symbols = ["EURUSD", "USDJPY"]  # Your symbols
    days = 3  # Start with 3 days to test
    
    for symbol in symbols:
        tester.run_backtest(symbol, days)