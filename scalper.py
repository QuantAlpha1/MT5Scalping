# .\venv\Scripts\activate

import argparse
import MetaTrader5 as mt5
import json
import pandas as pd
import time
from datetime import datetime, time as dt_time, timedelta, time as dt_time
import pytz
import numpy as np
from typing import Tuple, Optional, List, Any, Dict, Union, Callable
from dataclasses import dataclass
from colorama import Fore
from typing import TypedDict, NotRequired


class MT5TypeHints:
    """Type hints for MetaTrader5 functions to satisfy PyLance"""
    @staticmethod
    def initialize() -> bool: ...
    @staticmethod
    def shutdown() -> None: ...
    @staticmethod
    def terminal_info() -> Any: ...
    @staticmethod
    def account_info() -> Any: ...
    @staticmethod
    def symbol_info(symbol: str) -> Any: ...
    @staticmethod
    def symbol_info_tick(symbol: str) -> Any: ...
    @staticmethod
    def positions_get(symbol: str | None = None) -> Any: ...
    @staticmethod
    def orders_get(symbol: str | None = None) -> Any: ...
    @staticmethod
    def history_deals_get(date_from: int | datetime, date_to: int | datetime) -> Any: ...
    @staticmethod
    def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int) -> Any: ...
    @staticmethod
    def order_send(request: dict) -> Any: ...
    @staticmethod
    def order_calc_margin(action: int, symbol: str, volume: float, price: float) -> float: ...
    @staticmethod
    def market_book_get(symbol: str) -> Any: ...


@dataclass
class MarketDepthLevel:
    price: float
    volume: float


@dataclass
class SymbolInfo:
    point: float
    digits: int
    volume_step: float
    volume_min: float
    volume_max: float
    trade_allowed: bool
    pip_value: float


class MT5Wrapper:
    """Complete MT5 wrapper with type safety, error handling, and timeframe constants"""
    
    # Timeframe constants
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440
    TIMEFRAME_W1 = 10080
    TIMEFRAME_MN1 = 43200
    
    # Timeframe mapping for conversion
    TIMEFRAMES = {
        'M1': TIMEFRAME_M1,
        'M5': TIMEFRAME_M5,
        'M15': TIMEFRAME_M15,
        'M30': TIMEFRAME_M30,
        'H1': TIMEFRAME_H1,
        'H4': TIMEFRAME_H4,
        'D1': TIMEFRAME_D1,
        'W1': TIMEFRAME_W1,
        'MN1': TIMEFRAME_MN1
    }

    # Order execution constants
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    
    # Order filling policies
    ORDER_FILLING_FOK = 0  # Fill Or Kill
    ORDER_FILLING_IOC = 1  # Immediate Or Cancel
    ORDER_FILLING_RETURN = 2
    
    # Order lifetime policies
    ORDER_TIME_GTC = 0  # Good Till Cancel
    ORDER_TIME_DAY = 1
    ORDER_TIME_SPECIFIED = 2
    ORDER_TIME_SPECIFIED_DAY = 3
    
    # Trade actions
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_REMOVE = 8
    TRADE_ACTION_CLOSE_BY = 10
    
    # Trade return codes
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_CANCEL = 10007
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_REJECTED = 10016
    TRADE_RETCODE_LONG_ONLY = 10017
    TRADE_RETCODE_SHORT_ONLY = 10018
    TRADE_RETCODE_CLOSE_ONLY = 10019

    # Deal entry types
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3


    @staticmethod
    def get_timeframe(timeframe: Union[str, int]) -> int:
        """
        Convert timeframe string to MT5 constant or validate integer timeframes.
        
        Args:
            timeframe: Either string (e.g., 'M15') or integer (e.g., 15)
            
        Returns:
            MT5 timeframe constant
            
        Raises:
            ValueError: If timeframe is invalid
        """
        if isinstance(timeframe, str):
            tf = timeframe.upper()
            if tf not in MT5Wrapper.TIMEFRAMES:
                raise ValueError(f"Invalid timeframe string: {timeframe}")
            return MT5Wrapper.TIMEFRAMES[tf]
        elif isinstance(timeframe, int):
            if timeframe not in MT5Wrapper.TIMEFRAMES.values():
                raise ValueError(f"Invalid timeframe value: {timeframe}")
            return timeframe
        raise ValueError("Timeframe must be string or integer")

    @staticmethod
    def validate_order_request(request: Dict[str, Any]) -> bool:
        """
        Validate an order request dictionary contains all required fields.
        
        Args:
            request: Order request dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = [
            'action', 'symbol', 'volume', 'type', 'price',
            'deviation', 'magic', 'comment', 'type_time', 'type_filling'
        ]
        return all(field in request for field in required_fields)
    
    @staticmethod
    def _call_mt5(func_name: str, *args, **kwargs) -> Any:
        """Safe wrapper for MT5 functions with error handling"""
        try:
            if not hasattr(mt5, func_name):
                raise AttributeError(f"MT5 missing function: {func_name}")
            
            func = getattr(mt5, func_name)
            result = func(*args, **kwargs)
            
            if result is None or (isinstance)(result, (tuple, list)) and not result:
                return None
            return result
        except Exception as e:
            print(f"MT5 {func_name} error: {str(e)}")
            return None
    
    # Connection Management
    @staticmethod
    def initialize() -> bool:
        """Initialize connection with retry logic"""
        for _ in range(3):
            result = MT5Wrapper._call_mt5('initialize')
            if result:
                return True
            time.sleep(1)
        return False

    @staticmethod
    def shutdown() -> None:
        """Shutdown connection"""
        MT5Wrapper._call_mt5('shutdown')
    
    @staticmethod
    def symbol_select(symbol: str, enable: bool) -> bool:
        """
        Select or deselect a symbol in the Market Watch.
        
        Args:
            symbol: The symbol to select/deselect (e.g., 'EURUSD')
            enable: True to select, False to deselect
            
        Returns:
            bool: True if successful, False otherwise
        """
        result = MT5Wrapper._call_mt5('symbol_select', symbol, enable)
        return bool(result)

    @staticmethod
    def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
        """
        Get comprehensive information about a trading symbol.
        
        Args:
            symbol: The financial instrument (e.g., 'EURUSD')
            
        Returns:
            SymbolInfo: Dataclass with symbol properties or None if failed
        """
        info = MT5Wrapper._call_mt5('symbol_info', symbol)
        if not info:
            return None
        
        try:
            return SymbolInfo(
                point=float(info.point),
                digits=int(info.digits),
                volume_step=float(info.volume_step),
                volume_min=float(info.volume_min),
                volume_max=float(info.volume_max),
                trade_allowed=bool(info.trade_allowed),
                pip_value=10.0  # Default, will be adjusted for JPY pairs
            )
        except AttributeError as e:
            print(f"Symbol info missing expected attribute: {e}")
            return None
        except Exception as e:
            print(f"Error processing symbol info: {e}")
            return None
    
    @staticmethod 
    def get_symbol_tick(symbol: str) -> dict | None:
        """Get tick data with proper typing"""
        tick = MT5Wrapper._call_mt5('symbol_info_tick', symbol)
        if not tick:
            return None
            
        return {
            'bid': float(tick.bid),
            'ask': float(tick.ask),
            'last': float(tick.last),
            'volume': float(tick.volume),
            'time': int(tick.time)
        }
    
    # Account Operations
    @staticmethod
    def get_account_info() -> Optional[Dict[str, Any]]:
        """
        Retrieve complete account information in dictionary format.
        
        Returns:
            Dictionary containing all account properties or None if failed.
            Example keys: 'login', 'balance', 'equity', 'margin', etc.
        """
        account = MT5Wrapper._call_mt5('account_info')
        
        if not account:
            print("Failed to retrieve account information")
            return None
        
        try:
            account_dict = account._asdict()
            
            # Convert numeric fields to proper Python types
            numeric_fields = ['balance', 'equity', 'margin', 'margin_free', 'margin_level']
            for field in numeric_fields:
                if field in account_dict:
                    account_dict[field] = float(account_dict[field])
                    
            return account_dict
        except AttributeError as e:
            print(f"Account info missing expected attributes: {e}")
            return None
        except Exception as e:
            print(f"Error processing account info: {e}")
            return None
    
    @staticmethod
    def terminal_info() -> dict | None:
        """Get terminal info as dict"""
        info = MT5Wrapper._call_mt5('terminal_info')
        return info._asdict() if info else None
    
    # Market Data
    @staticmethod
    def get_rates(symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[List[Dict[str, float]]]:
        """
        Retrieve historical price data (OHLCV) from a starting position.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15)
            start_pos: Starting bar position (0 = current bar)
            count: Number of bars to retrieve
            
        Returns:
            List of dictionaries with OHLCV data or None if failed.
            Each dictionary contains: time, open, high, low, close, real_volume
        """
        rates = MT5Wrapper._call_mt5('copy_rates_from_pos', symbol, timeframe, start_pos, count)
        
        if rates is None or len(rates) == 0:
            print(f"No rate data returned for {symbol} (TF:{timeframe})")
            return None
        
        try:
            return [{
                'time': float(rate.time),
                'open': float(rate.open),
                'high': float(rate.high),
                'low': float(rate.low),
                'close': float(rate.close),
                'real_volume': float(rate.real_volume),
                'tick_volume': float(rate.tick_volume)  # Added for completeness
            } for rate in rates]
        except (AttributeError, TypeError) as e:
            print(f"Rate data format error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing rates for {symbol}: {e}")
            return None
    
    # Trade Operations
    @staticmethod
    def get_positions(symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve open positions with optional filters.
        
        Args:
            symbol: Optional symbol filter (e.g., 'EURUSD')
            magic: Optional magic number filter
            
        Returns:
            List of position dictionaries, each containing:
            - ticket: int
            - symbol: str
            - type: str
            - volume: float
            - price_open: float
            - sl: float
            - tp: float 
            - magic: int
            Returns empty list on error or if no positions match filters
        """
        try:
            # Get positions using safe wrapper
            positions = MT5Wrapper._call_mt5('positions_get', symbol=symbol) if symbol else MT5Wrapper._call_mt5('positions_get')
            
            if not positions:
                return []

            position_dicts = []
            for p in positions:
                try:
                    pos_dict = {
                        'ticket': int(p.ticket),
                        'symbol': str(p.symbol),
                        'type': str(p.type),
                        'volume': float(p.volume),
                        'price_open': float(p.price_open),
                        'sl': float(p.sl),
                        'tp': float(p.tp),
                        'magic': int(p.magic),
                        'time': int(p.time),
                        'time_update': int(p.time_update),
                        'time_msc': int(p.time_msc),
                        'price_current': float(p.price_current),
                        'swap': float(p.swap),
                        'profit': float(p.profit),
                        'comment': str(p.comment),
                        'identifier': int(p.identifier)
                    }
                    
                    # Apply magic filter if specified
                    if magic is None or pos_dict['magic'] == magic:
                        position_dicts.append(pos_dict)
                        
                except (AttributeError, ValueError) as e:
                    print(f"Skipping malformed position: {e}")
                    continue
                    
            return position_dicts
            
        except Exception as e:
            print(f"Failed to retrieve positions: {e}")
            return []
    
    @staticmethod
    def calculate_margin(order_type: int, symbol: str, volume: float, price: float) -> Optional[float]:
        """
        Calculate required margin for a potential trade.
        
        Args:
            order_type: Trade operation type (ORDER_TYPE_BUY/ORDER_TYPE_SELL)
            symbol: Trading symbol (e.g., 'EURUSD')
            volume: Trade volume in lots
            price: Entry price
            
        Returns:
            Required margin in account currency (float) or None if calculation failed
        """
        try:
            margin = MT5Wrapper._call_mt5('order_calc_margin', order_type, symbol, volume, price)
            if margin is None:
                print(f"Margin calculation returned None for {symbol} {volume} lots @ {price}")
                return None
                
            calculated_margin = float(margin)
            
            # Validate the result makes sense
            if calculated_margin < 0:
                print(f"Invalid negative margin: {calculated_margin}")
                return None
                
            return calculated_margin
            
        except ValueError as e:
            print(f"Invalid margin calculation result: {e}")
            return None
        except Exception as e:
            print(f"Margin calculation error for {symbol}: {e}")
            return None
    
    @staticmethod
    def send_order(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send trade order to the server with enhanced validation and error handling.
        
        Args:
            request: Dictionary containing order parameters including:
                - action: TRADE_ACTION_DEAL/TRADE_ACTION_PENDING
                - symbol: Trading symbol
                - volume: Order volume
                - type: ORDER_TYPE_BUY/ORDER_TYPE_SELL
                - price: Execution price
                - sl: Stop loss price
                - tp: Take profit price
                - deviation: Max price deviation
                - magic: Expert ID
                - comment: Order comment
                - type_time: ORDER_TIME_GTC/etc
                - type_filling: ORDER_FILLING_FOK/etc
                
        Returns:
            Dictionary with order result containing:
                - retcode: int (MT5 return code)
                - order: int (order ticket)
                - volume: float (executed volume)
                - price: float (execution price) 
                - comment: str (server message)
                - request_id: int (request identifier)
                - retcode_external: int (additional error code)
                - deal: int (deal ticket if available)
            Returns None if order failed
        """
        try:
            # Validate required fields
            required_fields = ['action', 'symbol', 'volume', 'type', 'price']
            for field in required_fields:
                if field not in request:
                    raise ValueError(f"Missing required field: {field}")

            result = MT5Wrapper._call_mt5('order_send', request)
            if not result:
                print("Order failed - no result returned")
                return None

            # Convert result to dictionary with additional fields
            order_result = {
                'retcode': int(result.retcode),
                'order': int(result.order),
                'volume': float(result.volume),
                'price': float(result.price),
                'comment': str(result.comment),
                'request_id': int(result.request_id),
                'retcode_external': int(result.retcode_external),
                'deal': int(result.deal) if hasattr(result, 'deal') else 0
            }

            # Log order result for debugging
            if order_result['retcode'] != MT5Wrapper.TRADE_RETCODE_DONE:
                error_msg = MT5Wrapper._get_retcode_message(order_result['retcode'])
                print(f"Order failed with retcode {order_result['retcode']}: {error_msg}")
                print(f"Server message: {order_result['comment']}")

            return order_result

        except ValueError as e:
            print(f"Invalid order request: {e}")
            return None
        except AttributeError as e:
            print(f"Malformed order result: {e}")
            return None
        except Exception as e:
            print(f"Unexpected order error: {e}")
            return None

    @staticmethod
    def _get_retcode_message(retcode: int) -> str:
        """Helper method to get human-readable error messages"""
        messages = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Partial fill",
            10011: "Request processing error",
            10012: "Request timeout",
            10013: "Invalid request",
            10014: "Invalid volume",
            10015: "Invalid price",
            10016: "Request rejected by server",
            10017: "Long positions only allowed",
            10018: "Short positions only allowed",
            10019: "Close-only orders allowed"
        }
        return messages.get(retcode, f"Unknown error code: {retcode}")
    
    
    @staticmethod
    def get_market_depth(symbol: str) -> Optional[Dict[str, Union[List[Tuple[float, float]], str, int]]]:
        """
        Retrieve complete market depth (order book) data with enhanced validation.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dictionary containing:
            - 'bid': List of (price, volume) tuples for buy orders (sorted highest to lowest)
            - 'ask': List of (price, volume) tuples for sell orders (sorted lowest to highest)
            - 'symbol': str (the requested symbol)
            - 'timestamp': int (milliseconds since epoch)
            Returns None if market depth is unavailable or error occurs
        """
        try:
            # Check for market depth support
            if not hasattr(mt5, 'market_book_get'):
                print(f"[Warning] Market depth not supported for {symbol} - requires MT5 build 2250+")
                return None
                
            # Get market depth data
            depth = MT5Wrapper._call_mt5('market_book_get', symbol)
            if not depth or not hasattr(depth, 'bid') or not hasattr(depth, 'ask'):
                print(f"No market depth data available for {symbol}")
                return None
                
            # Process bids (sort from highest to lowest)
            bids = sorted(
                [(float(level.price), float(level.volume)) for level in depth.bid],
                key=lambda x: -x[0]  # Sort by price descending
            )
            # Add this after checking hasattr(depth, 'bid')
            if not isinstance(depth.bid, (list, tuple)) or not isinstance(depth.ask, (list, tuple)):
                print(f"Invalid market depth structure for {symbol}")
                return None
            
            # Process asks (sort from lowest to highest)
            asks = sorted(
                [(float(level.price), float(level.volume)) for level in depth.ask],
                key=lambda x: x[0]  # Sort by price ascending
            )
            
            # Validate there's at least one level on each side
            if not bids or not asks:
                print(f"Incomplete market depth for {symbol} - missing one side")
                return None
                
            return {
                'bid': bids,
                'ask': asks,
                'symbol': symbol,
                'timestamp': int(time.time() * 1000)  # Milliseconds precision
            }
            
        except AttributeError as e:
            print(f"Market depth structure error for {symbol}: {e}")
            return None
        except ValueError as e:
            print(f"Market depth data conversion error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting market depth for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_history_deals(date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve historical deal records with comprehensive trade information.
        
        Args:
            date_from: Start datetime (inclusive)
            date_to: End datetime (inclusive)
            
        Returns:
            List of dictionaries containing complete deal information:
            - ticket: int (deal ticket number)
            - position: int (position ID)
            - time: datetime (execution time)
            - symbol: str (trading symbol)
            - type: str (deal type as string)
            - entry: str (entry type as string)
            - volume: float (trade volume in lots)
            - price: float (execution price)
            - profit: float (profit in account currency)
            - commission: float (commission charged)
            - swap: float (swap amount)
            - fee: float (fee amount)
            - comment: str (deal comment)
            - magic: int (expert magic number)
            - reason: int (deal reason code)
            Returns empty list on error or if no deals found
        """
        try:
            # Validate date range
            if date_from > date_to:
                print(f"Invalid date range: {date_from} to {date_to}")
                return []
                
            # Get deals through safe wrapper
            deals = MT5Wrapper._call_mt5('history_deals_get', date_from, date_to)
            if not deals:
                return []
                
            processed_deals = []
            for deal in deals:
                try:
                    deal_dict = {
                        'ticket': int(deal.ticket),
                        'position': int(deal.position_id),
                        'time': datetime.fromtimestamp(deal.time),
                        'symbol': str(deal.symbol),
                        'type': str(deal.type),
                        'entry': str(deal.entry),
                        'volume': float(deal.volume),
                        'price': float(deal.price),
                        'profit': float(deal.profit),
                        'commission': float(deal.commission),
                        'swap': float(deal.swap),
                        'fee': float(deal.fee),
                        'comment': str(deal.comment),
                        'magic': int(deal.magic),
                        'reason': int(deal.reason),
                        'time_msc': datetime.fromtimestamp(deal.time_msc // 1000) if hasattr(deal, 'time_msc') else None,
                        'external_id': str(deal.external_id) if hasattr(deal, 'external_id') else None
                    }
                    processed_deals.append(deal_dict)
                    
                except (AttributeError, ValueError) as e:
                    print(f"Skipping malformed deal {getattr(deal, 'ticket', 'unknown')}: {e}")
                    continue
                    
            # Sort deals chronologically
            processed_deals.sort(key=lambda x: x['time'])
            
            return processed_deals
            
        except Exception as e:
            print(f"Failed to retrieve historical deals: {e}")
            return []

    @staticmethod
    def get_orders(symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve pending orders with comprehensive filtering and order details.
        
        Args:
            symbol: Optional symbol filter (e.g., 'EURUSD')
            magic: Optional expert advisor magic number filter
            
        Returns:
            List of dictionaries containing complete order information:
            - ticket: int (order ticket number)
            - symbol: str (trading symbol)
            - type: str (order type as string)
            - volume: float (current remaining volume)
            - price_open: float (requested open price)
            - sl: float (stop loss price)
            - tp: float (take profit price)
            - price_current: float (current market price)
            - magic: int (expert magic number)
            - comment: str (order comment)
            - time_setup: datetime (order placement time)
            - time_expiration: datetime (order expiration time) or None
            - time_done: datetime (order execution/cancel time) or None
            - state: str (order state)
            - type_filling: str (execution policy)
            - type_time: str (order lifetime policy)
            Returns empty list on error or if no matching orders found
        """
        try:
            # Get orders through safe wrapper
            orders = MT5Wrapper._call_mt5('orders_get', symbol=symbol) if symbol else MT5Wrapper._call_mt5('orders_get')
            if not orders:
                return []

            order_dicts = []
            for o in orders:
                try:
                    order = {
                        'ticket': int(o.ticket),
                        'symbol': str(o.symbol),
                        'type': str(o.type),
                        'volume': float(o.volume_current),
                        'price_open': float(o.price_open),
                        'sl': float(o.sl),
                        'tp': float(o.tp),
                        'price_current': float(o.price_current),
                        'magic': int(o.magic),
                        'comment': str(o.comment),
                        'time_setup': datetime.fromtimestamp(o.time_setup),
                        'time_expiration': datetime.fromtimestamp(o.time_expiration) if o.time_expiration > 0 else None,
                        'time_done': datetime.fromtimestamp(o.time_done) if hasattr(o, 'time_done') and o.time_done > 0 else None,
                        'state': str(o.state) if hasattr(o, 'state') else None,
                        'type_filling': str(o.type_filling) if hasattr(o, 'type_filling') else None,
                        'type_time': str(o.type_time) if hasattr(o, 'type_time') else None,
                        'position_id': int(o.position_id) if hasattr(o, 'position_id') else None,
                        'position_by_id': int(o.position_by_id) if hasattr(o, 'position_by_id') else None
                    }

                    # Apply magic filter if specified
                    if magic is None or order['magic'] == magic:
                        order_dicts.append(order)

                except (AttributeError, ValueError) as e:
                    print(f"Skipping malformed order {getattr(o, 'ticket', 'unknown')}: {e}")
                    continue

            # Sort orders by setup time (oldest first)
            order_dicts.sort(key=lambda x: x['time_setup'])
            
            return order_dicts

        except Exception as e:
            print(f"Failed to retrieve orders: {e}")
            return []



class RiskBreach(Exception):
    """Custom exception for risk management violations"""
    def __init__(self, message="Risk threshold breached"):
        self.message = message
        super().__init__(self.message)



class RiskManager:
    """Handles all risk-related checks and operations"""
    def __init__(self, config: dict, pnl_calculator: Optional[Callable[[], float]] = None):
        self.config = config
        self._calculate_daily_pnl = pnl_calculator
        self.drawdown_start_balance = None
        
    
    def check_drawdown(self, current_equity: float) -> None:
        """Safe drawdown check with PNL awareness"""
        if self.drawdown_start_balance is None:
            self.drawdown_start_balance = current_equity
            return
            
        # Safe access to PNL calculator
        if self._calculate_daily_pnl is not None:
            try:
                daily_pnl = self._calculate_daily_pnl()  # Call the stored function
                if daily_pnl < -0.5 * (self.config['max_daily_drawdown']/100) * self.drawdown_start_balance:
                    print(f"PNL Warning: ${daily_pnl:.2f}")
            except Exception as e:
                print(f"PNL check failed: {str(e)}")

        drawdown_pct = (self.drawdown_start_balance - current_equity) / self.drawdown_start_balance * 100
        if drawdown_pct >= self.config['max_daily_drawdown']:
            raise RiskBreach(f"Drawdown limit: {drawdown_pct:.2f}%")
            
    def reset_daily_drawdown(self):
        """Reset at start of new trading day"""
        self.drawdown_start_balance = None
        print("Daily drawdown reset")

    def check_position_count(self, current_count: int) -> None:
        if current_count >= self.config.get('max_simultaneous_trades', 5):
            raise RiskBreach(f"Max positions reached: {current_count}")



class Scalper:
    """Production-Ready MT5 Scalper for EURUSD and USDJPY"""
    
    def __init__(self, risk_percent: float = 0.5, risk_reward: float = 1.5):
        self._verify_broker_conditions() 
        self.risk_manager = RiskManager(
            config=self.config,
            pnl_calculator=self.calculate_daily_pnl  # Pass the method directly
        )
        self.last_trade_time = {}
        self.config = {
            'risk_percent': risk_percent,
            'risk_reward': risk_reward,
            'max_daily_drawdown': 2.0,  # Percentage
            'drawdown_start_time': None,
            'magic_number': 234000,
            'deviation': 10,
            'max_spread': 2.0,
            'trading_hours': {
                'start': dt_time(8, 0, tzinfo=pytz.timezone('Europe/London')),
                'end': dt_time(17, 0, tzinfo=pytz.timezone('Europe/London'))
            },
            'max_trade_duration': 15,  # minutes
            'trail_start': 10,  # pips
            'trail_step': 5     # pips
        }

        self.config.update({
            'volatility_thresholds': {
                'EURUSD': {'min': 0.0003, 'max': 0.002},
                'USDJPY': {'min': 0.03, 'max': 0.20}
            },
            'trading_sessions': {
                'london_open': {'start': dt_time(7,55), 'end': dt_time(9,5)},
                'ny_close': {'start': dt_time(16,0), 'end': dt_time(17,5)}
            }
        })

        self.config['volatility_thresholds'].update({
            'EURUSD': {'off_peak_reduction': 0.65},  # 35% reduction
            'USDJPY': {'off_peak_reduction': 0.75}   # 25% reduction
        })

        self.min_acceptable_volatility = {
            'EURUSD': 0.0003,
            'USDJPY': 0.03  
        }
        self.max_acceptable_volatility = {
            'EURUSD': 0.002,
            'USDJPY': 0.20
        }


        self.trade_log = [] 
        
        if not MT5Wrapper.initialize():
            raise ConnectionError("Failed to initialize MT5")
        print("MT5 initialized successfully")
        
        # Preload symbol info
        self.symbols = self._setup_symbols(["EURUSD", "USDJPY"])

    def __del__(self):
        MT5Wrapper.shutdown()
        print("MT5 connection closed")

    def _setup_symbols(self, symbols: list) -> dict:
        """Preload symbol properties with enhanced validation and error handling"""
        symbol_info = {}
        
        for symbol in symbols:
            # Symbol selection with retry logic
            if not MT5Wrapper.symbol_select(symbol, True):
                print(f"Failed to select {symbol} - skipping")
                continue
                
            # Get symbol info with error handling
            try:
                info = MT5Wrapper.get_symbol_info(symbol)
                if info is None:
                    print(f"Could not get symbol info for {symbol} - skipping")
                    continue
                    
                # Basic symbol properties
                symbol_info[symbol] = {
                    'point': info.point,
                    'digits': info.digits,
                    'volume_step': info.volume_step,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'trade_allowed': info.trade_allowed,
                    'pip_value': 10  # Default for non-JPY pairs
                }
                
                # JPY-specific handling with tick validation
                if "JPY" in symbol:
                    tick = MT5Wrapper.get_symbol_tick(symbol)  # Changed from symbol_info_tick to get_symbol_tick
                    if tick and tick['bid'] > 0:  # Now accessing as dictionary
                        symbol_info[symbol]['pip_value'] = 1000 / tick['bid']
                    else:
                        print(f"Invalid tick data for {symbol}, using default pip value")
                        
                # Additional validation for critical values
                if symbol_info[symbol]['point'] <= 0:
                    print(f"Invalid point value for {symbol} - skipping")
                    del symbol_info[symbol]
                    continue
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
                
        return symbol_info

    def _is_optimal_trading_time(self):
        """Filter for high probability trading windows"""
        now = datetime.now(pytz.timezone('Europe/London'))
        current_time = now.time()
        current_hour = current_time.hour
        
        # London open (8-9am) and US open (2-3pm London time)
        optimal_windows = [
            (dt_time(8, 0), dt_time(9, 0)),   # London open
            (dt_time(14, 0), dt_time(15, 0)),  # NY open
            (dt_time(15, 30), dt_time(16, 30)) # NY mid-session
        ]
        
        return any(start <= current_time < end for start, end in optimal_windows)

    def _is_trading_hours(self) -> bool:
        """Robust trading hour check with timezone awareness"""
        london_tz = pytz.timezone('Europe/London')
        now = datetime.now(london_tz).time()
        return self.config['trading_hours']['start'] <= now <= self.config['trading_hours']['end']

    def _is_london_open(self) -> bool:
        """Check if current time is within London open window (8-9am London)"""
        london_tz = pytz.timezone('Europe/London')
        now = datetime.now(london_tz).time()
        session = self.config['trading_sessions']['london_open']
        return session['start'] <= now <= session['end']

    def _is_ny_close(self) -> bool:
        """Check if current time is within NY close window (4-5pm London/11am-12pm NY)"""
        london_tz = pytz.timezone('Europe/London') 
        now = datetime.now(london_tz).time()
        return dt_time(16, 0) <= now <= dt_time(17, 5)  # 5-min buffer

    def calculate_position_size(self, symbol: str, entry: float, stop_loss: float) -> float:
        """Enhanced risk-based sizing with volatility scaling"""
        account = MT5Wrapper.get_account_info()
        if not account:
            return 0.0
        
        # 1. Original risk calculation (unchanged)
        risk_amount = account.get('balance', 0.0) * (self.config['risk_percent'] / 100)
        point = self.symbols[symbol]['point']
        risk_points = abs(entry - stop_loss) / point
        
        # Currency-specific pip value (unchanged)
        if "JPY" in symbol:
            pip_value = 1000 / entry  # JPY pairs
        else:
            pip_value = 10  # Non-JPY pairs
        
        # 2. Base size calculation (unchanged)
        base_size = risk_amount / (risk_points * point * pip_value)
        
        # 3. ➕ NEW: Volatility Adjustment (Single Block)
        if hasattr(self, 'min_acceptable_volatility'):  # Safe check
            current_atr = self.calculate_atr(symbol)
            if current_atr > 0:
                # Dynamic scaling (0.8x to 1.5x multiplier)
                atr_ratio = current_atr / self.min_acceptable_volatility.get(symbol[:3], 0.0005)
                volatility_factor = min(1.5, max(0.8, atr_ratio))
                base_size *= volatility_factor
        
        return self._normalize_volume(symbol, base_size)

    def calculate_atr(self, symbol: str, timeframe: int = MT5Wrapper.TIMEFRAME_M5, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: MT5 timeframe constant (default: M5)
            period: ATR period (default: 14)
            
        Returns:
            float: ATR value or 0.0 if calculation fails
        """
        try:
            # Use the provided timeframe parameter instead of hardcoded value
            rates = MT5Wrapper.get_rates(symbol, timeframe, 0, period+1)
            if rates is None or len(rates) < period+1:
                print(f"Insufficient data for ATR calculation on {symbol} (TF: {timeframe})")
                return 0.0
            
            df = pd.DataFrame(rates)
            df['prev_close'] = df['close'].shift(1)
            
            # Calculate True Range components
            hl = (df['high'] - df['low']).abs()
            hc = (df['high'] - df['prev_close']).abs()
            cl = (df['prev_close'] - df['low']).abs()
            
            # Get max of the three components
            df['tr'] = pd.concat([hl, hc, cl], axis=1).max(axis='columns')
            
            return df['tr'].tail(period).mean()
            
        except Exception as e:
            print(f"ATR calculation error for {symbol} (TF: {timeframe}): {str(e)}")
            return 0.0

    def _validate_atr(self, atr: float, symbol: str) -> bool:
        """More robust volatility check"""
        if atr <= 0:
            return False
        pair = symbol[:6]
        min_atr = self.min_acceptable_volatility.get(pair, 0.0003)
        max_atr = self.max_acceptable_volatility.get(pair, 0.002)
        return min_atr <= atr <= max_atr

    def entry_signal(self, symbol: str) -> Optional[str]:
        """Enhanced multi-factor entry signal with weighted confirmations"""
        if self.config.get('debug', False):
            self.show_order_flow(symbol)
        try:
            # Get data for multiple timeframes
            m1_rates = MT5Wrapper.get_rates(symbol, MT5Wrapper.TIMEFRAME_M1, 0, 15)
            m5_rates = MT5Wrapper.get_rates(symbol, MT5Wrapper.TIMEFRAME_M5, 0, 10)
            
            if m1_rates is None or m5_rates is None or len(m1_rates) < 15 or len(m5_rates) < 10:
                return None
                
            # M1 Breakout logic
            m1_closes = [r['close'] for r in m1_rates]
            m1_breakout_buy = m1_closes[-1] > max(m1_closes[-6:-1])
            m1_breakout_sell = m1_closes[-1] < min(m1_closes[-6:-1])
            
            # M5 Trend filter
            m5_closes = [r['close'] for r in m5_rates]
            m5_ma = sum(m5_closes[-5:])/5  # Simple 5-period MA
            m5_trend_filter_buy = m5_closes[-1] > m5_ma
            m5_trend_filter_sell = m5_closes[-1] < m5_ma
            
            # Volume spike confirmation (fixed - removed ._asdict())
            volume_confirmed = False
            if len(m1_rates) > 10 and 'real_volume' in m1_rates[0]:
                avg_volume = sum(r['real_volume'] for r in m1_rates[-10:-1])/9
                volume_spike = m1_rates[-1]['real_volume'] > avg_volume * 1.5
                volume_confirmed = volume_spike
            
            # Price action confirmation with confidence
            price_action_signal, pa_confidence = self._detect_price_action(m1_rates)
            
            # Order flow confirmation
            buy_flow, sell_flow = self._order_flow_confirmation(symbol)

            # Only consider order flow if we have meaningful data
            order_flow_weight = 0.0
            if buy_flow > 0.6 or sell_flow > 0.6:  # Strong signal
                order_flow_weight = 0.25
            elif buy_flow > 0.4 or sell_flow > 0.4:  # Moderate signal
                order_flow_weight = 0.15
                
            # Weighted decision making
            buy_score = 0
            sell_score = 0
            
            atr = self.calculate_atr(symbol)
            if not self._validate_atr(atr, symbol):
                print(f"ATR validation failed for {symbol}: {atr:.5f}")
                return None
        
            daily_atr = atr * (24*12)
            if not self.is_acceptable_volatility(symbol, daily_atr):
                return None
                
            # Weighted decision making
            buy_score = 0
            sell_score = 0
            
            # Breakout (40% weight)
            if m1_breakout_buy:
                buy_score += 0.4
            if m1_breakout_sell:
                sell_score += 0.4

            # Order flow contribution
            if order_flow_weight > 0:
                if buy_flow > sell_flow:
                    buy_score += order_flow_weight * (buy_flow - sell_flow)
                else:
                    sell_score += order_flow_weight * (sell_flow - buy_flow)
            
            # Trend filter (20% weight)
            if m5_trend_filter_buy:
                buy_score += 0.2
            if m5_trend_filter_sell:
                sell_score += 0.2
                
            # Volume (15% weight)
            if volume_confirmed:
                if m1_breakout_buy:
                    buy_score += 0.15
                if m1_breakout_sell:
                    sell_score += 0.15
                    
            # Price action (15% weight)
            if price_action_signal == 'buy':
                buy_score += 0.15 * pa_confidence
            elif price_action_signal == 'sell':
                sell_score += 0.15 * pa_confidence
                
            # Order flow (10% weight)
            if buy_flow:
                buy_score += 0.1
            if sell_flow:
                sell_score += 0.1
                
            # Only enter if we have strong confirmation (>= 0.75 score)
            if buy_score >= 0.75 and buy_score > sell_score:
                return 'buy'
            elif sell_score >= 0.75 and sell_score > buy_score:
                return 'sell'
                
        except Exception as e:
            print(f"Signal error for {symbol}: {str(e)}")
        return None
    
    def _detect_price_action(self, rates: list) -> Tuple[Optional[str], float]:
        """Detect price action patterns with confidence scoring
        Returns:
            tuple: (signal_direction, confidence_score) where score is 0-1
        """
        if len(rates) < 3:
            return None, 0.0
            
        current = rates[-1]
        prev = rates[-2]
        prev_prev = rates[-3]
        
        confidence = 0.0
        signal = None
        
        # Calculate basic candle properties
        body_size = abs(current['open'] - current['close'])
        total_range = current['high'] - current['low']
        
        if total_range > 0:  # Avoid division by zero
            # Pinbar detection with confidence scoring
            upper_wick = current['high'] - max(current['open'], current['close'])
            lower_wick = min(current['open'], current['close']) - current['low']
            
            # Bullish pinbar
            if (lower_wick >= 2 * body_size and 
                lower_wick >= 0.6 * total_range):
                signal = 'buy'
                confidence = min(0.9, lower_wick / (total_range * 1.5))
                
            # Bearish pinbar
            elif (upper_wick >= 2 * body_size and 
                upper_wick >= 0.6 * total_range):
                signal = 'sell'
                confidence = min(0.9, upper_wick / (total_range * 1.5))
        
        # Engulfing pattern detection with confidence
        # Bullish engulfing
        if (current['close'] > current['open'] and
            prev['close'] < prev['open'] and
            current['open'] < prev['close'] and
            current['close'] > prev['open']):
            
            engulfing_size = current['close'] - current['open']
            prev_size = prev['open'] - prev['close']
            engulfing_ratio = engulfing_size / prev_size if prev_size > 0 else 1
            
            if engulfing_ratio > 1.2:  # Only consider strong engulfing
                signal = 'buy'
                confidence = max(confidence, min(0.8, engulfing_ratio / 2))
                
        # Bearish engulfing
        elif (current['close'] < current['open'] and
            prev['close'] > prev['open'] and
            current['open'] > prev['close'] and
            current['close'] < prev['open']):
            
            engulfing_size = current['open'] - current['close']
            prev_size = prev['close'] - prev['open']
            engulfing_ratio = engulfing_size / prev_size if prev_size > 0 else 1
            
            if engulfing_ratio > 1.2:
                signal = 'sell'
                confidence = max(confidence, min(0.8, engulfing_ratio / 2))
        
        return signal, confidence

    def place_trade(self, symbol: str, signal: str, atr: Optional[float] = None) -> bool:
        """Production-grade trade execution"""
        
        # Initial checks - no tick required
        if not self._is_trading_hours():
            return False
        if not self.symbols[symbol]['trade_allowed']:
            print(f"Trading not allowed for {symbol}")
            return False
        if signal not in ['buy', 'sell']:
            print(f"Invalid signal: {signal}")
            return False
        if self.check_open_positions(symbol):
            return False

        # Get account info first (returns dict)
        account = MT5Wrapper.get_account_info()
        if not account:
            print("Failed to get account info")
            return False
        
        # First tick retrieval (returns dict)
        tick = MT5Wrapper.get_symbol_tick(symbol)
        if not tick:
            print(f"Failed to get tick for {symbol}")
            return False
            
        # Spread check with tolerance (using dictionary access)
        current_spread = (tick['ask'] - tick['bid']) / self.symbols[symbol]['point']
        if current_spread > self.config['max_spread'] * 1.5:
            print(f"Spread too wide: {current_spread:.1f} points")
            return False
        
        # ATR checks
        atr = self.calculate_atr(symbol) if atr is None else atr
        if not self._validate_atr(atr, symbol):
            return False

        # Slippage check (using dictionary access)
        expected_price = tick['ask'] if signal == 'buy' else tick['bid']
        if abs(tick['last'] - expected_price) > 3*self.symbols[symbol]['point']:
            print(f"High slippage detected ({tick['last']} vs {expected_price}) - aborting")
            return False

        # Entry price using same tick
        entry = expected_price
        stop_loss, take_profit = self.calculate_sl_tp(symbol, entry, signal == 'buy', atr)
        
        if not self.validate_trade(entry, stop_loss, take_profit):
            return False   

        # Session-based risk adjustment 
        original_risk = self.config['risk_percent']
        
        if self._is_london_open():
            self.config['risk_percent'] = min(original_risk * 1.3, 1.0)  # Cap at 1%
            print(f"London open active - risk adjusted to {self.config['risk_percent']:.2f}%")
        elif self._is_ny_close():
            self.config['risk_percent'] = min(max(self.config['risk_percent'], 0.1), 2.0)
            print(f"NY close approaching - risk reduced to {self.config['risk_percent']:.2f}%")

        # Dynamic risk adjustment
        original_risk = self.config['risk_percent']
        adjusted_risk = self._adjust_risk_based_on_volatility(symbol)
        
        if adjusted_risk != original_risk:
            print(f"Volatility adjustment: Risk changed from {original_risk}% to {adjusted_risk}%")
            self.config['risk_percent'] = adjusted_risk
            
        # Position size calculation
        size = self.calculate_position_size(symbol, entry, stop_loss)
        size = self._get_safe_position_size(
            symbol=symbol,
            target_size=size,
            is_buy=(signal == 'buy')  # Convert signal to boolean
        )
        size = self._time_adjusted_size(symbol, size)
        self.config['risk_percent'] = min(max(self.config['risk_percent'], 0.1), 2.0)
            
        if size <= 0:
            print(f"Invalid position size for {symbol}: {size:.2f} lots")
            return False
        
        if not self.validate_position_size(symbol, size):
            print(f"Invalid size {size:.2f} for {symbol}. Min: {self.symbols[symbol]['volume_min']}, Max: {self.symbols[symbol]['volume_max']}")
            return False

        # Risk management check (positions returns list of dicts)
        current_positions = len([p for p in MT5Wrapper.get_positions() if p['magic'] == self.config['magic_number']])
        self.risk_manager.check_position_count(current_positions)

        # Define order type
        order_type = MT5Wrapper.ORDER_TYPE_BUY if signal == 'buy' else MT5Wrapper.ORDER_TYPE_SELL
        
        # Margin check (using dict access)
        margin_req = MT5Wrapper.calculate_margin(order_type, symbol, size, entry)
        if margin_req > account['margin_free']:
            print(f"Insufficient margin: {margin_req:.2f} required, {account['margin_free']:.2f} available")
            return False

        # Execute trade
        request = {
            "action": MT5Wrapper.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size,
            "type": order_type,
            "price": entry,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Scalper",
            "type_time": MT5Wrapper.ORDER_TIME_GTC,
            "type_filling": MT5Wrapper.ORDER_FILLING_IOC,
        }

        # Add retry logic
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            result = MT5Wrapper.send_order(request)  # Returns dict
            if result and result['retcode'] == MT5Wrapper.TRADE_RETCODE_DONE:
                break
            time.sleep(1)
        else:
            print(f"Order failed after {max_retries} attempts")
            return False

        if not result or result['retcode'] != MT5Wrapper.TRADE_RETCODE_DONE:
            print(f"Order failed with retcode {result.get('retcode', 'UNKNOWN')}")
            return False
            
        # Enhanced partial fill handling (using dict access)
        if result.get('volume', 0) < size:
            print(f"Partial fill: {result.get('volume', 0)}/{size} lots")
            self._handle_partial_fill(result, symbol, signal)
            
        # Log successful trade (using dict access)
        self.trade_log.append({
            'time': datetime.now(),
            'symbol': symbol,
            'direction': signal,
            'size': size,
            'entry': entry,
            'sl': stop_loss,
            'tp': take_profit,
            'status': 'open',
            'ticket': result.get('order', 0),
            'fill_ratio': result.get('volume', 0)/size if size > 0 else 1.0
        })

        # Log trade event
        self.log_trade_event('open', {
            'symbol': symbol,
            'size': size,
            'price': entry
        })
        
        print(f"Trade executed: {symbol} {signal} at {entry} ({size:.2f} lots)")
        return True

    def _get_safe_position_size(self, symbol: str, target_size: float, is_buy: bool) -> float:
        """
        Type-safe liquidity check that prevents oversized orders
        Args:
            symbol: Trading instrument (e.g. "EURUSD")
            target_size: Original calculated position size
            is_buy: True for buy orders, False for sell orders
        Returns:
            Adjusted position size respecting liquidity limits
        """
        # 1. Apply maximum lot size constraint
        max_lots = self.symbols[symbol].get('volume_max', 100)
        size = min(target_size, max_lots)

        # 2. Check market liquidity
        depth = MT5Wrapper.get_market_depth(symbol)
        if not depth or not isinstance(depth, dict):
            return size  # Return size after max lots check

        bids = depth.get('bid', [])
        asks = depth.get('ask', [])
        
        if not isinstance(bids, list) or not isinstance(asks, list):
            return size  # Return size after max lots check

        # 3. Process relevant side (bids for sells, asks for buys)
        levels = asks if is_buy else bids
        valid_volumes = []
        
        for level in levels[:3]:  # Top 3 price levels
            if isinstance(level, (tuple, list)) and len(level) >= 2:
                try:
                    volume = float(level[1])
                    if volume >= 0:  # Only accept valid volumes
                        valid_volumes.append(volume)
                except (TypeError, ValueError):
                    continue

        # 4. Apply liquidity constraint (max 20% of available volume)
        if valid_volumes:
            available_liquidity = sum(valid_volumes)
            return min(size, available_liquidity * 0.2)
        
        return size  # Fallback to max-lots-constrained size

    def _handle_partial_fill(self, result: Dict[str, Any], symbol: str, signal: str) -> None:
        """
        Handles partial order fills with position adjustment and risk management
        
        Args:
            result: The order result dictionary containing:
                - 'volume': Actually filled volume
                - 'request': Original request details
            symbol: Trading symbol (e.g., 'EURUSD')
            signal: Trade direction ('buy' or 'sell')
        """
        try:
            filled_volume = result.get('volume', 0)
            requested_volume = result.get('request', {}).get('volume', 0)
            
            if filled_volume <= 0 or filled_volume >= requested_volume:
                return
                
            print(f"Handling partial fill: {filled_volume:.2f}/{requested_volume:.2f} lots")
            
            # Calculate remaining volume and validate
            remaining_volume = requested_volume - filled_volume
            if remaining_volume <= 0:
                print("No remaining volume to fill")
                return
                
            entry_price = result.get('price', 0)
            
            # Calculate adjusted risk for remaining volume
            risk_adjusted_volume = self.calculate_position_size(
                symbol,
                entry_price,
                result.get('request', {}).get('sl', 0))
                
            # Take the minimum between remaining volume and risk-adjusted volume
            adjusted_volume = min(remaining_volume, risk_adjusted_volume)
            
            if adjusted_volume <= 0:
                print("No valid adjusted volume after partial fill")
                return
                
            # Create new order for remaining volume
            new_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": adjusted_volume,
                "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": result.get('request', {}).get('sl', 0),
                "tp": result.get('request', {}).get('tp', 0),
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": "Partial fill completion",
                "type_time": MT5Wrapper.ORDER_TIME_GTC,
                "type_filling": MT5Wrapper.ORDER_FILLING_IOC,
            }
            
            # Execute adjusted order with retry logic
            retry_count = 0
            while retry_count < 3:
                try:
                    fill_result = MT5Wrapper.send_order(new_request)
                    if fill_result and fill_result.get('retcode') == MT5Wrapper.TRADE_RETCODE_DONE:
                        print(f"Successfully filled remaining {adjusted_volume:.2f} lots")
                        # Update trade log
                        if hasattr(self, 'trade_log') and self.trade_log:
                            self.trade_log[-1]['partial_fills'] = self.trade_log[-1].get('partial_fills', 0) + 1
                        break
                except Exception as e:
                    print(f"Error during partial fill retry {retry_count}: {str(e)}")
                finally:
                    retry_count += 1
                    time.sleep(1)
            else:
                print(f"Failed to complete partial fill after 3 attempts")
                
        except KeyError as e:
            print(f"Missing key in result dictionary: {str(e)}")
        except Exception as e:
            print(f"Unexpected error handling partial fill: {str(e)}")
            # Consider adding notification or logging here

    def calculate_sl_tp(self, symbol: str, entry: float, is_long: bool, atr: float) -> Tuple[float, float]:
        """ATR-based stops with minimum distance"""
        min_dist = 10 * self.symbols[symbol]['point']
        sl_dist = max(1.5 * atr, min_dist)
        tp_dist = sl_dist * self.config['risk_reward']
        
        if is_long:
            return (entry - sl_dist, entry + tp_dist)
        return (entry + sl_dist, entry - tp_dist)

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """Safe volume normalization"""
        step = self.symbols[symbol]['volume_step']
        return round(volume / step) * step

    def validate_trade(self, entry: float, sl: float, tp: float) -> bool:
        """Risk-reward validation"""
        try:
            risk = abs(entry - sl)
            reward = abs(entry - tp)
            return reward / risk >= self.config['risk_reward']
        except ZeroDivisionError:
            return False

    def validate_position_size(self, symbol, size):
        """Validate position size against symbol's min/max limits"""
        if symbol not in self.symbols:
            return False
            
        info = self.symbols[symbol]
        
        # Check if volume limits exist in the symbol info
        if 'volume_min' not in info or 'volume_max' not in info:
            # Fallback to MT5's symbol info if not in our cache
            symbol_info = MT5Wrapper.get_symbol_info(symbol)
            if symbol_info:
                return (size >= symbol_info.volume_min and 
                        size <= symbol_info.volume_max)
            return False
            
        return size >= info['volume_min'] and size <= info['volume_max']

    def check_open_positions(self, symbol: str) -> bool:
        """Check positions with magic number using proper dictionary access
        
        Args:
            symbol: The trading symbol to check positions for
            
        Returns:
            bool: True if positions exist with matching magic number, False otherwise
        """
        positions = MT5Wrapper.get_positions(symbol=symbol)
        if not positions:  # Handles both None and empty list cases
            return False
            
        # Access magic number using dictionary syntax
        return any(
            pos.get('magic', 0) == self.config['magic_number'] 
            for pos in positions
            if isinstance(pos, dict)  # Additional type safety
        )

    def manage_trades(self):
        """Professional trade management with ATR flow using MT5Wrapper"""
        # Get positions using wrapper
        positions = MT5Wrapper.get_positions()
        current_positions = len([p for p in positions if p['magic'] == self.config['magic_number']])
        self.risk_manager.check_position_count(current_positions)

        # First check for any closed positions to update logs
        self.check_closed_positions()
        
        # Pre-calculate ATR for all symbols once
        atr_cache = {
            symbol: {
                'raw': self.calculate_atr(symbol),
                'daily': self.calculate_atr(symbol) * (24*12)
            } for symbol in self.symbols
        }
        
        if not positions:
            return

        london_tz = pytz.timezone('Europe/London')
        current_time = datetime.now(london_tz)

        for pos in positions:
            if pos['magic'] != self.config['magic_number']:
                continue

            # Initialize symbol here so it's available in exception handling
            symbol = pos['symbol']
            try:
                pos_type = pos['type']
                
                # Get current price using wrapper
                tick = MT5Wrapper.get_symbol_tick(symbol)
                if not tick:
                    continue
                    
                current_price = tick['bid'] if pos_type == mt5.ORDER_TYPE_BUY else tick['ask']
                
                # Get pre-calculated ATR with safety check
                atr = atr_cache.get(symbol)
                if not atr:
                    continue
                    
                daily_atr = atr['raw'] * (24*12)
                
                # Volatility check
                if not self.is_acceptable_volatility(symbol, daily_atr):
                    self.close_position(pos)
                    continue

                # Time-based exit
                open_time = datetime.fromtimestamp(pos['time'], tz=london_tz)
                duration = (current_time - open_time).total_seconds() / 60
                if duration > self.config['max_trade_duration']:
                    self.close_position(pos)
                    continue

                # Risk/reward calculation
                risk = abs(pos['price_open'] - pos['sl'])
                profit = abs(current_price - pos['price_open'])
                rr_ratio = profit / risk if risk > 0 else 0

                # Partial profit taking
                if rr_ratio >= 1.5 and pos['volume'] > 0.02:
                    self.close_partial_position(pos, pos['volume'] * 0.5)

                # Trailing stop with ATR flow
                self.update_trailing_stop(
                    position=pos,
                    current_price=current_price,
                    atr=atr['raw']
                )

            except Exception as e:
                print(f"Trade management error for position {pos.get('ticket', 'unknown')} (symbol: {symbol}): {str(e)}")

    def update_trailing_stop(self, position, current_price, atr: Optional[float] = None):
        """Enhanced trailing stop implementation with ATR"""
        symbol = position.symbol
        point = self.symbols[symbol]['point']
        pip_value = point * 10
        
        if atr is None:
            atr = self.calculate_atr(symbol)  # Fallback
        
        profit_pips = abs(current_price - position.price_open) / pip_value
        
        # Dynamic ATR trailing when profit is good
        if profit_pips >= self.config['trail_start'] * 2:
            dynamic_factor = min(2.0, 1.0 + (profit_pips/10))
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (atr * dynamic_factor)
                # Ensure new SL is valid and better than current
                if new_sl > max(position.sl, position.price_open):
                    self.modify_sl(position, new_sl)
            else:  # SELL position
                new_sl = current_price + (atr * dynamic_factor)
                if new_sl < min(position.sl, position.price_open) or position.sl == 0:
                    self.modify_sl(position, new_sl)
        else:
            # Default fixed trailing
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - self.config['trail_step'] * pip_value
                if new_sl > max(position.sl, position.price_open):
                    self.modify_sl(position, new_sl)
            else:  # SELL position
                new_sl = current_price + self.config['trail_step'] * pip_value
                if new_sl < min(position.sl, position.price_open) or position.sl == 0:
                    self.modify_sl(position, new_sl)

    def is_acceptable_volatility(self, symbol: str, daily_atr: float) -> bool:
        """Check if volatility is within acceptable range for the symbol"""
        pair = symbol[:6]  # Get just the pair (e.g., 'EURUSD' from 'EURUSD.a')
        min_vol = self.min_acceptable_volatility.get(pair, 0.0003)
        max_vol = self.max_acceptable_volatility.get(pair, 0.002)
        return min_vol <= daily_atr <= max_vol

    def close_position(self, position):
        """Updated to safely handle potential None returns from MT5Wrapper"""
        # Get tick data with null check
        tick = MT5Wrapper.get_symbol_tick(position['symbol'])
        if not tick:
            print(f"Failed to get tick data for {position['symbol']}")
            return False
        
        # Determine price based on position type
        price = tick['ask'] if position['type'] == mt5.ORDER_TYPE_BUY else tick['bid']
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position['ticket'],
            "symbol": position['symbol'],
            "volume": position['volume'],
            "type": mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Exit",
            "type_time": MT5Wrapper.ORDER_TIME_GTC,
            "type_filling": MT5Wrapper.ORDER_FILLING_IOC,
        }
        
        result = MT5Wrapper.send_order(request)
        if result and result['retcode'] == MT5Wrapper.TRADE_RETCODE_DONE:
            self._update_trade_log(position['ticket'], 'closed', result['price'])
            print(f"Closed position {position['ticket']} at {result['price']}")
            return True
        
        print(f"Failed to close position {position['ticket']}")
        return False

    def log_trade_event(self, event_type: str, details: dict) -> None:
        """Log trade events with account equity and performance data"""
        account_info = None
        if MT5Wrapper.initialize():
            account_info = MT5Wrapper.get_account_info()
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'symbol': details.get('symbol'),
            'price': details.get('price'),
            'size': details.get('size'),
            'equity': account_info.get('equity') if account_info else None,
            'daily_pnl': self.calculate_daily_pnl()
        }
        
        self.trade_log.append(log_data)

    def calculate_daily_pnl(self) -> float:
        """Calculate today's realized PnL from closed trades in trade_log"""
        today = datetime.now().date()
        pnl = 0.0
        for trade in self.trade_log:
            if trade.get('status') == 'closed' and trade.get('time', datetime.now()).date() == today:
                pnl += self.calculate_trade_profit(trade)
        return pnl

    def close_partial_position(self, position: Dict[str, Any], volume: float) -> bool:
        """Closes a portion of an existing position with proper dictionary access
        
        Args:
            position: Dictionary containing position details
            volume: Volume to close (in lots)
            
        Returns:
            bool: True if successful, False otherwise
        """
        request = self._close_request(position, volume)
        result = MT5Wrapper.send_order(request)  # Returns dictionary
        
        if not result:
            print("Failed to get order result")
            return False
            
        # Access dictionary values safely
        if result.get('retcode') == MT5Wrapper.TRADE_RETCODE_DONE:
            self.log_trade_event('partial_close', {
                'symbol': position.get('symbol', 'UNKNOWN'),
                'size': volume,
                'price': result.get('price', 0.0)
            })
            
        pnl = self.calculate_daily_pnl()
        print(f"Daily PNL after partial close: ${pnl:.2f}")
        
        return result.get('retcode') == MT5Wrapper.TRADE_RETCODE_DONE
            
    def check_closed_positions(self):
        """Check for recently closed positions using wrapper"""
        now = datetime.now()
        closed_deals = MT5Wrapper.get_history_deals(now - timedelta(minutes=30), now)
        
        for deal in closed_deals:
            if deal['entry'] == MT5Wrapper.DEAL_ENTRY_OUT:
                self._update_trade_log(deal['position'], 'closed', deal['price'])

    def _close_request(self, position, volume):
        # Get the current tick data safely
        tick = MT5Wrapper.get_symbol_tick(position.symbol)
        if not tick:
            raise ValueError(f"Failed to get tick data for {position.symbol}")

        # Determine exit price based on position type
        exit_price = tick['ask'] if position.type == mt5.ORDER_TYPE_BUY else tick['bid']

        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": exit_price,  # Use the pre-fetched price
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Exit",
            "type_time": MT5Wrapper.ORDER_TIME_GTC,
            "type_filling": MT5Wrapper.ORDER_FILLING_IOC,
        }

    def modify_sl(self, position: Dict[str, Any], new_sl: float) -> bool:
        """Modify stop loss with proper dictionary access
        
        Args:
            position: Dictionary containing position details
            new_sl: New stop loss price
            
        Returns:
            bool: True if modification was successful, False otherwise
        """
        try:
            request = {
                "action": MT5Wrapper.TRADE_ACTION_SLTP,
                "position": position.get('ticket', 0),
                "symbol": position.get('symbol', ''),
                "sl": new_sl,
                "tp": position.get('tp', 0.0),
                "deviation": self.config.get('deviation', 10),
                "magic": self.config.get('magic_number', 0),
                "comment": "Trailing stop",
                "type_time": MT5Wrapper.ORDER_TIME_GTC,
            }
            
            result = MT5Wrapper.send_order(request)
            
            # Safely access dictionary result
            if not result:
                print("Failed to get modification result")
                return False
                
            return result.get('retcode', -1) == MT5Wrapper.TRADE_RETCODE_DONE
            
        except Exception as e:
            print(f"Error modifying stop loss: {str(e)}")
            return False

    def _update_trade_log(self, ticket, status, exit_price=None):
     for trade in self.trade_log:
         if trade['ticket'] == ticket:
             trade['status'] = status
             if exit_price is not None:
                 trade['exit_price'] = exit_price
             break

    def performance_report(self):
        """Enhanced performance analytics"""
        closed_trades = [t for t in self.trade_log if t['status'] == 'closed']
        if not closed_trades:
            print("No closed trades")
            return

        # Add daily PNL display
        daily_pnl = self.calculate_daily_pnl()
        print(f"\nDaily P&L: ${daily_pnl:.2f}")
            
        wins = [t for t in closed_trades if self.calculate_trade_profit(t) > 0]
        losses = [t for t in closed_trades if self.calculate_trade_profit(t) <= 0]
        
        win_rate = len(wins) / len(closed_trades)
        avg_win = np.mean([self.calculate_trade_profit(t) for t in wins]) if wins else 0
        avg_loss = np.mean([abs(self.calculate_trade_profit(t)) for t in losses]) if losses else 0
        profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else float('inf')
        
        print(f"\nPerformance Report ({len(closed_trades)} trades)")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy: ${(win_rate * avg_win - (1 - win_rate) * avg_loss):.2f}")

    def _order_flow_confirmation(self, symbol: str) -> Tuple[float, float]:
        """
        Calculate order flow confirmation based on market depth and volume analysis.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Tuple containing:
            - buy_flow_ratio: float (0.0 to 1.0)
            - sell_flow_ratio: float (0.0 to 1.0)
        """
        # Check market depth first
        depth = MT5Wrapper.get_market_depth(symbol)
        if not depth:
            return 0.0, 0.0
            
        try:
            # Safely get bid and ask data with proper type checking
            bids = depth.get('bid')
            asks = depth.get('ask')
            
            # Initialize volumes
            bid_vol = 0.0
            ask_vol = 0.0
            
            # Process bids if they exist and are in correct format
            if isinstance(bids, list):
                for level in bids[:3]:  # Only first 3 levels
                    if isinstance(level, (tuple, list)) and len(level) >= 2:
                        try:
                            vol = level[1]
                            if vol is not None:
                                bid_vol += float(vol)
                        except (TypeError, ValueError):
                            continue
            
            # Process asks if they exist and are in correct format
            if isinstance(asks, list):
                for level in asks[:3]:  # Only first 3 levels
                    if isinstance(level, (tuple, list)) and len(level) >= 2:
                        try:
                            vol = level[1]
                            if vol is not None:
                                ask_vol += float(vol)
                        except (TypeError, ValueError):
                            continue
            
            total = bid_vol + ask_vol
            if total > 0:
                return bid_vol/total, ask_vol/total
                
        except Exception as e:
            print(f"Error processing market depth for {symbol}: {e}")
            return 0.0, 0.0

        # Fallback to volume spike analysis
        rates = MT5Wrapper.get_rates(symbol, MT5Wrapper.TIMEFRAME_M1, 0, 5)
        if rates and len(rates) >= 3:
            try:
                current_vol = float(rates[-1].get('real_volume', 0.0))
                avg_vol = np.mean([float(r.get('real_volume', 0.0)) for r in rates[:-1]])
                
                if current_vol > avg_vol * 1.5:
                    return (0.7, 0.3) if rates[-1]['close'] > rates[-2]['close'] else (0.3, 0.7)
            except (TypeError, ValueError, KeyError) as e:
                print(f"Error processing volume data for {symbol}: {e}")
        
        return 0.0, 0.0

    def calculate_trade_profit(self, trade):
        if trade['status'] != 'closed':
            return 0
            
        # Calculate price difference
        if trade['direction'] == 'buy':
            price_diff = trade['exit_price'] - trade['entry']
        else:  # sell
            price_diff = trade['entry'] - trade['exit_price']
        
        # JPY-specific calculation (convert yen profit to account currency)
        if "JPY" in trade['symbol']:
            # Profit in JPY: (price_diff / 0.01) * 1000 * size
            # Then convert to account currency (USD) using exit price
            jpy_profit = price_diff * 100000  # Profit in JPY
            return jpy_profit / trade['exit_price'] * trade['size']
        
        # Standard pairs (EURUSD)
        return price_diff * 100000 * trade['size']  # 100,000 units per standard lot

    def _adjust_risk_based_on_volatility(self, symbol: str) -> float:
        """Dynamic risk adjustment with symbol-specific thresholds and safety bounds"""
        atr = self.calculate_atr(symbol)
        base_risk = self.config['risk_percent']
        
        # Configurable thresholds (can move these to config)
        thresholds = {
            'EURUSD': {'low': 0.0005, 'high': 0.0015, 'max_mult': 1.3, 'min_mult': 0.4},
            'USDJPY': {'low': 0.05, 'high': 0.15, 'max_mult': 1.3, 'min_mult': 0.4}
        }
        
        pair = symbol[:6]
        params = thresholds.get(pair, {'low': 0.0003, 'high': 0.003, 'max_mult': 1.2, 'min_mult': 0.5})
        
        # Safety bounds
        MIN_RISK = 0.1  # 0.1% minimum
        MAX_RISK = 2.0  # 2.0% maximum
        
        if atr > params['high']:
            adjusted = base_risk * params['min_mult']
        elif atr < params['low']:
            adjusted = base_risk * params['max_mult']
        else:
            return base_risk
            
        return min(MAX_RISK, max(MIN_RISK, adjusted))

    def show_order_flow(self, symbol: str) -> None:
        """Safe order flow display with type checking"""
        depth = MT5Wrapper.get_market_depth(symbol)
        if not depth or not isinstance(depth, dict):
            print(f"No valid depth data for {symbol}")
            return

        def safe_volume_sum(levels: Any) -> float:
            """Handle both MT5 DepthLevel objects and raw tuples/lists"""
            if not levels:
                return 0.0
                
            # Case 1: List of DepthLevel objects (from MT5)
            if hasattr(levels[0], 'price') and hasattr(levels[0], 'volume'):
                return sum(float(level.volume) for level in levels)
                
            # Case 2: List of tuples (price, volume)
            elif isinstance(levels[0], (tuple, list)) and len(levels[0]) >= 2:
                return sum(float(v) for _, v in levels)
                
            return 0.0

        try:
            bid_vol = safe_volume_sum(depth.get('bid', []))
            ask_vol = safe_volume_sum(depth.get('ask', []))
            
            print(f"\n{symbol} Order Flow:")
            print(f"Total Bid Volume: {bid_vol:.2f}")
            print(f"Total Ask Volume: {ask_vol:.2f}")
            print(f"Imbalance Ratio: {(bid_vol - ask_vol)/max(bid_vol, ask_vol, 1):+.2%}")
            
        except Exception as e:
            print(f"Order flow error: {str(e)}")

    def run(self):
        """Robust main trading loop with proper dictionary access and error handling"""
        print("Starting scalping bot...")

        last_reconnect = time.time()
        last_day_check = datetime.now()
        trade_count = 0
        self.last_dst_state = None  # Initialize DST state
        
        while True:
            try:
                current_time = datetime.now()
                
                # Daily reset check
                if current_time.date() != last_day_check.date():
                    self.risk_manager.reset_daily_drawdown()
                    last_day_check = current_time
                    
                # Connection management (using wrapper methods)
                terminal_info = MT5Wrapper.terminal_info()
                if not MT5Wrapper.initialize() or not terminal_info or not terminal_info.get('connected', False):
                    self.reconnect()
                    
                if time.time() - last_reconnect > 3600:  # Hourly reconnect
                    MT5Wrapper.shutdown()
                    MT5Wrapper.initialize()
                    last_reconnect = time.time()

                # Sync with broker time (safe symbol access)
                if hasattr(self, 'symbols') and self.symbols:
                    symbol = next(iter(self.symbols))  # Get first symbol
                    tick = MT5Wrapper.get_symbol_tick(symbol)
                    if tick and 'time' in tick:
                        broker_time = datetime.fromtimestamp(tick['time'])
                        local_drift = datetime.now() - broker_time
                        if abs(local_drift) > timedelta(seconds=2):
                            print(f"Time drift detected: {local_drift}")

                # Handle DST transitions
                london_tz = pytz.timezone('Europe/London')
                current_dst = london_tz.localize(datetime.now()).dst()
                if current_dst != self.last_dst_state:
                    print("Daylight savings change detected")
                    self.last_dst_state = current_dst

                # Trading decision
                should_trade = (self._is_optimal_trading_time() or 
                            not self.config.get('strict_optimal_time', False))
                
                if should_trade:
                    print(f"{current_time} Optimal trading time detected")
                    
                    # Trading signals and execution
                    account = MT5Wrapper.get_account_info()
                    if account:
                        equity = account.get('equity', 0.0)
                        self.risk_manager.check_drawdown(equity)
                        
                        for symbol in getattr(self, 'symbols', {}):  # Safe symbols access
                            signal = self.entry_signal(symbol)
                            if signal:
                                print(f"{current_time} {symbol} {signal.upper()} signal")
                                self.place_trade(symbol, signal)
                    else:
                        print("Failed to retrieve account info")
                
                # Trade management
                self.manage_trades()
                
                # Periodic reporting
                trade_count += 1
                if trade_count % 10 == 0:
                    current_pnl = self.calculate_daily_pnl()
                    if abs(current_pnl) > 1000:
                        print(f"Significant daily PNL movement: ${current_pnl:.2f}")
                    self.performance_report()
                    
                time.sleep(5)
                
            except RiskBreach as e:
                print(f"RISK MANAGEMENT TRIGGERED: {str(e)}")
                self.emergency_stop()
                break
                
            except KeyboardInterrupt:
                print("Stopped by user")
                break
                
            except Exception as e:
                print(f"System error: {str(e)}")
                self.reconnect()
                time.sleep(30)

    def _verify_broker_conditions(self):
        """Verify broker conditions before trading"""
        if not MT5Wrapper.initialize():
            raise ConnectionError("Failed to initialize MT5")
        
        account_info = MT5Wrapper.get_account_info()
        if account_info is None:
            raise ConnectionError("Failed to retrieve account info")
        
        if not account_info.get('trade_allowed', False):  # Safely access dict key
            raise PermissionError("Trading is not allowed on this account")
        
        print("Broker conditions verified successfully")

    def emergency_stop(self):
        """Close all positions and cancel orders with proper dictionary access"""
        print("Executing emergency stop...")
        
        # Close positions
        positions = MT5Wrapper.get_positions()
        for pos in positions:
            if pos.get('magic', 0) == self.config.get('magic_number', 0):
                self.close_position(pos)
        
        # Cancel orders
        orders = MT5Wrapper.get_orders()
        for order in orders:
            if order.get('magic', 0) == self.config.get('magic_number', 0):
                MT5Wrapper.send_order({
                    "action": MT5Wrapper.TRADE_ACTION_REMOVE,
                    "order": order.get('ticket', 0)
                })
        
        self.risk_manager.reset_daily_drawdown()
        print("Emergency stop complete")

    def log_to_file(self, message: str):
        """Structured JSON logging
        
        Args:
            message: Log message to record
            
        Handles cases where:
        - Account info is unavailable
        - File operations fail
        - JSON serialization fails
        """
        try:
            # Safely get account info with None check
            account_info = MT5Wrapper.get_account_info()
            equity = account_info.get('equity') if account_info else None
            
            # Create log entry with fallbacks
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "equity": equity,
                "message": message,
                "positions": len(MT5Wrapper.get_positions() or [])  # Handle None case
            }
            
            # Write to file with error handling
            with open("trading_log.json", "a", encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")  # Newline for each entry
                
        except Exception as e:
            print(f"Failed to write log entry: {str(e)}")

    def reconnect(self):
        """Reconnect to MetaTrader5 terminal"""
        try:
            MT5Wrapper.shutdown()
        except Exception:
            pass
        finally:
            time.sleep(2)
            if not MT5Wrapper.initialize():
                raise ConnectionError("Failed to reconnect to MT5 terminal")
            print("Reconnected to MT5 terminal")

    def backtest(self, start_date: datetime, end_date: datetime):
        """Run strategy on historical data
        
        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
        """
        for symbol in self.symbols:
            # Get all available rates between dates
            rates = []
            current_date = start_date
            while current_date < end_date:
                # Get rates in chunks (1000 bars at a time)
                chunk = MT5Wrapper.get_rates(
                    symbol=symbol,
                    timeframe=MT5Wrapper.TIMEFRAME_M1,
                    start_pos=0,  # Start from current date
                    count=1000   # Number of bars to retrieve
                )
                if not chunk:
                    break
                rates.extend(chunk)
                current_date = datetime.fromtimestamp(chunk[-1]['time']) + timedelta(minutes=1)
            
            if not rates:
                print(f"No historical data found for {symbol}")
                continue

            # Simulate real-time processing
            for i in range(30, len(rates)):  # Warm-up period
                try:
                    # Create a rates subset up to current point
                    current_rates = rates[:i+1]
                    
                    # Simulate tick
                    self.entry_signal(symbol)  
                    
                    # Simulate position management
                    self.manage_trades()
                    
                    # Optional: Print progress
                    if i % 1000 == 0:
                        print(f"Processed {i}/{len(rates)} bars for {symbol}")
                        
                except Exception as e:
                    print(f"Error during backtest for {symbol} at index {i}: {str(e)}")
                    continue

    def _time_adjusted_size(self, symbol: str, size: float) -> float:
        """Time-based position sizing with symbol-specific adjustments"""
        now = datetime.now(pytz.timezone('Europe/London')).time()
        
        # 1. Get symbol-specific configuration
        pair_config = self.config['volatility_thresholds'].get(symbol[:6], {})
        base_reduction = pair_config.get('off_peak_reduction', 0.7)  # Default 30% reduction
        
        # 2. Lunch hours reduction (all pairs)
        if dt_time(12, 0) <= now <= dt_time(13, 30):
            return size * 0.6  # 40% reduction during lunch
                
        # 3. Full size during active sessions
        if self._is_london_open() or self._is_ny_close():
            return size
                
        # 4. Apply symbol-specific reduction for off-peak
        return size * base_reduction
                    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scalper Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--days', type=int, default=7, help='Backtest duration in days')
    args = parser.parse_args()

    bot = Scalper(risk_percent=0.5, risk_reward=1.5)
    
    try:
        if args.backtest:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            print(f"\nRunning backtest for {args.days} days ({start_date.date()} to {end_date.date()})")
            bot.backtest(start_date=start_date, end_date=end_date)
        else:
            print("\nRunning in LIVE TRADING mode")
            bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    finally:
        bot.performance_report()




# Create performance dashboard (real-time monitoring)
