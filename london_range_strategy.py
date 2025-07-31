"""
London Range Break Trading Strategy
Python implementation of the MQL5 London Breakout EA
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TradeType(Enum):
    """Trade type enumeration"""
    ALL = "all"
    BUY_ONLY = "buy_only"
    SELL_ONLY = "sell_only"

@dataclass
class Position:
    """Position information tracking"""
    ticket: str
    entry_price: float
    position_type: str  # 'buy' or 'sell'
    lot_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    london_range: float
    session_id: str
    is_trailing_active: bool = False
    
@dataclass
class PendingOrder:
    """Pending order information"""
    ticket: str
    order_type: str  # 'buy_stop' or 'sell_stop'
    price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    session_id: str

class LondonRangeBreakStrategy:
    """
    London Range Break Strategy Implementation
    
    This strategy identifies the pre-London session range (3 AM - 8 AM UTC)
    and places pending orders to trade breakouts during London session
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the London Range Break strategy
        
        Args:
            config (Dict): Strategy configuration parameters
        """
        # Trading parameters
        self.lot_size = config.get('lot_size', 0.01)
        self.trade_type = TradeType(config.get('trade_type', 'all'))
        self.magic_number = config.get('magic_number', 1234567890)
        self.rr_ratio = config.get('rr_ratio', 1.0)
        self.stop_loss_points = config.get('stop_loss_points', 2700)
        self.order_offset_points = config.get('order_offset_points', 1)
        self.delete_opposite_order = config.get('delete_opposite_order', True)
        
        # Trailing stop parameters
        self.use_trailing = config.get('use_trailing', True)
        self.trailing_points = config.get('trailing_points', 1100)
        self.min_profit_points = config.get('min_profit_points', 200)
        
        # London session parameters
        self.london_start_hour = config.get('london_start_hour', 9)
        self.london_start_minute = config.get('london_start_minute', 0)
        self.london_end_hour = config.get('london_end_hour', 8)  # Next day
        self.london_end_minute = config.get('london_end_minute', 0)
        self.min_range_points = config.get('min_range_points', 100)
        self.max_range_points = config.get('max_range_points', 5000)
        
        # Risk management
        self.max_open_trades = config.get('max_open_trades', 2)
        self.max_daily_drawdown_percent = config.get('max_daily_drawdown_percent', 5.0)
        
        # Pre-London session parameters (fixed)
        self.pre_london_start_hour = 3
        self.pre_london_start_minute = 0
        self.pre_london_end_hour = 8
        self.pre_london_end_minute = 0
        
        # Point size for XAUUSD
        self.point_size = config.get('point_size', 0.01)
        
        # Strategy state
        self.positions: List[Position] = []
        self.pending_orders: List[PendingOrder] = []
        self.session_data: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.current_equity = 0.0
        self.no_trade_today = False
        self.session_analysis_done = False
        self.last_session_date = None
        
        # Session range data
        self.pre_london_high = 0.0
        self.pre_london_low = 0.0
        self.pre_london_range_points = 0.0
        self.buy_break_level = 0.0
        self.sell_break_level = 0.0
        
    def is_new_session(self, current_time: datetime) -> bool:
        """Check if we have a new trading session (new day)"""
        session_date = current_time.date()
        if session_date != self.last_session_date:
            self.last_session_date = session_date
            self._reset_session_state()
            return True
        return False
    
    def _reset_session_state(self):
        """Reset state for new session"""
        self.session_analysis_done = False
        self.no_trade_today = False
        self.pending_orders.clear()
        self.pre_london_high = 0.0
        self.pre_london_low = 0.0
        self.pre_london_range_points = 0.0
        self.buy_break_level = 0.0
        self.sell_break_level = 0.0
    
    def is_pre_london_session(self, current_time: datetime) -> bool:
        """Check if current time is in pre-London session (3 AM - 8 AM UTC)"""
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        start_minutes = self.pre_london_start_hour * 60 + self.pre_london_start_minute
        end_minutes = self.pre_london_end_hour * 60 + self.pre_london_end_minute
        current_minutes = current_hour * 60 + current_minute
        
        return start_minutes <= current_minutes < end_minutes
    
    def is_london_session(self, current_time: datetime) -> bool:
        """Check if current time is in London session"""
        current_hour = current_time.hour
        # London session typically 8 AM - 4 PM UTC (can overlap to next day)
        return 8 <= current_hour < 16
    
    def calculate_session_range(self, data: pd.DataFrame, session_date: datetime.date) -> Tuple[float, float, float]:
        """
        Calculate pre-London session range for given date
        
        Args:
            data (pd.DataFrame): OHLCV data
            session_date (datetime.date): Date to calculate range for
            
        Returns:
            Tuple[float, float, float]: (high, low, range_in_points)
        """
        # Filter data for pre-London session on given date
        session_start = datetime.combine(session_date, time(self.pre_london_start_hour, self.pre_london_start_minute))
        session_end = datetime.combine(session_date, time(self.pre_london_end_hour, self.pre_london_end_minute))
        
        # Get data for pre-London session
        session_mask = (data.index >= session_start) & (data.index < session_end)
        session_data = data[session_mask]
        
        if len(session_data) == 0:
            return 0.0, 0.0, 0.0
        
        high = session_data['high'].max()
        low = session_data['low'].min()
        range_points = (high - low) / self.point_size
        
        return high, low, range_points
    
    def is_valid_range(self, range_points: float) -> bool:
        """Check if range is within valid limits"""
        return self.min_range_points <= range_points <= self.max_range_points
    
    def calculate_lot_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size (simplified - fixed lot size)"""
        return self.lot_size
    
    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """
        Generate trading signals based on London Range Break strategy
        
        Args:
            data (pd.DataFrame): OHLCV data up to current time
            current_time (datetime): Current timestamp
            
        Returns:
            List[Dict]: List of signals
        """
        signals = []
        
        # Check for new session
        if self.is_new_session(current_time):
            logger.info(f"New trading session detected: {current_time.date()}")
        
        # Update session data if we're at London session start and haven't analyzed yet
        if (not self.session_analysis_done and 
            current_time.hour == self.london_start_hour and 
            current_time.minute >= self.london_start_minute):
            
            self._analyze_session_range(data, current_time)
        
        # Generate breakout signals if session is analyzed and valid
        if self.session_analysis_done and not self.no_trade_today:
            signals.extend(self._check_breakout_signals(data, current_time))
        
        return signals
    
    def _analyze_session_range(self, data: pd.DataFrame, current_time: datetime):
        """Analyze pre-London range and set up breakout levels"""
        session_date = current_time.date()
        
        # Calculate pre-London range
        high, low, range_points = self.calculate_session_range(data, session_date)
        
        if not self.is_valid_range(range_points):
            logger.info(f"Invalid range {range_points:.0f} points. No trading today.")
            self.no_trade_today = True
            self.session_analysis_done = True
            return
        
        # Store range data
        self.pre_london_high = high
        self.pre_london_low = low
        self.pre_london_range_points = range_points
        
        # Calculate breakout levels
        self.buy_break_level = high + (self.order_offset_points * self.point_size)
        self.sell_break_level = low - (self.order_offset_points * self.point_size)
        
        # Store session data
        session_id = session_date.strftime('%Y%m%d')
        self.session_data[session_id] = {
            'high': high,
            'low': low,
            'range_points': range_points,
            'buy_level': self.buy_break_level,
            'sell_level': self.sell_break_level
        }
        
        self.session_analysis_done = True
        
        logger.info(f"Session analysis complete:")
        logger.info(f"  Range: {range_points:.0f} points ({high:.5f} - {low:.5f})")
        logger.info(f"  Buy level: {self.buy_break_level:.5f}")
        logger.info(f"  Sell level: {self.sell_break_level:.5f}")
    
    def _check_breakout_signals(self, data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """Check for breakout signals"""
        signals = []
        
        if len(data) == 0:
            return signals
        
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        
        # Check for buy breakout
        if (self.trade_type in [TradeType.ALL, TradeType.BUY_ONLY] and
            current_high > self.buy_break_level and
            len(self.positions) < self.max_open_trades):
            
            signal = self._create_buy_signal(current_time, current_price)
            if signal:
                signals.append(signal)
        
        # Check for sell breakout
        if (self.trade_type in [TradeType.ALL, TradeType.SELL_ONLY] and
            current_low < self.sell_break_level and
            len(self.positions) < self.max_open_trades):
            
            signal = self._create_sell_signal(current_time, current_price)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _create_buy_signal(self, current_time: datetime, current_price: float) -> Optional[Dict]:
        """Create buy signal"""
        entry_price = self.buy_break_level
        stop_loss = entry_price - (self.stop_loss_points * self.point_size)
        take_profit = entry_price + (self.stop_loss_points * self.rr_ratio * self.point_size)
        lot_size = self.calculate_lot_size(entry_price, stop_loss)
        
        signal = {
            'timestamp': current_time,
            'signal_type': 'buy',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'strategy': 'london_range_break',
            'session_id': current_time.date().strftime('%Y%m%d'),
            'range_points': self.pre_london_range_points,
            'confidence': self._calculate_signal_confidence('buy', current_price)
        }
        
        return signal
    
    def _create_sell_signal(self, current_time: datetime, current_price: float) -> Optional[Dict]:
        """Create sell signal"""
        entry_price = self.sell_break_level
        stop_loss = entry_price + (self.stop_loss_points * self.point_size)
        take_profit = entry_price - (self.stop_loss_points * self.rr_ratio * self.point_size)
        lot_size = self.calculate_lot_size(entry_price, stop_loss)
        
        signal = {
            'timestamp': current_time,
            'signal_type': 'sell',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'strategy': 'london_range_break',
            'session_id': current_time.date().strftime('%Y%m%d'),
            'range_points': self.pre_london_range_points,
            'confidence': self._calculate_signal_confidence('sell', current_price)
        }
        
        return signal
    
    def _calculate_signal_confidence(self, signal_type: str, current_price: float) -> float:
        """Calculate signal confidence based on range quality and breakout strength"""
        if self.pre_london_range_points == 0:
            return 0.5
        
        # Base confidence on range quality
        range_confidence = 0.5
        if 200 <= self.pre_london_range_points <= 3000:
            range_confidence = 0.8
        elif 100 <= self.pre_london_range_points <= 5000:
            range_confidence = 0.6
        
        # Adjust for breakout strength
        if signal_type == 'buy':
            breakout_strength = (current_price - self.buy_break_level) / self.pre_london_range_points
        else:
            breakout_strength = (self.sell_break_level - current_price) / self.pre_london_range_points
        
        breakout_confidence = min(1.0, 0.5 + breakout_strength * 5)
        
        return (range_confidence + breakout_confidence) / 2
    
    def execute_signal(self, signal: Dict) -> Optional[Position]:
        """
        Execute a trading signal (simulation)
        
        Args:
            signal (Dict): Signal to execute
            
        Returns:
            Optional[Position]: Created position if successful
        """
        try:
            position = Position(
                ticket=f"{signal['session_id']}_{signal['signal_type']}_{len(self.positions)}",
                entry_price=signal['entry_price'],
                position_type=signal['signal_type'],
                lot_size=signal['lot_size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                entry_time=signal['timestamp'],
                london_range=signal['range_points'],
                session_id=signal['session_id']
            )
            
            self.positions.append(position)
            
            logger.info(f"Position opened: {signal['signal_type'].upper()} at {signal['entry_price']:.5f}")
            logger.info(f"  SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
            return None
    
    def update_positions(self, current_data: pd.DataFrame, current_time: datetime):
        """Update open positions with trailing stops and exit conditions"""
        if len(self.positions) == 0:
            return
        
        current_price = current_data['close'].iloc[-1]
        current_bid = current_price  # Simplified
        current_ask = current_price  # Simplified
        
        positions_to_close = []
        
        for position in self.positions:
            # Check stop loss and take profit
            if position.position_type == 'buy':
                if current_bid <= position.stop_loss or current_bid >= position.take_profit:
                    positions_to_close.append(position)
                    continue
                
                # Trailing stop logic
                if self.use_trailing:
                    profit_points = (current_bid - position.entry_price) / self.point_size
                    if profit_points >= (self.min_profit_points + self.trailing_points):
                        new_sl = current_bid - (self.trailing_points * self.point_size)
                        if new_sl > position.stop_loss:
                            position.stop_loss = new_sl
                            position.is_trailing_active = True
                            logger.info(f"Trailing stop updated for {position.ticket}: {new_sl:.5f}")
            
            else:  # sell position
                if current_ask >= position.stop_loss or current_ask <= position.take_profit:
                    positions_to_close.append(position)
                    continue
                
                # Trailing stop logic
                if self.use_trailing:
                    profit_points = (position.entry_price - current_ask) / self.point_size
                    if profit_points >= (self.min_profit_points + self.trailing_points):
                        new_sl = current_ask + (self.trailing_points * self.point_size)
                        if new_sl < position.stop_loss:
                            position.stop_loss = new_sl
                            position.is_trailing_active = True
                            logger.info(f"Trailing stop updated for {position.ticket}: {new_sl:.5f}")
        
        # Close positions that hit stop/profit targets
        for position in positions_to_close:
            self._close_position(position, current_price, current_time)
    
    def _close_position(self, position: Position, exit_price: float, exit_time: datetime):
        """Close a position and calculate P&L"""
        if position.position_type == 'buy':
            pnl = (exit_price - position.entry_price) * position.lot_size * 100000  # Simplified
        else:
            pnl = (position.entry_price - exit_price) * position.lot_size * 100000  # Simplified
        
        self.daily_pnl += pnl
        
        logger.info(f"Position closed: {position.ticket}")
        logger.info(f"  Entry: {position.entry_price:.5f}, Exit: {exit_price:.5f}")
        logger.info(f"  P&L: ${pnl:.2f}")
        
        self.positions.remove(position)
    
    def get_strategy_state(self) -> Dict:
        """Get current strategy state for monitoring"""
        return {
            'session_date': self.last_session_date,
            'session_analyzed': self.session_analysis_done,
            'no_trade_today': self.no_trade_today,
            'pre_london_high': self.pre_london_high,
            'pre_london_low': self.pre_london_low,
            'range_points': self.pre_london_range_points,
            'buy_break_level': self.buy_break_level,
            'sell_break_level': self.sell_break_level,
            'open_positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'position_details': [
                {
                    'ticket': pos.ticket,
                    'type': pos.position_type,
                    'entry_price': pos.entry_price,
                    'current_sl': pos.stop_loss,
                    'trailing_active': pos.is_trailing_active
                }
                for pos in self.positions
            ]
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.current_equity = 0.0

class LondonRangeBreakBacktester:
    """Backtesting framework for London Range Break strategy"""
    
    def __init__(self, strategy: LondonRangeBreakStrategy, initial_balance: float = 10000.0):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity_curve = []
        self.trade_log = []
        self.daily_stats = []
        
    def run_backtest(self, data: pd.DataFrame, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data (pd.DataFrame): OHLCV data with datetime index
            start_date (datetime, optional): Start date for backtest
            end_date (datetime, optional): End date for backtest
            
        Returns:
            Dict: Backtest results
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Total bars: {len(data)}")
        
        current_equity = self.initial_balance
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_data = data.iloc[:i+1]  # Data up to current point
            
            # Generate signals
            signals = self.strategy.generate_signals(current_data, timestamp)
            
            # Execute signals
            for signal in signals:
                position = self.strategy.execute_signal(signal)
                if position:
                    self.trade_log.append({
                        'timestamp': timestamp,
                        'action': 'open',
                        'position_type': signal['signal_type'],
                        'price': signal['entry_price'],
                        'lot_size': signal['lot_size'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'confidence': signal['confidence']
                    })
            
            # Update positions
            self.strategy.update_positions(current_data, timestamp)
            
            # Calculate current equity
            current_price = row['close']
            position_value = 0.0
            
            for position in self.strategy.positions:
                if position.position_type == 'buy':
                    position_pnl = (current_price - position.entry_price) * position.lot_size * 100000
                else:
                    position_pnl = (position.entry_price - current_price) * position.lot_size * 100000
                position_value += position_pnl
            
            current_equity = self.current_balance + position_value + self.strategy.daily_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'balance': self.current_balance,
                'positions': len(self.strategy.positions)
            })
        
        return self._generate_results()
    
    def _generate_results(self) -> Dict:
        """Generate backtest results summary"""
        equity_series = pd.Series([point['equity'] for point in self.equity_curve])
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] / self.initial_balance - 1) * 100
        max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
        
        # Trade statistics
        total_trades = len(self.trade_log)
        
        results = {
            'initial_balance': self.initial_balance,
            'final_equity': equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_balance,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'equity_curve': self.equity_curve,
            'trade_log': self.trade_log,
            'strategy_state': self.strategy.get_strategy_state()
        }
        
        return results