"""
Live Signal Generator for MT5 Integration
Generates real-time trading signals using London Range Break strategy with ML and MCTS
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
import json
import os
from dataclasses import dataclass, asdict

# Import our custom modules
from csv_data_loader import CSVDataLoader
from london_range_strategy import LondonRangeBreakStrategy
from ml_london_strategy import MLLondonRangeBreakStrategy
from mcts_london_strategy import LondonRangeBreakMCTS, LondonRangeTradingState

logger = logging.getLogger(__name__)

@dataclass
class LiveSignal:
    """Live trading signal structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold', 'close'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    strategy: str
    session_id: str
    
    # ML enhancements
    ml_confidence: Optional[float] = None
    breakout_probability: Optional[float] = None
    direction_probability: Optional[float] = None
    profit_category: Optional[int] = None
    
    # MCTS enhancements
    mcts_confidence: Optional[float] = None
    expected_reward: Optional[float] = None
    
    # Range information
    pre_london_high: Optional[float] = None
    pre_london_low: Optional[float] = None
    range_points: Optional[float] = None
    
    # Market analysis
    market_regime: Optional[str] = None
    trading_recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_mt5_format(self) -> Dict:
        """Convert signal to MT5-compatible format"""
        return {
            'symbol': self.symbol,
            'action': self.signal_type.upper(),
            'volume': self.lot_size,
            'price': self.entry_price,
            'sl': self.stop_loss,
            'tp': self.take_profit,
            'magic': 1234567890,
            'comment': f"LRB_{self.strategy}_{self.confidence:.2f}",
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }

class MT5DataProvider:
    """
    Mock MT5 data provider - replace with actual MT5 connection
    In real implementation, this would connect to MetaTrader 5
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.is_connected = False
        self.last_tick = None
        
    def connect(self) -> bool:
        """Connect to MT5 (mock implementation)"""
        try:
            # In real implementation, use MT5 library:
            # import MetaTrader5 as mt5
            # self.is_connected = mt5.initialize()
            
            self.is_connected = True  # Mock connection
            logger.info(f"Connected to MT5 for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        # mt5.shutdown()
        self.is_connected = False
        logger.info("Disconnected from MT5")
    
    def get_current_tick(self) -> Optional[Dict]:
        """Get current tick data (mock implementation)"""
        if not self.is_connected:
            return None
        
        # In real implementation:
        # tick = mt5.symbol_info_tick(self.symbol)
        # Mock tick data
        current_time = datetime.now()
        if self.last_tick is None:
            price = 2000.0  # Starting price for XAUUSD
        else:
            # Simulate price movement
            change = np.random.normal(0, 0.5)  # Random walk
            price = max(self.last_tick['bid'] + change, 1000.0)
        
        tick = {
            'time': current_time,
            'bid': price,
            'ask': price + 0.3,  # Spread
            'last': price,
            'volume': 100
        }
        
        self.last_tick = tick
        return tick
    
    def get_historical_data(self, timeframe: str = "M15", count: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data (mock implementation)"""
        if not self.is_connected:
            return None
        
        # In real implementation:
        # rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, count)
        
        # Mock historical data generation
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=count * 15)
        
        # Generate mock OHLCV data
        times = pd.date_range(start=start_time, end=end_time, freq='15min')[:-1]
        n = len(times)
        
        # Generate realistic price movements
        base_price = 2000.0
        returns = np.random.normal(0, 0.001, n)  # 0.1% volatility
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV
        data = []
        for i, (time, close) in enumerate(zip(times, prices)):
            volatility = np.random.uniform(0.5, 2.0)
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'datetime': time,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': np.random.randint(50, 200)
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        return df

class LiveSignalGenerator:
    """
    Live signal generator for London Range Break strategy
    Integrates base strategy, ML models, and MCTS for optimal signal generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live signal generator
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config
        self.symbol = config.get('symbol', 'XAUUSD')
        self.timeframe = config.get('timeframe', 'M15')
        self.update_interval = config.get('update_interval', 60)  # seconds
        
        # Components
        self.data_provider = MT5DataProvider(self.symbol)
        self.data_loader = CSVDataLoader()
        
        # Strategy components
        self.base_strategy = LondonRangeBreakStrategy(config.get('strategy_config', {}))
        self.ml_strategy = MLLondonRangeBreakStrategy(config.get('ml_strategy_config', {}))
        self.mcts = LondonRangeBreakMCTS(config.get('mcts_config', {}))
        
        # Signal management
        self.signal_queue = queue.Queue()
        self.latest_signals = []
        self.signal_history = []
        
        # State management
        self.is_running = False
        self.current_data = None
        self.last_update = None
        self.market_hours_only = config.get('market_hours_only', True)
        
        # Performance tracking
        self.performance_stats = {
            'signals_generated': 0,
            'ml_enhancements': 0,
            'mcts_optimizations': 0,
            'execution_times': []
        }
        
        # File outputs
        self.output_dir = config.get('output_dir', 'signals/')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load ML models if available
        self.ml_models_loaded = self.ml_strategy.load_models()
        
    def start(self) -> bool:
        """Start the live signal generation"""
        logger.info("Starting live signal generator...")
        
        # Connect to data provider
        if not self.data_provider.connect():
            logger.error("Failed to connect to data provider")
            return False
        
        # Load initial historical data
        if not self._load_initial_data():
            logger.error("Failed to load initial data")
            return False
        
        self.is_running = True
        
        # Start signal generation thread
        self.signal_thread = threading.Thread(target=self._signal_generation_loop)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        logger.info("Live signal generator started successfully")
        return True
    
    def stop(self):
        """Stop the live signal generation"""
        logger.info("Stopping live signal generator...")
        
        self.is_running = False
        
        if hasattr(self, 'signal_thread'):
            self.signal_thread.join(timeout=5)
        
        self.data_provider.disconnect()
        
        # Save final performance stats
        self._save_performance_stats()
        
        logger.info("Live signal generator stopped")
    
    def _load_initial_data(self) -> bool:
        """Load initial historical data"""
        try:
            # Get historical data from MT5
            historical_data = self.data_provider.get_historical_data(
                timeframe=self.timeframe, 
                count=1000
            )
            
            if historical_data is None or len(historical_data) == 0:
                logger.error("No historical data available")
                return False
            
            # Process data for strategy use
            self.data_loader.data = historical_data
            self.current_data = self.data_loader.prepare_london_range_features()
            
            logger.info(f"Loaded {len(historical_data)} historical bars")
            return True
            
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")
            return False
    
    def _signal_generation_loop(self):
        """Main signal generation loop"""
        logger.info("Signal generation loop started")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update market data
                if self._update_market_data():
                    # Check if we should generate signals (market hours, etc.)
                    if self._should_generate_signals():
                        # Generate signals
                        signals = self._generate_comprehensive_signals()
                        
                        # Process and queue signals
                        for signal in signals:
                            self._process_signal(signal)
                    
                # Record execution time
                execution_time = time.time() - start_time
                self.performance_stats['execution_times'].append(execution_time)
                
                # Sleep until next update
                sleep_time = max(0, self.update_interval - execution_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def _update_market_data(self) -> bool:
        """Update market data with latest tick"""
        try:
            # Get current tick
            tick = self.data_provider.get_current_tick()
            if tick is None:
                return False
            
            # Convert tick to OHLCV format and append to data
            current_time = tick['time']
            
            # Round to nearest 15-minute interval
            rounded_time = current_time.replace(
                minute=(current_time.minute // 15) * 15,
                second=0,
                microsecond=0
            )
            
            # Create new bar or update existing
            if self.current_data is None or len(self.current_data) == 0:
                return False
            
            last_bar_time = self.current_data.index[-1]
            
            if rounded_time > last_bar_time:
                # New bar
                new_bar = {
                    'open': tick['last'],
                    'high': tick['last'],
                    'low': tick['last'],
                    'close': tick['last'],
                    'volume': tick['volume']
                }
                
                # Add to dataframe
                new_data = pd.DataFrame([new_bar], index=[rounded_time])
                self.current_data = pd.concat([self.current_data, new_data])
                
                # Keep only recent data (last 1000 bars)
                if len(self.current_data) > 1000:
                    self.current_data = self.current_data.tail(1000)
                
                # Reprocess features
                self.current_data = self.data_loader.prepare_london_range_features(
                    self.current_data
                )
                
                self.last_update = current_time
                return True
            
            else:
                # Update current bar
                if rounded_time == last_bar_time:
                    self.current_data.loc[last_bar_time, 'high'] = max(
                        self.current_data.loc[last_bar_time, 'high'],
                        tick['last']
                    )
                    self.current_data.loc[last_bar_time, 'low'] = min(
                        self.current_data.loc[last_bar_time, 'low'],
                        tick['last']
                    )
                    self.current_data.loc[last_bar_time, 'close'] = tick['last']
                    self.current_data.loc[last_bar_time, 'volume'] += tick['volume']
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            return False
    
    def _should_generate_signals(self) -> bool:
        """Check if we should generate signals based on market conditions"""
        current_time = datetime.now()
        
        # Market hours check (if enabled)
        if self.market_hours_only:
            # London session: 8 AM - 4 PM UTC
            hour = current_time.hour
            if not (8 <= hour < 16):
                return False
        
        # Don't generate signals too frequently
        if (self.last_update and 
            (current_time - self.last_update).total_seconds() < 60):
            return False
        
        # Check if we have enough data
        if self.current_data is None or len(self.current_data) < 100:
            return False
        
        return True
    
    def _generate_comprehensive_signals(self) -> List[LiveSignal]:
        """Generate signals using all available methods"""
        signals = []
        current_time = datetime.now()
        
        try:
            # 1. Base strategy signals
            base_signals = self.base_strategy.generate_signals(
                self.current_data, current_time
            )
            
            for base_signal in base_signals:
                # Create live signal
                live_signal = LiveSignal(
                    timestamp=current_time,
                    symbol=self.symbol,
                    signal_type=base_signal['signal_type'],
                    confidence=base_signal['confidence'],
                    entry_price=base_signal['entry_price'],
                    stop_loss=base_signal['stop_loss'],
                    take_profit=base_signal['take_profit'],
                    lot_size=base_signal['lot_size'],
                    strategy='base_london_range_break',
                    session_id=base_signal['session_id'],
                    pre_london_high=base_signal.get('pre_london_high'),
                    pre_london_low=base_signal.get('pre_london_low'),
                    range_points=base_signal.get('range_points')
                )
                
                # 2. ML Enhancement
                if self.ml_models_loaded:
                    live_signal = self._enhance_with_ml(live_signal, base_signal)
                
                # 3. MCTS Optimization
                live_signal = self._enhance_with_mcts(live_signal)
                
                signals.append(live_signal)
                self.performance_stats['signals_generated'] += 1
        
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
        
        return signals
    
    def _enhance_with_ml(self, signal: LiveSignal, base_signal: Dict) -> LiveSignal:
        """Enhance signal with ML predictions"""
        try:
            # Get ML analysis
            ml_analysis = self.ml_strategy.get_ml_analysis(self.current_data)
            
            if ml_analysis:
                ml_predictions = ml_analysis.get('ml_predictions', {})
                
                # Update signal with ML data
                signal.ml_confidence = ml_predictions.get('breakout_success_probability')
                signal.breakout_probability = ml_predictions.get('breakout_success_probability')
                signal.direction_probability = ml_predictions.get('bullish_probability')
                signal.profit_category = ml_predictions.get('profit_category')
                signal.market_regime = ml_analysis.get('market_regime')
                signal.trading_recommendation = ml_analysis.get('trading_recommendation')
                
                # Recalculate confidence using ML
                if signal.ml_confidence is not None:
                    signal.confidence = 0.6 * signal.ml_confidence + 0.4 * signal.confidence
                
                self.performance_stats['ml_enhancements'] += 1
        
        except Exception as e:
            logger.error(f"Error enhancing signal with ML: {str(e)}")
        
        return signal
    
    def _enhance_with_mcts(self, signal: LiveSignal) -> LiveSignal:
        """Enhance signal with MCTS optimization"""
        try:
            # Create MCTS state from current market conditions
            current_price = self.current_data['close'].iloc[-1]
            current_features = self._get_current_features()
            
            if current_features is not None and len(current_features) > 0:
                # Create trading state
                trading_state = LondonRangeTradingState(
                    current_price=current_price,
                    current_features=current_features,
                    position=0.0,  # Assume flat
                    cash=10000.0,  # Starting cash
                    portfolio_value=10000.0,
                    step=0,
                    max_steps=32,  # 8 hours * 4 (15-min bars)
                    pre_london_high=signal.pre_london_high or current_price,
                    pre_london_low=signal.pre_london_low or current_price,
                    range_points=signal.range_points or 0,
                    buy_break_level=signal.pre_london_high or current_price,
                    sell_break_level=signal.pre_london_low or current_price,
                    session_id=signal.session_id,
                    time_since_london_open=1,  # Simplified
                    price_in_range=False,
                    above_range=False,
                    below_range=False,
                    range_quality='good',
                    volatility_level='medium',
                    breakout_strength=0.0
                )
                
                # Run MCTS search
                ml_predictor = self.ml_strategy.direction_model if self.ml_models_loaded else None
                best_action, mcts_confidence = self.mcts.search(trading_state, ml_predictor)
                
                # Update signal with MCTS results
                signal.mcts_confidence = mcts_confidence
                
                # Adjust signal based on MCTS recommendation
                if best_action == 'hold' and mcts_confidence > 0.7:
                    signal.signal_type = 'hold'
                    signal.confidence *= 0.5  # Reduce confidence for hold signals
                elif best_action in ['buy_aggressive', 'sell_aggressive'] and mcts_confidence > 0.8:
                    signal.confidence = min(1.0, signal.confidence * 1.2)  # Boost confidence
                
                self.performance_stats['mcts_optimizations'] += 1
        
        except Exception as e:
            logger.error(f"Error enhancing signal with MCTS: {str(e)}")
        
        return signal
    
    def _get_current_features(self) -> Optional[np.ndarray]:
        """Get current market features for MCTS"""
        try:
            if self.current_data is None or len(self.current_data) == 0:
                return None
            
            # Get features from last row
            feature_columns = [
                'pre_london_range', 'range_quality', 'price_in_range_pct',
                'volatility_vs_range', 'rsi_14', 'atr_14', 'volatility_20',
                'price_change', 'hour', 'day_of_week', 'is_london_session'
            ]
            
            available_features = [col for col in feature_columns 
                                if col in self.current_data.columns]
            
            if not available_features:
                return None
            
            features = self.current_data[available_features].iloc[-1].values
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting current features: {str(e)}")
            return None
    
    def _process_signal(self, signal: LiveSignal):
        """Process and queue a signal"""
        try:
            # Add to signal queue
            self.signal_queue.put(signal)
            
            # Update latest signals list
            self.latest_signals.append(signal)
            if len(self.latest_signals) > 10:  # Keep only last 10 signals
                self.latest_signals.pop(0)
            
            # Add to history
            self.signal_history.append(signal)
            
            # Save signal to file
            self._save_signal_to_file(signal)
            
            # Log signal
            logger.info(f"Signal generated: {signal.signal_type.upper()} "
                       f"{signal.symbol} @ {signal.entry_price:.5f} "
                       f"(Confidence: {signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    def _save_signal_to_file(self, signal: LiveSignal):
        """Save signal to JSON file"""
        try:
            filename = f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Load existing signals
            signals_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    signals_data = json.load(f)
            
            # Add new signal
            signals_data.append(signal.to_dict())
            
            # Save back to file
            with open(filepath, 'w') as f:
                json.dump(signals_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving signal to file: {str(e)}")
    
    def _save_performance_stats(self):
        """Save performance statistics"""
        try:
            stats_file = os.path.join(self.output_dir, 'performance_stats.json')
            
            # Calculate additional stats
            if self.performance_stats['execution_times']:
                avg_execution_time = np.mean(self.performance_stats['execution_times'])
                max_execution_time = np.max(self.performance_stats['execution_times'])
            else:
                avg_execution_time = 0
                max_execution_time = 0
            
            stats = {
                'total_signals': self.performance_stats['signals_generated'],
                'ml_enhancements': self.performance_stats['ml_enhancements'],
                'mcts_optimizations': self.performance_stats['mcts_optimizations'],
                'average_execution_time': avg_execution_time,
                'max_execution_time': max_execution_time,
                'ml_models_loaded': self.ml_models_loaded,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving performance stats: {str(e)}")
    
    def get_latest_signals(self, count: int = 5) -> List[LiveSignal]:
        """Get latest signals"""
        return self.latest_signals[-count:] if self.latest_signals else []
    
    def get_signal_queue_size(self) -> int:
        """Get current signal queue size"""
        return self.signal_queue.qsize()
    
    def get_next_signal(self, timeout: Optional[float] = None) -> Optional[LiveSignal]:
        """Get next signal from queue"""
        try:
            return self.signal_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        if stats['execution_times']:
            stats['avg_execution_time'] = np.mean(stats['execution_times'])
            stats['max_execution_time'] = np.max(stats['execution_times'])
        else:
            stats['avg_execution_time'] = 0
            stats['max_execution_time'] = 0
        
        stats['ml_models_loaded'] = self.ml_models_loaded
        stats['queue_size'] = self.signal_queue.qsize()
        stats['latest_signals_count'] = len(self.latest_signals)
        
        return stats
    
    def get_market_analysis(self) -> Dict:
        """Get current market analysis"""
        if self.current_data is None:
            return {}
        
        try:
            # Get base strategy state
            strategy_state = self.base_strategy.get_strategy_state()
            
            # Get ML analysis if available
            ml_analysis = {}
            if self.ml_models_loaded:
                ml_analysis = self.ml_strategy.get_ml_analysis(self.current_data)
            
            # Get MCTS statistics
            mcts_stats = self.mcts.get_search_statistics()
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'current_price': self.current_data['close'].iloc[-1],
                'strategy_state': strategy_state,
                'ml_analysis': ml_analysis,
                'mcts_stats': mcts_stats,
                'data_points': len(self.current_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return {}

# Example usage and configuration
def create_live_signal_generator() -> LiveSignalGenerator:
    """Create and configure live signal generator"""
    
    config = {
        'symbol': 'XAUUSD',
        'timeframe': 'M15',
        'update_interval': 60,  # seconds
        'market_hours_only': True,
        'output_dir': 'live_signals/',
        
        'strategy_config': {
            'lot_size': 0.01,
            'trade_type': 'all',
            'rr_ratio': 1.0,
            'stop_loss_points': 2700,
            'order_offset_points': 1,
            'use_trailing': True,
            'trailing_points': 1100,
            'min_profit_points': 200,
            'london_start_hour': 9,
            'min_range_points': 100,
            'max_range_points': 5000,
            'max_open_trades': 2,
            'max_daily_drawdown_percent': 5.0,
            'point_size': 0.01
        },
        
        'ml_strategy_config': {
            'use_ml_filter': True,
            'ml_confidence_threshold': 0.6,
            'ml_models_path': 'models/'
        },
        
        'mcts_config': {
            'exploration_constant': 1.414,
            'max_iterations': 500,  # Reduced for live trading
            'max_depth': 15,
            'rollout_depth': 8,
            'range_break_reward': 2.0,
            'false_breakout_penalty': -1.0,
            'time_decay_factor': 0.95
        }
    }
    
    return LiveSignalGenerator(config)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create signal generator
    signal_generator = create_live_signal_generator()
    
    # Start signal generation
    if signal_generator.start():
        try:
            # Run for demonstration
            while True:
                # Get latest signals
                latest_signals = signal_generator.get_latest_signals()
                if latest_signals:
                    for signal in latest_signals[-1:]:  # Show only latest
                        print(f"Latest Signal: {signal.signal_type} {signal.symbol} "
                              f"@ {signal.entry_price:.5f} (Confidence: {signal.confidence:.2f})")
                
                # Get market analysis
                analysis = signal_generator.get_market_analysis()
                if analysis:
                    print(f"Market Analysis: {analysis.get('strategy_state', {}).get('range_points', 'N/A')} points")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\nStopping signal generator...")
            signal_generator.stop()
    else:
        print("Failed to start signal generator")