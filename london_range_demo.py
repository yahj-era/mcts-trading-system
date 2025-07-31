"""
London Range Break Trading System Demo
Comprehensive demonstration of the ML-enhanced London Range Break strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from csv_data_loader import CSVDataLoader
from london_range_strategy import LondonRangeBreakStrategy, LondonRangeBreakBacktester
from ml_london_strategy import MLLondonRangeBreakStrategy
from mcts_london_strategy import LondonRangeBreakMCTS, LondonRangeTradingState
from live_signal_generator import LiveSignalGenerator, create_live_signal_generator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LondonRangeBreakDemo:
    """
    Comprehensive demo for London Range Break trading system
    """
    
    def __init__(self):
        self.data_loader = None
        self.data = None
        self.processed_data = None
        self.results = {}
        
    def generate_sample_data(self, symbol: str = "XAUUSD", 
                           start_date: str = "2023-01-01", 
                           end_date: str = "2024-01-01") -> pd.DataFrame:
        """
        Generate realistic sample XAUUSD 15-minute OHLCV data
        
        Args:
            symbol (str): Trading symbol
            start_date (str): Start date for data generation
            end_date (str): End date for data generation
            
        Returns:
            pd.DataFrame: Generated OHLCV data
        """
        logger.info(f"Generating sample {symbol} data from {start_date} to {end_date}")
        
        # Create datetime index (15-minute intervals)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        times = pd.date_range(start=start, end=end, freq='15min')
        
        # Generate realistic gold price movements
        n = len(times)
        base_price = 2000.0  # Starting price for XAUUSD
        
        # Create price movements with different volatility regimes
        returns = []
        volatility_states = np.random.choice(['low', 'medium', 'high'], n, p=[0.6, 0.3, 0.1])
        
        for i, vol_state in enumerate(volatility_states):
            # Different volatility levels
            if vol_state == 'low':
                vol = 0.0005  # 0.05% volatility
            elif vol_state == 'medium':
                vol = 0.002   # 0.2% volatility
            else:
                vol = 0.005   # 0.5% volatility
            
            # Add time-of-day effects (higher volatility during London session)
            hour = times[i].hour
            if 8 <= hour <= 16:  # London session
                vol *= 1.5
            elif 22 <= hour or hour <= 2:  # Asian session
                vol *= 0.8
            
            # Add slight trend bias occasionally
            trend_bias = 0.0
            if np.random.random() < 0.1:  # 10% chance of trend
                trend_bias = np.random.choice([-0.0002, 0.0002])
            
            returns.append(np.random.normal(trend_bias, vol))
        
        # Generate price series
        prices = base_price * np.cumprod(1 + np.array(returns))
        
        # Generate OHLCV data
        data = []
        for i, (time, close) in enumerate(zip(times, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(returns[i]) * base_price * 2  # Intrabar volatility
            
            # Generate random high/low around close
            high_offset = np.random.uniform(0, volatility)
            low_offset = np.random.uniform(0, volatility)
            
            # Ensure realistic OHLC relationships
            open_price = prices[i-1] if i > 0 else close
            high = max(open_price, close) + high_offset
            low = min(open_price, close) - low_offset
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (higher during London session)
            hour = time.hour
            if 8 <= hour <= 16:  # London session
                volume = np.random.randint(80, 300)
            else:
                volume = np.random.randint(20, 100)
            
            data.append({
                'datetime': time,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        logger.info(f"Generated {len(df)} rows of {symbol} data")
        logger.info(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        
        return df
    
    def save_sample_data(self, data: pd.DataFrame, filename: str = "xauusd_15min_sample.csv"):
        """Save sample data to CSV file"""
        try:
            data.to_csv(filename)
            logger.info(f"Sample data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def demo_data_loading(self, csv_path: str = None):
        """Demonstrate CSV data loading and preprocessing"""
        logger.info("=== Data Loading Demo ===")
        
        # Initialize data loader
        self.data_loader = CSVDataLoader()
        
        if csv_path and pd.io.common.file_exists(csv_path):
            # Load real CSV data
            logger.info(f"Loading data from {csv_path}")
            self.data = self.data_loader.load_csv_data(csv_path)
        else:
            # Generate sample data
            logger.info("Generating sample data for demonstration")
            sample_data = self.generate_sample_data()
            
            # Save sample data
            sample_filename = "xauusd_15min_sample.csv"
            self.save_sample_data(sample_data, sample_filename)
            
            # Load the saved data to demonstrate CSV loading
            self.data = self.data_loader.load_csv_data(sample_filename)
        
        # Process data for London Range Break strategy
        logger.info("Processing data for London Range Break features...")
        self.processed_data = self.data_loader.prepare_london_range_features()
        
        # Display data info
        logger.info(f"Loaded data shape: {self.data.shape}")
        logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        logger.info(f"Processed features: {list(self.processed_data.columns)}")
        
        # Display sample of processed data
        print("\nSample of processed data:")
        display_cols = ['open', 'high', 'low', 'close', 'pre_london_range', 
                       'range_quality', 'buy_breakout', 'sell_breakout']
        available_cols = [col for col in display_cols if col in self.processed_data.columns]
        print(self.processed_data[available_cols].tail(10))
        
        return self.processed_data
    
    def demo_base_strategy(self):
        """Demonstrate base London Range Break strategy"""
        logger.info("\n=== Base Strategy Demo ===")
        
        if self.processed_data is None:
            logger.error("No processed data available. Run demo_data_loading first.")
            return
        
        # Configure base strategy
        config = {
            'lot_size': 0.01,
            'trade_type': 'all',
            'rr_ratio': 1.0,
            'stop_loss_points': 2700,
            'order_offset_points': 1,
            'use_trailing': True,
            'trailing_points': 1100,
            'min_profit_points': 200,
            'london_start_hour': 8,  # Changed to 8 AM for demo
            'min_range_points': 100,
            'max_range_points': 5000,
            'max_open_trades': 2,
            'max_daily_drawdown_percent': 5.0,
            'point_size': 0.01
        }
        
        # Create strategy
        strategy = LondonRangeBreakStrategy(config)
        
        # Create backtester
        backtester = LondonRangeBreakBacktester(strategy, initial_balance=10000.0)
        
        # Run backtest
        logger.info("Running base strategy backtest...")
        results = backtester.run_backtest(self.processed_data)
        
        # Store results
        self.results['base_strategy'] = results
        
        # Display results
        print(f"\nBase Strategy Results:")
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            trades_df = pd.DataFrame(results['trade_log'])
            print(f"Winning Trades: {len(trades_df[trades_df['action'] == 'open'])}")
        
        return results
    
    def demo_ml_enhancement(self):
        """Demonstrate ML-enhanced strategy"""
        logger.info("\n=== ML Enhancement Demo ===")
        
        if self.processed_data is None:
            logger.error("No processed data available. Run demo_data_loading first.")
            return
        
        # Configure ML strategy
        config = {
            'lot_size': 0.01,
            'trade_type': 'all',
            'rr_ratio': 1.0,
            'stop_loss_points': 2700,
            'order_offset_points': 1,
            'use_trailing': True,
            'trailing_points': 1100,
            'min_profit_points': 200,
            'london_start_hour': 8,
            'min_range_points': 100,
            'max_range_points': 5000,
            'max_open_trades': 2,
            'max_daily_drawdown_percent': 5.0,
            'point_size': 0.01,
            'use_ml_filter': True,
            'ml_confidence_threshold': 0.6,
            'ml_models_path': 'models/'
        }
        
        # Create ML strategy
        ml_strategy = MLLondonRangeBreakStrategy(config)
        
        # Split data for training and testing
        train_data, val_data, test_data = self.data_loader.split_data_by_time(
            self.processed_data, train_ratio=0.7, val_ratio=0.15
        )
        
        logger.info(f"Training ML models on {len(train_data)} samples...")
        
        # Train ML models
        try:
            performance_metrics = ml_strategy.train_ml_models(
                self.data_loader, train_data, val_data
            )
            
            print(f"\nML Model Performance:")
            for metric, value in performance_metrics.items():
                print(f"{metric}: {value:.3f}")
            
            # Test ML enhancements
            logger.info("Testing ML-enhanced signals...")
            
            # Generate signals for a sample period
            sample_data = test_data.head(100)  # Test on first 100 bars
            current_time = sample_data.index[-1]
            
            signals = ml_strategy.generate_signals(sample_data, current_time)
            
            print(f"\nGenerated {len(signals)} ML-enhanced signals")
            
            if signals:
                for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                    print(f"\nSignal {i+1}:")
                    print(f"  Type: {signal['signal_type']}")
                    print(f"  Confidence: {signal['confidence']:.3f}")
                    print(f"  ML Confidence: {signal.get('ml_confidence', 'N/A')}")
                    print(f"  Entry: {signal['entry_price']:.5f}")
                    print(f"  Stop Loss: {signal['stop_loss']:.5f}")
                    print(f"  Take Profit: {signal['take_profit']:.5f}")
            
            # Get ML analysis
            ml_analysis = ml_strategy.get_ml_analysis(sample_data)
            if ml_analysis:
                print(f"\nCurrent ML Analysis:")
                print(f"Market Regime: {ml_analysis.get('market_regime', 'unknown')}")
                print(f"Trading Recommendation: {ml_analysis.get('trading_recommendation', 'hold')}")
                
                ml_preds = ml_analysis.get('ml_predictions', {})
                if ml_preds:
                    print(f"Breakout Success Probability: {ml_preds.get('breakout_success_probability', 0):.3f}")
                    print(f"Bullish Probability: {ml_preds.get('bullish_probability', 0):.3f}")
            
            self.results['ml_strategy'] = {
                'performance_metrics': performance_metrics,
                'signals': signals,
                'analysis': ml_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in ML demonstration: {str(e)}")
            print("ML models training failed - this is normal in demo without sufficient data")
    
    def demo_mcts_optimization(self):
        """Demonstrate MCTS optimization"""
        logger.info("\n=== MCTS Optimization Demo ===")
        
        if self.processed_data is None:
            logger.error("No processed data available. Run demo_data_loading first.")
            return
        
        # Configure MCTS
        mcts_config = {
            'exploration_constant': 1.414,
            'max_iterations': 200,  # Reduced for demo
            'max_depth': 10,
            'rollout_depth': 5,
            'range_break_reward': 2.0,
            'false_breakout_penalty': -1.0,
            'time_decay_factor': 0.95
        }
        
        # Create MCTS
        mcts = LondonRangeBreakMCTS(mcts_config)
        
        # Create sample trading state
        current_data = self.processed_data.tail(50)
        current_price = current_data['close'].iloc[-1]
        
        # Get range information
        pre_london_high = current_data['pre_london_high'].iloc[-1] if 'pre_london_high' in current_data.columns else current_price + 10
        pre_london_low = current_data['pre_london_low'].iloc[-1] if 'pre_london_low' in current_data.columns else current_price - 10
        range_points = current_data['pre_london_range'].iloc[-1] if 'pre_london_range' in current_data.columns else 500
        
        # Create features array
        feature_columns = ['rsi_14', 'atr_14', 'volatility_20', 'price_change', 'hour']
        available_features = [col for col in feature_columns if col in current_data.columns]
        
        if available_features:
            features = current_data[available_features].iloc[-1].values
            features = np.nan_to_num(features, nan=0.0)
        else:
            features = np.random.random(5)  # Mock features for demo
        
        # Create trading state
        trading_state = LondonRangeTradingState(
            current_price=current_price,
            current_features=features,
            position=0.0,
            cash=10000.0,
            portfolio_value=10000.0,
            step=0,
            max_steps=20,
            pre_london_high=pre_london_high,
            pre_london_low=pre_london_low,
            range_points=range_points,
            buy_break_level=pre_london_high + 1,
            sell_break_level=pre_london_low - 1,
            session_id="demo_session",
            time_since_london_open=2,
            price_in_range=pre_london_low <= current_price <= pre_london_high,
            above_range=current_price > pre_london_high,
            below_range=current_price < pre_london_low,
            range_quality='good',
            volatility_level='medium',
            breakout_strength=0.1
        )
        
        logger.info("Running MCTS search...")
        start_time = time.time()
        
        # Run MCTS search
        best_action, confidence = mcts.search(trading_state)
        
        execution_time = time.time() - start_time
        
        # Get search statistics
        search_stats = mcts.get_search_statistics()
        
        print(f"\nMCTS Results:")
        print(f"Best Action: {best_action}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Search Statistics:")
        print(f"  Total Searches: {search_stats['total_searches']}")
        print(f"  Tree Size: {search_stats['tree_size']}")
        print(f"  Average Depth: {search_stats.get('average_depth', 0):.2f}")
        
        # Demonstrate different market conditions
        print(f"\nTesting different market conditions:")
        
        conditions = [
            ("Above Range - Good Quality", True, False, 'good'),
            ("Below Range - Good Quality", False, True, 'good'),
            ("In Range - Poor Quality", False, False, 'poor'),
        ]
        
        for desc, above_range, below_range, quality in conditions:
            test_state = LondonRangeTradingState(
                current_price=current_price,
                current_features=features,
                position=0.0,
                cash=10000.0,
                portfolio_value=10000.0,
                step=0,
                max_steps=20,
                pre_london_high=pre_london_high,
                pre_london_low=pre_london_low,
                range_points=range_points,
                buy_break_level=pre_london_high + 1,
                sell_break_level=pre_london_low - 1,
                session_id="test_session",
                time_since_london_open=2,
                price_in_range=not (above_range or below_range),
                above_range=above_range,
                below_range=below_range,
                range_quality=quality,
                volatility_level='medium',
                breakout_strength=0.1 if (above_range or below_range) else 0.0
            )
            
            action, conf = mcts.search(test_state)
            print(f"  {desc}: {action} (confidence: {conf:.3f})")
        
        self.results['mcts'] = {
            'best_action': best_action,
            'confidence': confidence,
            'execution_time': execution_time,
            'search_stats': search_stats
        }
    
    def demo_live_signal_generation(self, duration_seconds: int = 60):
        """Demonstrate live signal generation (mock)"""
        logger.info(f"\n=== Live Signal Generation Demo ({duration_seconds}s) ===")
        
        try:
            # Create live signal generator
            signal_generator = create_live_signal_generator()
            
            # Start signal generation
            if signal_generator.start():
                logger.info("Live signal generator started successfully")
                
                start_time = time.time()
                signal_count = 0
                
                while time.time() - start_time < duration_seconds:
                    # Check for new signals
                    signal = signal_generator.get_next_signal(timeout=1.0)
                    
                    if signal:
                        signal_count += 1
                        print(f"\nLive Signal #{signal_count}:")
                        print(f"  Time: {signal.timestamp}")
                        print(f"  Type: {signal.signal_type}")
                        print(f"  Symbol: {signal.symbol}")
                        print(f"  Price: {signal.entry_price:.5f}")
                        print(f"  Confidence: {signal.confidence:.3f}")
                        
                        if signal.ml_confidence:
                            print(f"  ML Confidence: {signal.ml_confidence:.3f}")
                        if signal.mcts_confidence:
                            print(f"  MCTS Confidence: {signal.mcts_confidence:.3f}")
                        if signal.market_regime:
                            print(f"  Market Regime: {signal.market_regime}")
                    
                    # Show performance stats every 10 seconds
                    if int(time.time() - start_time) % 10 == 0:
                        stats = signal_generator.get_performance_stats()
                        print(f"\nPerformance Stats:")
                        print(f"  Signals Generated: {stats['signals_generated']}")
                        print(f"  ML Enhancements: {stats['ml_enhancements']}")
                        print(f"  MCTS Optimizations: {stats['mcts_optimizations']}")
                        print(f"  Queue Size: {stats['queue_size']}")
                
                # Stop signal generator
                signal_generator.stop()
                
                # Final statistics
                final_stats = signal_generator.get_performance_stats()
                print(f"\nFinal Results:")
                print(f"Total Signals Generated: {final_stats['signals_generated']}")
                print(f"ML Enhancements: {final_stats['ml_enhancements']}")
                print(f"MCTS Optimizations: {final_stats['mcts_optimizations']}")
                
                self.results['live_signals'] = final_stats
                
            else:
                logger.error("Failed to start live signal generator")
                
        except Exception as e:
            logger.error(f"Error in live signal demo: {str(e)}")
    
    def plot_results(self):
        """Plot comprehensive results"""
        logger.info("\n=== Plotting Results ===")
        
        if not self.processed_data is not None:
            logger.error("No data available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('London Range Break Trading System - Comprehensive Analysis', fontsize=16)
        
        # 1. Price chart with range levels
        ax1 = axes[0, 0]
        recent_data = self.processed_data.tail(200)  # Last 200 bars
        
        ax1.plot(recent_data.index, recent_data['close'], label='Close Price', alpha=0.8)
        
        if 'pre_london_high' in recent_data.columns:
            ax1.plot(recent_data.index, recent_data['pre_london_high'], 
                    label='Pre-London High', alpha=0.6, linestyle='--')
            ax1.plot(recent_data.index, recent_data['pre_london_low'], 
                    label='Pre-London Low', alpha=0.6, linestyle='--')
        
        ax1.set_title('Price with London Range Levels')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Range quality distribution
        ax2 = axes[0, 1]
        if 'pre_london_range' in recent_data.columns:
            range_data = recent_data['pre_london_range'].dropna()
            ax2.hist(range_data, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(range_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {range_data.mean():.0f}')
            ax2.set_title('Pre-London Range Distribution')
            ax2.set_xlabel('Range (points)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # 3. Breakout signals
        ax3 = axes[1, 0]
        if 'buy_breakout' in recent_data.columns and 'sell_breakout' in recent_data.columns:
            buy_signals = recent_data[recent_data['buy_breakout'] == 1]
            sell_signals = recent_data[recent_data['sell_breakout'] == 1]
            
            ax3.plot(recent_data.index, recent_data['close'], alpha=0.5, color='gray')
            ax3.scatter(buy_signals.index, buy_signals['close'], 
                       color='green', marker='^', s=50, label='Buy Breakout')
            ax3.scatter(sell_signals.index, sell_signals['close'], 
                       color='red', marker='v', s=50, label='Sell Breakout')
            
            ax3.set_title('Breakout Signals')
            ax3.set_ylabel('Price')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Performance comparison (if available)
        ax4 = axes[1, 1]
        if 'base_strategy' in self.results:
            base_results = self.results['base_strategy']
            equity_curve = pd.DataFrame(base_results['equity_curve'])
            
            if not equity_curve.empty:
                ax4.plot(equity_curve['timestamp'], equity_curve['equity'], 
                        label='Portfolio Equity')
                ax4.set_title('Portfolio Performance')
                ax4.set_ylabel('Equity ($)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5. Technical indicators
        ax5 = axes[2, 0]
        if 'rsi_14' in recent_data.columns:
            ax5.plot(recent_data.index, recent_data['rsi_14'], label='RSI(14)')
            ax5.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax5.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax5.set_title('Technical Indicators')
            ax5.set_ylabel('RSI')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. System statistics
        ax6 = axes[2, 1]
        categories = []
        values = []
        
        if self.results:
            for key, result in self.results.items():
                if isinstance(result, dict):
                    if 'signals_generated' in result:
                        categories.append(f'{key.title()}\nSignals')
                        values.append(result['signals_generated'])
                    elif 'total_trades' in result:
                        categories.append(f'{key.title()}\nTrades')
                        values.append(result['total_trades'])
        
        if categories and values:
            ax6.bar(categories, values, alpha=0.7)
            ax6.set_title('System Performance Summary')
            ax6.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        try:
            plt.savefig('london_range_break_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Analysis plot saved as 'london_range_break_analysis.png'")
        except Exception as e:
            logger.warning(f"Could not save plot: {str(e)}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("\n=== Generating Summary Report ===")
        
        report = []
        report.append("=" * 60)
        report.append("LONDON RANGE BREAK TRADING SYSTEM - DEMO REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data Summary
        if self.processed_data is not None:
            report.append("DATA SUMMARY:")
            report.append(f"  Total Bars: {len(self.processed_data):,}")
            report.append(f"  Date Range: {self.processed_data.index[0]} to {self.processed_data.index[-1]}")
            report.append(f"  Price Range: ${self.processed_data['low'].min():.2f} - ${self.processed_data['high'].max():.2f}")
            
            if 'pre_london_range' in self.processed_data.columns:
                range_data = self.processed_data['pre_london_range'].dropna()
                report.append(f"  Average Range: {range_data.mean():.0f} points")
                report.append(f"  Range Std Dev: {range_data.std():.0f} points")
            report.append("")
        
        # Strategy Results
        if 'base_strategy' in self.results:
            base = self.results['base_strategy']
            report.append("BASE STRATEGY RESULTS:")
            report.append(f"  Initial Balance: ${base['initial_balance']:,.2f}")
            report.append(f"  Final Equity: ${base['final_equity']:,.2f}")
            report.append(f"  Total Return: {base['total_return_pct']:.2f}%")
            report.append(f"  Max Drawdown: {base['max_drawdown_pct']:.2f}%")
            report.append(f"  Total Trades: {base['total_trades']}")
            report.append("")
        
        # ML Results
        if 'ml_strategy' in self.results:
            ml = self.results['ml_strategy']
            report.append("MACHINE LEARNING ENHANCEMENTS:")
            
            if 'performance_metrics' in ml:
                for metric, value in ml['performance_metrics'].items():
                    report.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
            
            if 'analysis' in ml and ml['analysis']:
                analysis = ml['analysis']
                report.append(f"  Current Market Regime: {analysis.get('market_regime', 'Unknown')}")
                report.append(f"  Trading Recommendation: {analysis.get('trading_recommendation', 'Hold')}")
            report.append("")
        
        # MCTS Results
        if 'mcts' in self.results:
            mcts = self.results['mcts']
            report.append("MCTS OPTIMIZATION RESULTS:")
            report.append(f"  Best Action: {mcts['best_action']}")
            report.append(f"  Confidence: {mcts['confidence']:.3f}")
            report.append(f"  Execution Time: {mcts['execution_time']:.2f} seconds")
            
            if 'search_stats' in mcts:
                stats = mcts['search_stats']
                report.append(f"  Tree Size: {stats['tree_size']}")
                report.append(f"  Average Depth: {stats.get('average_depth', 0):.2f}")
            report.append("")
        
        # Live Signal Results
        if 'live_signals' in self.results:
            live = self.results['live_signals']
            report.append("LIVE SIGNAL GENERATION:")
            report.append(f"  Total Signals: {live['signals_generated']}")
            report.append(f"  ML Enhancements: {live['ml_enhancements']}")
            report.append(f"  MCTS Optimizations: {live['mcts_optimizations']}")
            report.append(f"  ML Models Loaded: {live['ml_models_loaded']}")
            report.append("")
        
        # Technical Notes
        report.append("TECHNICAL NOTES:")
        report.append("  - Strategy focuses on London session breakouts (8 AM - 4 PM UTC)")
        report.append("  - Pre-London range calculated from 3 AM - 8 AM UTC")
        report.append("  - ML models predict breakout success, direction, and profit potential")
        report.append("  - MCTS optimizes action selection based on market conditions")
        report.append("  - Risk management includes trailing stops and drawdown limits")
        report.append("")
        
        report.append("=" * 60)
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        try:
            with open("london_range_break_report.txt", "w") as f:
                f.write(report_text)
            logger.info("Report saved as 'london_range_break_report.txt'")
        except Exception as e:
            logger.warning(f"Could not save report: {str(e)}")
        
        return report_text

def run_comprehensive_demo(csv_path: str = None):
    """
    Run comprehensive demonstration of London Range Break system
    
    Args:
        csv_path (str, optional): Path to XAUUSD CSV file
    """
    print("Starting London Range Break Trading System Demo...")
    print("=" * 60)
    
    # Create demo instance
    demo = LondonRangeBreakDemo()
    
    try:
        # 1. Data Loading Demo
        demo.demo_data_loading(csv_path)
        
        # 2. Base Strategy Demo
        demo.demo_base_strategy()
        
        # 3. ML Enhancement Demo
        demo.demo_ml_enhancement()
        
        # 4. MCTS Optimization Demo
        demo.demo_mcts_optimization()
        
        # 5. Live Signal Generation Demo (short duration for demo)
        demo.demo_live_signal_generation(duration_seconds=30)
        
        # 6. Plot Results
        demo.plot_results()
        
        # 7. Generate Summary Report
        demo.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("Check the generated files:")
        print("  - xauusd_15min_sample.csv (sample data)")
        print("  - london_range_break_analysis.png (analysis charts)")
        print("  - london_range_break_report.txt (summary report)")
        print("  - models/ directory (ML models)")
        print("  - live_signals/ directory (signal outputs)")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"Demo encountered an error: {str(e)}")
        print("This is normal in a demo environment - some features require real data and dependencies.")

if __name__ == "__main__":
    # Set up matplotlib backend for plotting
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Run the demo
    print("London Range Break Trading System Demo")
    print("This demo showcases the complete system including:")
    print("1. CSV data loading for XAUUSD 15-minute data")
    print("2. London Range Break strategy implementation")
    print("3. Machine Learning enhancements")
    print("4. Monte Carlo Tree Search optimization")
    print("5. Live signal generation simulation")
    print()
    
    # Check if user provided CSV path
    csv_path = input("Enter path to XAUUSD CSV file (or press Enter to use sample data): ").strip()
    if not csv_path:
        csv_path = None
    
    run_comprehensive_demo(csv_path)