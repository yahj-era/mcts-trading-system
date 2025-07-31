# London Range Break Trading System

## Overview

This is a comprehensive ML-enhanced London Range Break trading system that has been upgraded from the original stock trading system to specifically handle XAUUSD (Gold) 15-minute data with advanced machine learning and Monte Carlo Tree Search (MCTS) optimization.

## Key Features

### 🎯 Strategy Focus
- **London Range Break Strategy**: Identifies pre-London session ranges (3 AM - 8 AM UTC) and trades breakouts during London session (8 AM - 4 PM UTC)
- **XAUUSD Specialization**: Optimized for Gold trading with appropriate point sizes and volatility handling
- **Session-Based Trading**: Focuses on high-probability breakouts during London market hours

### 🤖 Machine Learning Integration
- **Breakout Success Prediction**: XGBoost model predicts probability of successful breakouts
- **Direction Forecasting**: Random Forest model predicts price direction
- **Profit Categorization**: Gradient Boosting model estimates profit potential
- **Real-time Analysis**: ML models provide live market regime classification

### 🌳 Monte Carlo Tree Search (MCTS)
- **Optimal Action Selection**: MCTS explores different trading actions and their outcomes
- **Range-Aware States**: Custom trading states that understand London range dynamics
- **Risk-Reward Optimization**: Balances exploration and exploitation for optimal decisions
- **Strategy-Specific Rewards**: Tailored reward system for range break scenarios

### 📊 Data Handling
- **CSV Data Loader**: Replaces Yahoo Finance with custom CSV data loading for your 60,000+ OHLCV rows
- **Feature Engineering**: Comprehensive technical indicators and London range-specific features
- **Data Preprocessing**: Automatic handling of missing values, outliers, and feature scaling

### 🔴 Live Trading Integration
- **MT5 Compatibility**: Signal generator designed for MetaTrader 5 integration
- **Real-time Signals**: Live signal generation with ML and MCTS enhancements
- **Risk Management**: Built-in position sizing, stop losses, and drawdown protection

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LONDON RANGE BREAK SYSTEM               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   CSV Data      │    │   Live MT5      │                │
│  │   Loader        │    │   Data Feed     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       │                                    │
│           ┌─────────────────────────────┐                  │
│           │    Feature Engineering      │                  │
│           │  - Technical Indicators     │                  │
│           │  - London Range Features    │                  │
│           │  - Time-based Features      │                  │
│           └─────────────────────────────┘                  │
│                       │                                    │
│  ┌────────────────────┼────────────────────┐               │
│  │                    │                    │               │
│  │  ┌─────────────────▼───┐  ┌─────────────▼─────────────┐ │
│  │  │   Base Strategy     │  │   ML Enhancement        │ │
│  │  │  - Range Detection  │  │  - Breakout Success     │ │
│  │  │  - Breakout Signals │  │  - Direction Prediction │ │
│  │  │  - Risk Management  │  │  - Profit Estimation    │ │
│  │  └─────────────────────┘  └─────────────────────────┘ │
│  │                    │                    │               │
│  │                    └────────┬───────────┘               │
│  │                             │                           │
│  │              ┌──────────────▼──────────────┐            │
│  │              │        MCTS Optimizer       │            │
│  │              │   - Action Selection        │            │
│  │              │   - Strategy Evaluation     │            │
│  │              │   - Risk-Reward Balance     │            │
│  │              └──────────────┬──────────────┘            │
│  └─────────────────────────────┼─────────────────────────────┘
│                               │                           │
│                ┌──────────────▼──────────────┐            │
│                │      Signal Generator       │            │
│                │   - Live Signal Creation    │            │
│                │   - MT5 Integration         │            │
│                │   - Performance Monitoring  │            │
│                └─────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd london-range-break-system
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create necessary directories**:
```bash
mkdir models
mkdir live_signals
mkdir data
```

## Usage

### 1. Quick Demo (Recommended First Step)

Run the comprehensive demo to see all system features:

```python
python london_range_demo.py
```

This will:
- Generate sample XAUUSD data or load your CSV file
- Demonstrate the base London Range Break strategy
- Train and test ML models
- Show MCTS optimization in action
- Simulate live signal generation
- Generate comprehensive analysis and reports

### 2. Using Your Own XAUUSD Data

Place your CSV file in the project directory and run:

```python
from csv_data_loader import CSVDataLoader

# Load your data
loader = CSVDataLoader()
data = loader.load_csv_data('your_xauusd_data.csv')

# Process for London Range Break features
processed_data = loader.prepare_london_range_features(data)
```

### 3. Training ML Models

```python
from ml_london_strategy import MLLondonRangeBreakStrategy

# Configure ML strategy
config = {
    'use_ml_filter': True,
    'ml_confidence_threshold': 0.6,
    'ml_models_path': 'models/'
}

# Create and train ML strategy
ml_strategy = MLLondonRangeBreakStrategy(config)
performance_metrics = ml_strategy.train_ml_models(loader, train_data, val_data)
```

### 4. Running Backtests

```python
from london_range_strategy import LondonRangeBreakStrategy, LondonRangeBreakBacktester

# Configure strategy
strategy_config = {
    'lot_size': 0.01,
    'trade_type': 'all',
    'stop_loss_points': 2700,
    'rr_ratio': 1.0,
    'min_range_points': 100,
    'max_range_points': 5000
}

# Create strategy and backtester
strategy = LondonRangeBreakStrategy(strategy_config)
backtester = LondonRangeBreakBacktester(strategy)

# Run backtest
results = backtester.run_backtest(processed_data)
```

### 5. Live Signal Generation

```python
from live_signal_generator import create_live_signal_generator

# Create signal generator
signal_generator = create_live_signal_generator()

# Start live signal generation
if signal_generator.start():
    while True:
        signal = signal_generator.get_next_signal(timeout=5.0)
        if signal:
            print(f"New signal: {signal.signal_type} @ {signal.entry_price}")
            # Process signal for MT5 execution
```

## Configuration

### Strategy Parameters

```python
strategy_config = {
    'lot_size': 0.01,                    # Position size
    'trade_type': 'all',                 # 'all', 'buy_only', 'sell_only'
    'rr_ratio': 1.0,                     # Risk-reward ratio
    'stop_loss_points': 2700,            # Stop loss in points
    'order_offset_points': 1,            # Entry offset from range
    'use_trailing': True,                # Enable trailing stops
    'trailing_points': 1100,             # Trailing distance
    'min_profit_points': 200,            # Min profit to start trailing
    'london_start_hour': 8,              # London session start (UTC)
    'min_range_points': 100,             # Minimum valid range
    'max_range_points': 5000,            # Maximum valid range
    'max_open_trades': 2,                # Max simultaneous positions
    'max_daily_drawdown_percent': 5.0,   # Daily drawdown limit
    'point_size': 0.01                   # Point size for XAUUSD
}
```

### ML Configuration

```python
ml_config = {
    'use_ml_filter': True,               # Enable ML filtering
    'ml_confidence_threshold': 0.6,      # Minimum ML confidence
    'ml_models_path': 'models/'          # Model storage directory
}
```

### MCTS Configuration

```python
mcts_config = {
    'exploration_constant': 1.414,       # UCB1 exploration parameter
    'max_iterations': 1000,              # MCTS iterations
    'max_depth': 20,                     # Maximum search depth
    'rollout_depth': 10,                 # Simulation depth
    'range_break_reward': 2.0,           # Reward for successful breakouts
    'false_breakout_penalty': -1.0,      # Penalty for false breakouts
    'time_decay_factor': 0.95            # Time-based reward decay
}
```

## Key Components

### 1. CSV Data Loader (`csv_data_loader.py`)
- Loads OHLCV data from CSV files
- Handles various CSV formats and column mappings
- Adds technical indicators and London Range Break-specific features
- Provides data splitting and preprocessing utilities

### 2. London Range Break Strategy (`london_range_strategy.py`)
- Core strategy implementation converted from MQL5 EA
- Identifies pre-London ranges and breakout levels
- Implements position management and risk controls
- Provides backtesting framework

### 3. ML Enhancement (`ml_london_strategy.py`)
- Three specialized ML models:
  - Breakout Success Predictor (XGBoost)
  - Direction Predictor (Random Forest)
  - Profit Estimator (Gradient Boosting)
- Real-time market analysis and regime classification
- Signal filtering and confidence enhancement

### 4. MCTS Optimization (`mcts_london_strategy.py`)
- Custom trading states for London Range Break scenarios
- Action selection optimization with UCB1 algorithm
- Strategy-specific reward functions
- Performance monitoring and statistics

### 5. Live Signal Generator (`live_signal_generator.py`)
- Real-time signal generation with MT5 integration
- Combines base strategy, ML, and MCTS components
- Signal queuing and performance monitoring
- JSON output for external systems

## Strategy Logic

### 1. Range Identification
- **Pre-London Session**: 3 AM - 8 AM UTC
- **Range Calculation**: High - Low during pre-London session
- **Quality Filter**: Range must be between 100-5000 points
- **Daily Reset**: New range calculated each trading day

### 2. Breakout Detection
- **Buy Signal**: Price breaks above pre-London high + offset
- **Sell Signal**: Price breaks below pre-London low - offset
- **Session Filter**: Only trade during London session (8 AM - 4 PM UTC)
- **Quality Gate**: Only trade on good quality ranges

### 3. Position Management
- **Entry**: Market orders on confirmed breakouts
- **Stop Loss**: Fixed points based on configuration
- **Take Profit**: Risk-reward ratio based targets
- **Trailing Stops**: Dynamic stop adjustment in profit

### 4. ML Enhancement
- **Breakout Success**: Predicts probability of sustained breakout
- **Direction Bias**: Forecasts likely price direction
- **Profit Potential**: Estimates expected profit category
- **Signal Filtering**: Only trades with high ML confidence

### 5. MCTS Optimization
- **Action Selection**: Chooses optimal action (buy/sell/hold)
- **Future Simulation**: Models potential outcomes
- **Risk Assessment**: Balances risk and reward
- **Confidence Scoring**: Provides action confidence levels

## File Structure

```
london-range-break-system/
├── csv_data_loader.py              # CSV data handling
├── london_range_strategy.py        # Base strategy implementation
├── ml_london_strategy.py           # ML enhancements
├── mcts_london_strategy.py         # MCTS optimization
├── live_signal_generator.py        # Live signal generation
├── london_range_demo.py            # Comprehensive demo
├── requirements.txt                # Dependencies
├── README_LONDON_RANGE_BREAK.md   # This file
├── models/                         # ML model storage
├── live_signals/                   # Signal outputs
└── data/                          # Data files
```

## Performance Metrics

The system tracks various performance metrics:

- **Trading Performance**: Return, Sharpe ratio, max drawdown
- **Signal Quality**: Accuracy, precision, recall of ML models
- **MCTS Efficiency**: Search depth, tree size, execution time
- **Risk Metrics**: Daily drawdown, position exposure, volatility

## MT5 Integration

For live trading with MetaTrader 5:

1. **Install MT5 Python package**:
```bash
pip install MetaTrader5
```

2. **Update MT5DataProvider** in `live_signal_generator.py` with real MT5 connection
3. **Configure signal routing** to MT5 Expert Advisor
4. **Set up risk management** parameters in MT5

## Troubleshooting

### Common Issues

1. **ML Model Training Failures**:
   - Ensure sufficient data (minimum 1000 samples)
   - Check for NaN values in features
   - Verify target variable creation

2. **MCTS Performance**:
   - Reduce iterations for faster execution
   - Adjust exploration constant for different behaviors
   - Monitor memory usage with large trees

3. **Data Loading Issues**:
   - Verify CSV format and column names
   - Check datetime parsing
   - Ensure data quality and completeness

4. **Signal Generation**:
   - Verify market hours configuration
   - Check range quality filters
   - Monitor ML model performance

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading system is for educational and research purposes only. Past performance does not guarantee future results. Always test thoroughly with demo accounts before live trading and never risk more than you can afford to lose.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.

---

**Happy Trading with the London Range Break System! 🚀**