# Stock Trading System with Monte Carlo Tree Search (MCTS)

A comprehensive machine learning-based stock trading system that uses Monte Carlo Tree Search for decision-making optimization. The system integrates technical analysis, machine learning predictions, and advanced decision algorithms to create an intelligent trading agent.

## 🚀 Features

### Core Components
- **📊 Data Collection**: Real-time and historical OHLCV data from Yahoo Finance
- **🔧 Feature Engineering**: 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **🤖 ML Models**: XGBoost, LSTM, and Transformer models for price prediction
- **🌳 MCTS Decision Engine**: Monte Carlo Tree Search with UCB1 for optimal action selection
- **📈 Backtesting Framework**: Comprehensive performance evaluation with risk metrics

### Advanced Capabilities
- **UCT Algorithm**: Upper Confidence Bounds for Trees balancing exploration/exploitation
- **Multi-step Simulation**: 5-10 step forward lookahead for decision optimization
- **Risk Management**: Transaction costs, position limits, and drawdown control
- **Performance Analytics**: Sharpe ratio, Sortino ratio, maximum drawdown, win rate, VaR, CVaR
- **Visualization**: Equity curves, trade analysis, strategy comparison charts

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│  Feature Engine  │───▶│   ML Predictor  │
│                 │    │                  │    │                 │
│ • Yahoo Finance │    │ • Technical      │    │ • XGBoost       │
│ • OHLCV Data    │    │   Indicators     │    │ • LSTM          │
│ • Real-time     │    │ • Price Features │    │ • Transformer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Backtester    │◀───│  Trading Agent   │◀───│  MCTS Engine    │
│                 │    │                  │    │                 │
│ • Performance   │    │ • Decision Logic │    │ • UCB1 Selection│
│ • Metrics       │    │ • Risk Rules     │    │ • Tree Search   │
│ • Visualization │    │ • Execution      │    │ • Simulation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

### Dependencies
```
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
tensorflow >= 2.8.0
yfinance >= 0.1.70
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

### Installation

#### Option 1: Google Colab (Recommended)
```bash
# Run this in a Colab cell
!pip install yfinance xgboost tensorflow scikit-learn matplotlib seaborn
```

#### Option 2: Local Installation
```bash
# Clone the repository
!git clone https://github.com/your-username/mcts-trading-system.git
%cd mcts-trading-system

# Install dependencies
!pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from stock_trading_system import DataCollector, MLPredictor
from mcts_trading import MCTSTradingAgent
from backtesting import TradingBacktester

# 1. Collect and prepare data
collector = DataCollector()
data = collector.fetch_data('AAPL', period='1y')
features = collector.engineer_features()

# 2. Train ML model
predictor = MLPredictor()
model, scaler, accuracy = predictor.train_xgboost(features)
print(f"Model accuracy: {accuracy:.4f}")

# 3. Initialize MCTS trading agent
agent = MCTSTradingAgent(
    predictor=predictor,
    initial_cash=10000,
    mcts_simulations=500,
    mcts_c=1.414
)

# 4. Run backtest
backtester = TradingBacktester()
results = backtester.run_backtest(
    agent=agent,
    price_data=features['Close'].values,
    feature_data=features[predictor.feature_columns].values
)

# 5. Analyze results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Advanced Configuration

```python
# Custom MCTS parameters
agent = MCTSTradingAgent(
    predictor=predictor,
    initial_cash=50000,
    transaction_cost=0.001,  # 0.1% transaction cost
    mcts_simulations=1000,   # More simulations = better decisions
    mcts_c=1.414            # UCB exploration parameter
)

# Multiple model ensemble
predictor.train_xgboost(features, target_horizon=1)
predictor.train_lstm(features, target_horizon=1, epochs=100)

# Custom backtesting period
subset_data = features.tail(252)  # Last year
results = backtester.run_backtest(agent, subset_data)
```

## 📊 Performance Metrics

The system provides comprehensive performance analytics:

### Return Metrics
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Year-over-year performance
- **Excess Return**: Performance vs benchmark

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return to max drawdown ratio
- **VaR/CVaR**: Value at Risk metrics

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Total wins / total losses
- **Average Trade Return**: Mean return per trade
- **Trade Frequency**: Number of trades per period

## 🌳 MCTS Algorithm Details

### Core Components

#### 1. Trading State
```python
@dataclass
class TradingState:
    current_price: float
    current_features: np.ndarray
    position: float           # -1 (short), 0 (flat), 1 (long)
    cash: float
    portfolio_value: float
    step: int
    max_steps: int
```

#### 2. UCB1 Selection
```python
def ucb_score(node, c=1.414):
    if node.visits == 0:
        return float('inf')
    
    exploitation = node.total_reward / node.visits
    exploration = c * sqrt(log(parent.visits) / node.visits)
    
    return exploitation + exploration
```

#### 3. Simulation Process
1. **Selection**: Navigate tree using UCB1 scores
2. **Expansion**: Add new child nodes for untried actions
3. **Simulation**: Random playout with ML-guided price predictions
4. **Backpropagation**: Update node statistics with simulation rewards

### Action Space
- **Buy**: Purchase stock (long position)
- **Sell**: Sell stock (close long position)  
- **Hold**: Maintain current position
- **Short**: Short sell stock (short position)
- **Cover**: Cover short position

### Reward Function
```python
def get_reward(self, previous_portfolio_value):
    return_pct = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
    position_penalty = abs(self.position) * 0.01  # Risk adjustment
    return return_pct - position_penalty
```

## 📈 Visualization

The system includes comprehensive visualization capabilities:

### Equity Curve Analysis
```python
from backtesting import BacktestVisualizer

# Plot complete performance analysis
fig = BacktestVisualizer.plot_equity_curve(results, benchmark_prices)
plt.show()

# Trade analysis
fig = BacktestVisualizer.plot_trade_analysis(results)
plt.show()
```

### Strategy Comparison
```python
# Compare multiple strategies
from backtesting import compare_strategies

strategies = [mcts_results, buy_hold_results, momentum_results]
strategy_names = ['MCTS', 'Buy & Hold', 'Momentum']

fig = compare_strategies(strategies, strategy_names)
plt.show()
```

## 🔧 Customization

### Parameter Tuning

#### MCTS Parameters
```python
# Conservative (risk-averse)
agent = MCTSTradingAgent(
    mcts_simulations=1000,
    mcts_c=0.5  # Less exploration
)

# Aggressive (high exploration)  
agent = MCTSTradingAgent(
    mcts_simulations=2000,
    mcts_c=2.0  # More exploration
)
```

#### ML Model Configuration
```python
# XGBoost tuning
predictor.train_xgboost(
    features,
    target_horizon=3,    # 3-day prediction
    test_size=0.3,       # 30% test split
    n_estimators=200,    # More trees
    max_depth=8          # Deeper trees
)

# LSTM configuration
predictor.train_lstm(
    features,
    sequence_length=30,  # 30-day sequences
    epochs=100,          # More training
    batch_size=64        # Larger batches
)
```

### Feature Engineering Extensions
```python
# Add custom indicators
def add_custom_features(data):
    # Stochastic Oscillator
    data['Stoch_K'] = ((data['Close'] - data['Low'].rolling(14).min()) / 
                       (data['High'].rolling(14).max() - data['Low'].rolling(14).min())) * 100
    
    # Williams %R
    data['Williams_R'] = ((data['High'].rolling(14).max() - data['Close']) / 
                          (data['High'].rolling(14).max() - data['Low'].rolling(14).min())) * -100
    
    # Custom momentum
    data['Custom_Momentum'] = data['Close'] / data['Close'].shift(21) - 1
    
    return data
```

## 🧪 Experimental Features

### Multi-Asset Trading
```python
# Portfolio of stocks
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
portfolio_agent = MultiAssetMCTSAgent(
    symbols=symbols,
    allocation_weights=[0.25, 0.25, 0.25, 0.25],
    correlation_threshold=0.7
)
```

### Options Integration
```python
# Options-aware agent
options_agent = OptionsMCTSAgent(
    underlying_predictor=predictor,
    option_pricing_model='black_scholes',
    volatility_surface=vol_surface
)
```

### Real-time Trading
```python
# Live trading integration
live_agent = LiveMCTSAgent(
    broker_api=broker,
    data_feed=live_feed,
    risk_limits={'max_position': 0.1, 'max_drawdown': 0.05}
)
```

## 📚 Research and Development

### Academic References
1. **MCTS**: Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo Planning
2. **UCB**: Auer, P. et al. (2002). Finite-time Analysis of the Multiarmed Bandit Problem
3. **Algorithmic Trading**: Chan, E. (2013). Algorithmic Trading: Winning Strategies

### Performance Benchmarks
- **S&P 500 Index**: Baseline comparison
- **Buy and Hold**: Simple benchmark strategy
- **Moving Average Crossover**: Technical analysis benchmark
- **Random Walk**: Statistical baseline

## ⚠️ Risk Disclaimer

**Important Notice**: This system is designed for educational and research purposes only.

### Trading Risks
- **Market Risk**: Stock prices can decline significantly
- **Model Risk**: ML predictions may be inaccurate
- **Execution Risk**: Slippage and timing issues
- **Liquidity Risk**: Inability to execute trades

### Recommendations
1. **Paper Trading**: Test thoroughly before live trading
2. **Position Sizing**: Never risk more than you can afford to lose
3. **Diversification**: Don't put all capital in one strategy
4. **Monitoring**: Continuously monitor system performance
5. **Professional Advice**: Consult financial advisors

## 🤝 Contributing

We welcome contributions to improve the system:

### Development Areas
- **New ML Models**: Implement additional prediction models
- **Risk Management**: Enhanced position sizing and stop-loss
- **Performance**: Optimization and speed improvements
- **Documentation**: Examples and tutorials
- **Testing**: Unit tests and integration tests

### Contribution Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free financial data
- **XGBoost Team**: For the excellent gradient boosting library
- **TensorFlow Team**: For deep learning capabilities
- **scikit-learn**: For machine learning utilities
- **Open Source Community**: For inspiration and support

## 📞 Contact

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Email**: [your-email@example.com] for direct contact

---

**⚡ Built with Python, powered by MCTS, optimized for performance!**

*"In trading, the edge comes not from predicting the future, but from making optimal decisions under uncertainty."*
