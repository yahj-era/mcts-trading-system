"""
Backtesting Framework for Stock Trading System
Comprehensive performance evaluation with metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """Calculate various trading performance metrics"""
    
    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """Calculate returns from portfolio values"""
        return np.diff(portfolio_values) / portfolio_values[:-1]
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * np.mean(excess_returns) / downside_deviation
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration
        
        Returns:
            max_dd: Maximum drawdown percentage
            start_idx: Start index of max drawdown
            end_idx: End index of max drawdown
        """
        if len(portfolio_values) == 0:
            return 0.0, 0, 0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        max_dd_start = 0
        max_dd_end = 0
        current_dd_start = 0
        
        for i, value in enumerate(portfolio_values):
            if value > peak:
                peak = value
                current_dd_start = i
            else:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_start = current_dd_start
                    max_dd_end = i
        
        return max_dd, max_dd_start, max_dd_end
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0 or len(portfolio_values) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd, _, _ = PerformanceMetrics.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (total wins / total losses)"""
        if len(returns) == 0:
            return 1.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = np.sum(np.abs(losses)) if len(losses) > 0 else 1e-10
        
        return total_wins / total_losses
    
    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def var_95(returns: np.ndarray) -> float:
        """Calculate Value at Risk at 95% confidence level"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, 5)
    
    @staticmethod
    def cvar_95(returns: np.ndarray) -> float:
        """Calculate Conditional Value at Risk at 95% confidence level"""
        if len(returns) == 0:
            return 0.0
        var_95 = PerformanceMetrics.var_95(returns)
        return np.mean(returns[returns <= var_95])

class TradingBacktester:
    """
    Comprehensive backtesting framework for trading strategies
    """
    
    def __init__(self, initial_cash: float = 10000, transaction_cost: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_cash: Initial cash amount
            transaction_cost: Transaction cost percentage
        """
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_values = [self.initial_cash]
        self.actions = []
        self.prices = []
        self.timestamps = []
        self.trade_log = []
        
    def execute_trade(self, action: str, price: float, timestamp: Optional[datetime] = None):
        """
        Execute a trading action
        
        Args:
            action: Trading action ('buy', 'sell', 'hold', 'short', 'cover')
            price: Current price
            timestamp: Timestamp of the trade
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        previous_cash = self.cash
        previous_position = self.position
        
        if action == 'buy':
            if self.position <= 0 and self.cash >= price * (1 + self.transaction_cost):
                # Buy one unit
                cost = price * (1 + self.transaction_cost)
                self.cash -= cost
                self.position += 1
                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': price,
                    'quantity': 1,
                    'cost': cost,
                    'cash_after': self.cash,
                    'position_after': self.position
                })
                
        elif action == 'sell':
            if self.position > 0:
                # Sell one unit
                proceeds = price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.position -= 1
                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': price,
                    'quantity': -1,
                    'cost': -proceeds,
                    'cash_after': self.cash,
                    'position_after': self.position
                })
                
        elif action == 'short':
            if self.position <= 0:
                # Short one unit
                proceeds = price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.position -= 1
                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': price,
                    'quantity': -1,
                    'cost': -proceeds,
                    'cash_after': self.cash,
                    'position_after': self.position
                })
                
        elif action == 'cover':
            if self.position < 0 and self.cash >= price * (1 + self.transaction_cost):
                # Cover short position
                cost = price * (1 + self.transaction_cost)
                self.cash -= cost
                self.position += 1
                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': price,
                    'quantity': 1,
                    'cost': cost,
                    'cash_after': self.cash,
                    'position_after': self.position
                })
        
        # Update portfolio value
        portfolio_value = self.cash + (self.position * price)
        self.portfolio_values.append(portfolio_value)
        self.actions.append(action)
        self.prices.append(price)
        self.timestamps.append(timestamp)
    
    def run_backtest(self, agent, price_data: np.ndarray, feature_data: np.ndarray, 
                    timestamps: Optional[List[datetime]] = None) -> Dict:
        """
        Run backtest using a trading agent
        
        Args:
            agent: Trading agent with make_decision method
            price_data: Array of historical prices
            feature_data: Array of corresponding features
            timestamps: List of timestamps for each data point
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        agent.reset()
        
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(days=i) for i in range(len(price_data))]
        
        # Run through historical data
        for i in range(1, len(price_data)):  # Start from 1 to have previous data
            current_price = price_data[i]
            current_features = feature_data[i]
            current_timestamp = timestamps[i]
            
            # Make trading decision
            action = agent.make_decision(
                current_price=current_price,
                current_features=current_features,
                price_data=price_data[:i+1],  # Available data up to current point
                feature_data=feature_data[:i+1],
                current_step=i,
                max_steps=len(price_data),
                current_position=self.position,
                current_cash=self.cash
            )
            
            # Execute trade
            self.execute_trade(action, current_price, current_timestamp)
        
        # Calculate performance metrics
        portfolio_values = np.array(self.portfolio_values)
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        
        results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'actions': self.actions,
            'prices': self.prices,
            'timestamps': self.timestamps,
            'trade_log': self.trade_log,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - self.initial_cash) / self.initial_cash,
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_values)[0],
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, portfolio_values),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'volatility': PerformanceMetrics.volatility(returns),
            'var_95': PerformanceMetrics.var_95(returns),
            'cvar_95': PerformanceMetrics.cvar_95(returns),
            'num_trades': len(self.trade_log),
            'avg_trade_return': np.mean(returns) if len(returns) > 0 else 0,
        }
        
        return results

class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    @staticmethod
    def plot_equity_curve(results: Dict, benchmark_prices: Optional[np.ndarray] = None, 
                         figsize: Tuple[int, int] = (15, 10)):
        """Plot equity curve and performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Trading Strategy Performance', fontsize=16, fontweight='bold')
        
        # Equity curve
        ax1 = axes[0, 0]
        timestamps = results['timestamps']
        portfolio_values = results['portfolio_values']
        
        ax1.plot(timestamps, portfolio_values, label='Strategy', linewidth=2, color='blue')
        
        if benchmark_prices is not None:
            # Calculate buy and hold benchmark
            initial_shares = results['portfolio_values'][0] / benchmark_prices[0]
            benchmark_values = initial_shares * benchmark_prices[:len(portfolio_values)]
            ax1.plot(timestamps, benchmark_values, label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = axes[0, 1]
        returns = results['returns']
        ax2.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Returns')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        ax3 = axes[1, 0]
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        
        ax3.fill_between(timestamps, drawdown, alpha=0.3, color='red')
        ax3.plot(timestamps, drawdown, color='red', linewidth=1)
        ax3.set_title(f'Drawdown (Max: {results["max_drawdown"]:.2%})')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_data = [
            ['Total Return', f"{results['total_return']:.2%}"],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.3f}"],
            ['Sortino Ratio', f"{results['sortino_ratio']:.3f}"],
            ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
            ['Calmar Ratio', f"{results['calmar_ratio']:.3f}"],
            ['Win Rate', f"{results['win_rate']:.2%}"],
            ['Profit Factor', f"{results['profit_factor']:.3f}"],
            ['Volatility', f"{results['volatility']:.2%}"],
            ['VaR 95%', f"{results['var_95']:.4f}"],
            ['CVaR 95%', f"{results['cvar_95']:.4f}"],
            ['# Trades', f"{results['num_trades']}"],
        ]
        
        table = ax4.table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(2):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_trade_analysis(results: Dict, figsize: Tuple[int, int] = (15, 8)):
        """Plot trade analysis"""
        
        if not results['trade_log']:
            print("No trades to analyze")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        trade_df = pd.DataFrame(results['trade_log'])
        
        # Actions distribution
        ax1 = axes[0, 0]
        action_counts = pd.Series(results['actions']).value_counts()
        ax1.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        ax1.set_title('Action Distribution')
        
        # Trade timing
        ax2 = axes[0, 1]
        trade_timestamps = [trade['timestamp'] for trade in results['trade_log']]
        trade_prices = [trade['price'] for trade in results['trade_log']]
        
        ax2.plot(results['timestamps'], results['prices'], alpha=0.7, color='gray', label='Price')
        
        buy_trades = [t for t in results['trade_log'] if t['action'] in ['buy', 'cover']]
        sell_trades = [t for t in results['trade_log'] if t['action'] in ['sell', 'short']]
        
        if buy_trades:
            buy_times = [t['timestamp'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax2.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy/Cover', zorder=5)
        
        if sell_trades:
            sell_times = [t['timestamp'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax2.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell/Short', zorder=5)
        
        ax2.set_title('Trade Execution Points')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Portfolio value over time
        ax3 = axes[1, 0]
        ax3.plot(results['timestamps'], results['portfolio_values'], linewidth=2, color='blue')
        ax3.set_title('Portfolio Value Evolution')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.grid(True, alpha=0.3)
        
        # Position over time
        ax4 = axes[1, 1]
        positions = []
        current_pos = 0
        for action in results['actions']:
            if action == 'buy':
                current_pos += 1
            elif action == 'sell':
                current_pos -= 1
            elif action == 'short':
                current_pos -= 1
            elif action == 'cover':
                current_pos += 1
            positions.append(current_pos)
        
        ax4.plot(results['timestamps'], positions, linewidth=2, color='orange')
        ax4.set_title('Position Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Position')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def compare_strategies(results_list: List[Dict], strategy_names: List[str], 
                      figsize: Tuple[int, int] = (15, 10)):
    """Compare multiple trading strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Equity curves comparison
    ax1 = axes[0, 0]
    for i, (results, name) in enumerate(zip(results_list, strategy_names)):
        timestamps = results['timestamps']
        portfolio_values = results['portfolio_values']
        ax1.plot(timestamps, portfolio_values, label=name, linewidth=2)
    
    ax1.set_title('Equity Curves Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    ax2 = axes[0, 1]
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    x = np.arange(len(strategy_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[metric] for results in results_list]
        if metric == 'max_drawdown':
            values = [-v for v in values]  # Make drawdown positive for visualization
        ax2.bar(x + i * width, values, width, label=metric_names[i])
    
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(strategy_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Returns distribution comparison
    ax3 = axes[1, 0]
    for i, (results, name) in enumerate(zip(results_list, strategy_names)):
        returns = results['returns']
        ax3.hist(returns, bins=50, alpha=0.6, label=name, density=True)
    
    ax3.set_title('Returns Distribution Comparison')
    ax3.set_xlabel('Daily Returns')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Risk-Return scatter
    ax4 = axes[1, 1]
    for i, (results, name) in enumerate(zip(results_list, strategy_names)):
        annual_return = results['total_return'] * (252 / len(results['returns']))
        volatility = results['volatility']
        ax4.scatter(volatility, annual_return, s=100, label=name)
    
    ax4.set_title('Risk-Return Profile')
    ax4.set_xlabel('Volatility (Annualized)')
    ax4.set_ylabel('Return (Annualized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Backtesting Framework - Implementation Complete")
    print("Classes implemented:")
    print("- PerformanceMetrics: Calculate trading performance metrics")
    print("- TradingBacktester: Run comprehensive backtests")
    print("- BacktestVisualizer: Create performance visualizations")
    print("- compare_strategies: Compare multiple trading strategies")