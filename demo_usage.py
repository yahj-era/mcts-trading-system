#!/usr/bin/env python3
"""
Stock Trading System with MCTS - Complete Demo
This script demonstrates the full usage of the trading system
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from stock_trading_system import DataCollector, MLPredictor
from mcts_trading import MCTSTradingAgent
from backtesting import TradingBacktester, BacktestVisualizer, PerformanceMetrics

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n🔹 {title}")
    print("-" * 40)

def demo_basic_usage():
    """Demonstrate basic system usage"""
    print_header("🚀 BASIC USAGE DEMO")
    
    # Step 1: Data Collection
    print_section("Data Collection")
    collector = DataCollector()
    
    print("📊 Fetching Apple (AAPL) stock data...")
    data = collector.fetch_data('AAPL', period='1y')
    
    if data is None:
        print("❌ Failed to fetch data. Please check internet connection.")
        return None, None, None
    
    print(f"✅ Fetched {len(data)} data points")
    print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Feature Engineering
    print_section("Feature Engineering")
    print("🔧 Engineering technical features...")
    features = collector.engineer_features()
    
    print(f"✅ Created {len(features.columns)} features")
    print(f"📏 Final dataset shape: {features.shape}")
    
    # Show sample features
    sample_features = ['RSI_14', 'MACD', 'BB_Position', 'Volume_Ratio']
    print("\n📊 Sample feature values (latest):")
    for feat in sample_features:
        if feat in features.columns:
            print(f"   {feat}: {features[feat].iloc[-1]:.4f}")
    
    # Step 3: ML Model Training
    print_section("Machine Learning Model Training")
    predictor = MLPredictor()
    
    print("🤖 Training XGBoost classifier...")
    try:
        model, scaler, accuracy = predictor.train_xgboost(features, target_horizon=1)
        print(f"✅ Model trained successfully!")
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Show feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': predictor.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            print("\n🏆 Top 5 Most Important Features:")
            for _, row in importance_df.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Test prediction
        print("\n🔮 Testing prediction on latest data...")
        test_features = features[predictor.feature_columns].iloc[-1:].values
        pred_class, pred_probs = predictor.predict(test_features, 'xgboost', 1)
        
        class_names = ['📉 Down', '➡️ Flat', '📈 Up']
        predicted_direction = class_names[pred_class[0]]
        confidence = pred_probs[0][pred_class[0]]
        
        print(f"   Prediction: {predicted_direction}")
        print(f"   Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return None, None, None
    
    return features, predictor, data

def demo_mcts_agent(features, predictor):
    """Demonstrate MCTS trading agent"""
    print_section("MCTS Trading Agent")
    
    # Initialize agent
    print("🤖 Initializing MCTS Trading Agent...")
    agent = MCTSTradingAgent(
        predictor=predictor,
        initial_cash=10000,
        transaction_cost=0.001,
        mcts_simulations=300,  # Reduced for demo speed
        mcts_c=1.414
    )
    
    print(f"✅ Agent initialized:")
    print(f"   💰 Initial Cash: ${agent.initial_cash:,}")
    print(f"   💸 Transaction Cost: {agent.transaction_cost:.1%}")
    print(f"   🔍 MCTS Simulations: {agent.mcts_simulations}")
    
    # Demonstrate single decision
    print("\n🎮 Making a single MCTS decision...")
    
    price_data = features['Close'].values
    feature_data = features[predictor.feature_columns].values
    
    # Use recent data point for demo
    demo_idx = -10
    current_price = price_data[demo_idx]
    current_features = feature_data[demo_idx]
    
    print(f"📍 Current Price: ${current_price:.2f}")
    print("🔄 Running MCTS simulations...")
    
    decision = agent.make_decision(
        current_price=current_price,
        current_features=current_features,
        price_data=price_data[:demo_idx+1],
        feature_data=feature_data[:demo_idx+1],
        current_step=demo_idx,
        max_steps=len(price_data),
        current_position=0,
        current_cash=10000
    )
    
    action_emojis = {
        'buy': '💰 BUY',
        'sell': '💸 SELL',
        'hold': '🤝 HOLD',
        'short': '📉 SHORT',
        'cover': '🔄 COVER'
    }
    
    print(f"🎯 MCTS Decision: {action_emojis.get(decision, decision)}")
    print(f"💡 Decision made after {agent.mcts_simulations} simulations")
    print(f"🔮 Each simulation explored 5-10 future steps")
    
    return agent

def demo_backtesting(features, predictor, agent):
    """Demonstrate comprehensive backtesting"""
    print_section("Comprehensive Backtesting")
    
    # Initialize backtester
    backtester = TradingBacktester(
        initial_cash=10000,
        transaction_cost=0.001
    )
    
    # Prepare data (use subset for demo speed)
    subset_size = 60  # Last 60 days
    price_data = features['Close'].values
    feature_data = features[predictor.feature_columns].values
    timestamps = features.index.tolist()
    
    subset_start = max(0, len(price_data) - subset_size)
    
    print(f"🚀 Running backtest on last {subset_size} days...")
    print("⏳ This may take a few minutes due to MCTS simulations...")
    
    # Run backtest
    results = backtester.run_backtest(
        agent=agent,
        price_data=price_data[subset_start:],
        feature_data=feature_data[subset_start:],
        timestamps=timestamps[subset_start:]
    )
    
    print("✅ Backtest completed!")
    
    # Display key results
    print_section("Performance Results")
    print(f"💰 Initial Value: ${backtester.initial_cash:,.2f}")
    print(f"💎 Final Value: ${results['final_value']:,.2f}")
    print(f"📈 Total Return: {results['total_return']:.2%}")
    print(f"⚡ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {results['win_rate']:.2%}")
    print(f"📊 Volatility: {results['volatility']:.2%}")
    print(f"🔄 Number of Trades: {results['num_trades']}")
    
    # Action distribution
    if results['actions']:
        action_counts = pd.Series(results['actions']).value_counts()
        print(f"\n🎮 Action Distribution:")
        for action, count in action_counts.items():
            pct = count / len(results['actions']) * 100
            emoji = {'buy': '💰', 'sell': '💸', 'hold': '🤝', 'short': '📉', 'cover': '🔄'}.get(action, '❓')
            print(f"   {emoji} {action.upper()}: {count} ({pct:.1f}%)")
    
    # Benchmark comparison
    benchmark_prices = price_data[subset_start:len(results['portfolio_values'])+subset_start]
    if len(benchmark_prices) > 0:
        initial_shares = backtester.initial_cash / benchmark_prices[0]
        bh_final_value = initial_shares * benchmark_prices[-1]
        bh_return = (bh_final_value - backtester.initial_cash) / backtester.initial_cash
        
        print(f"\n📊 Strategy Comparison:")
        print(f"🤖 MCTS Strategy: {results['total_return']:.2%}")
        print(f"📈 Buy & Hold: {bh_return:.2%}")
        
        if results['total_return'] > bh_return:
            outperformance = results['total_return'] - bh_return
            print(f"🏆 MCTS outperformed by {outperformance:.2%}!")
        else:
            underperformance = bh_return - results['total_return']
            print(f"📉 MCTS underperformed by {underperformance:.2%}")
    
    return results, benchmark_prices

def demo_visualization(results, benchmark_prices):
    """Demonstrate visualization capabilities"""
    print_section("Performance Visualization")
    
    print("📊 Generating performance charts...")
    
    try:
        # Create equity curve plot
        fig = BacktestVisualizer.plot_equity_curve(results, benchmark_prices)
        plt.show()
        
        # Create trade analysis plot
        if results['trade_log']:
            fig = BacktestVisualizer.plot_trade_analysis(results)
            plt.show()
        else:
            print("ℹ️ No trades executed for trade analysis")
            
        print("✅ Visualization completed!")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")

def demo_parameter_tuning(features, predictor):
    """Demonstrate parameter tuning"""
    print_section("Parameter Tuning Demo")
    
    print("🧪 Testing different MCTS configurations...")
    
    configs = [
        {'name': 'Conservative', 'simulations': 100, 'c': 0.5},
        {'name': 'Balanced', 'simulations': 200, 'c': 1.414},
        {'name': 'Aggressive', 'simulations': 300, 'c': 2.0}
    ]
    
    results_comparison = []
    
    # Use smaller subset for quick demo
    subset_size = 30
    price_data = features['Close'].values
    feature_data = features[predictor.feature_columns].values
    timestamps = features.index.tolist()
    subset_start = max(0, len(price_data) - subset_size)
    
    for config in configs:
        print(f"\n   Testing {config['name']} configuration...")
        
        # Create agent with specific configuration
        test_agent = MCTSTradingAgent(
            predictor=predictor,
            initial_cash=10000,
            transaction_cost=0.001,
            mcts_simulations=config['simulations'],
            mcts_c=config['c']
        )
        
        # Run quick backtest
        test_backtester = TradingBacktester(initial_cash=10000, transaction_cost=0.001)
        
        try:
            test_results = test_backtester.run_backtest(
                agent=test_agent,
                price_data=price_data[subset_start:],
                feature_data=feature_data[subset_start:],
                timestamps=timestamps[subset_start:]
            )
            
            results_comparison.append({
                'Config': config['name'],
                'Return': f"{test_results['total_return']:.2%}",
                'Sharpe': f"{test_results['sharpe_ratio']:.3f}",
                'Trades': test_results['num_trades']
            })
            
            print(f"     📈 Return: {test_results['total_return']:.2%}")
            print(f"     ⚡ Sharpe: {test_results['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            results_comparison.append({
                'Config': config['name'],
                'Return': 'Error',
                'Sharpe': 'Error',
                'Trades': 0
            })
    
    # Display comparison
    print(f"\n📊 Configuration Comparison:")
    comparison_df = pd.DataFrame(results_comparison)
    print(comparison_df.to_string(index=False))

def main():
    """Main demo function"""
    print_header("🎯 STOCK TRADING SYSTEM WITH MCTS")
    print("A comprehensive ML-based trading system demonstration")
    print("🍎 Using Apple Inc. (AAPL) as example stock")
    
    try:
        # Step 1: Basic Usage
        features, predictor, raw_data = demo_basic_usage()
        if features is None:
            print("❌ Demo failed at data collection stage")
            return
        
        # Step 2: MCTS Agent Demo
        agent = demo_mcts_agent(features, predictor)
        
        # Step 3: Backtesting Demo
        results, benchmark_prices = demo_backtesting(features, predictor, agent)
        
        # Step 4: Visualization Demo
        demo_visualization(results, benchmark_prices)
        
        # Step 5: Parameter Tuning Demo
        demo_parameter_tuning(features, predictor)
        
        # Final Summary
        print_header("🎉 DEMO COMPLETED SUCCESSFULLY")
        print("✅ Data collection and feature engineering")
        print("✅ ML model training and prediction")
        print("✅ MCTS decision making")
        print("✅ Comprehensive backtesting")
        print("✅ Performance visualization")
        print("✅ Parameter tuning demonstration")
        
        print("\n💡 Next Steps:")
        print("   1. Experiment with different stocks")
        print("   2. Tune MCTS parameters for better performance")
        print("   3. Add more sophisticated features")
        print("   4. Implement real-time trading capabilities")
        print("   5. Scale to multiple assets")
        
        print("\n⚠️  Important Reminder:")
        print("   This system is for educational purposes only.")
        print("   Always test thoroughly before live trading.")
        print("   Past performance does not guarantee future results.")
        
    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()