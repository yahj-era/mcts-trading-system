"""
Monte Carlo Tree Search for Stock Trading
Implements MCTS algorithm for optimizing trading decisions with UCT
"""

import numpy as np
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy

@dataclass
class TradingState:
    """
    Represents the current state of the trading environment
    """
    current_price: float
    current_features: np.ndarray
    position: float  # -1 (short), 0 (flat), 1 (long)
    cash: float
    portfolio_value: float
    step: int
    max_steps: int
    transaction_cost: float = 0.001  # 0.1% transaction cost
    
    def __post_init__(self):
        """Calculate initial portfolio value"""
        if hasattr(self, '_initialized'):
            return
        self.portfolio_value = self.cash + (self.position * self.current_price)
        self._initialized = True
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return self.step >= self.max_steps
    
    def get_possible_actions(self) -> List[str]:
        """Get list of possible actions from current state"""
        actions = ['hold']
        
        # Can buy if we have cash and not already long
        if self.cash > self.current_price * (1 + self.transaction_cost) and self.position <= 0:
            actions.append('buy')
        
        # Can sell if we have a position
        if self.position > 0:
            actions.append('sell')
        
        # Can short if we don't have a long position
        if self.position <= 0:
            actions.append('short')
        
        # Can cover short if we have short position
        if self.position < 0:
            actions.append('cover')
        
        return actions
    
    def apply_action(self, action: str, next_price: float, next_features: np.ndarray) -> 'TradingState':
        """Apply an action and return the new state"""
        new_state = copy.deepcopy(self)
        new_state.current_price = next_price
        new_state.current_features = next_features
        new_state.step += 1
        
        # Calculate transaction cost
        if action != 'hold':
            cost = self.current_price * self.transaction_cost
        else:
            cost = 0
        
        if action == 'buy':
            if new_state.position <= 0 and new_state.cash >= self.current_price * (1 + self.transaction_cost):
                # Buy one unit
                new_state.cash -= self.current_price * (1 + self.transaction_cost)
                new_state.position += 1
                
        elif action == 'sell':
            if new_state.position > 0:
                # Sell one unit
                new_state.cash += self.current_price * (1 - self.transaction_cost)
                new_state.position -= 1
                
        elif action == 'short':
            if new_state.position <= 0:
                # Short one unit
                new_state.cash += self.current_price * (1 - self.transaction_cost)
                new_state.position -= 1
                
        elif action == 'cover':
            if new_state.position < 0 and new_state.cash >= self.current_price * (1 + self.transaction_cost):
                # Cover short position
                new_state.cash -= self.current_price * (1 + self.transaction_cost)
                new_state.position += 1
        
        # Update portfolio value
        new_state.portfolio_value = new_state.cash + (new_state.position * next_price)
        
        return new_state
    
    def get_reward(self, previous_portfolio_value: float) -> float:
        """Calculate reward based on portfolio value change"""
        return_pct = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Risk-adjusted return (simple Sharpe-like ratio)
        # Penalize large positions and encourage diversification
        position_penalty = abs(self.position) * 0.01
        
        return return_pct - position_penalty
    
    def __hash__(self):
        """Make state hashable for use in dictionaries"""
        return hash((
            round(self.current_price, 2),
            self.position,
            round(self.cash, 2),
            self.step
        ))
    
    def __eq__(self, other):
        """Check equality for state comparison"""
        if not isinstance(other, TradingState):
            return False
        return (
            abs(self.current_price - other.current_price) < 0.01 and
            self.position == other.position and
            abs(self.cash - other.cash) < 0.01 and
            self.step == other.step
        )

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree
    """
    
    def __init__(self, state: TradingState, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.get_possible_actions().copy()
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        return self.state.is_terminal()
    
    def get_ucb_score(self, c: float = 1.414) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def select_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select child with highest UCB score"""
        return max(self.children.values(), key=lambda child: child.get_ucb_score(c))
    
    def expand(self, action: str, next_state: TradingState) -> 'MCTSNode':
        """Expand the tree by adding a new child node"""
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node
    
    def update(self, reward: float):
        """Update node statistics with simulation result"""
        self.visits += 1
        self.total_reward += reward
    
    def get_best_action(self) -> str:
        """Get the action leading to the child with most visits"""
        if not self.children:
            return random.choice(self.state.get_possible_actions())
        
        return max(self.children.items(), key=lambda x: x[1].visits)[0]

class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search implementation for trading optimization
    """
    
    def __init__(self, predictor, price_data: np.ndarray, feature_data: np.ndarray, 
                 c: float = 1.414, n_simulations: int = 1000):
        """
        Initialize MCTS
        
        Args:
            predictor: ML model for price prediction
            price_data: Array of historical prices
            feature_data: Array of corresponding features
            c: UCB exploration parameter
            n_simulations: Number of MCTS simulations
        """
        self.predictor = predictor
        self.price_data = price_data
        self.feature_data = feature_data
        self.c = c
        self.n_simulations = n_simulations
        
    def simulate_future_path(self, start_idx: int, steps: int) -> Tuple[List[float], List[np.ndarray]]:
        """
        Simulate future price path using ML predictions
        
        Args:
            start_idx: Starting index in the data
            steps: Number of steps to simulate
            
        Returns:
            Tuple of (prices, features) for the simulated path
        """
        prices = []
        features = []
        
        current_idx = start_idx
        current_price = self.price_data[current_idx]
        
        for step in range(steps):
            if current_idx + 1 < len(self.price_data):
                # Use actual data if available
                current_idx += 1
                next_price = self.price_data[current_idx]
                next_features = self.feature_data[current_idx]
            else:
                # Use ML prediction for future steps
                try:
                    current_features = self.feature_data[min(current_idx, len(self.feature_data) - 1)]
                    
                    # Get prediction probabilities
                    _, probabilities = self.predictor.predict(
                        current_features.reshape(1, -1), 
                        model_type='xgboost', 
                        target_horizon=1
                    )
                    
                    # Sample from prediction distribution
                    pred_class = np.random.choice([0, 1, 2], p=probabilities[0])
                    
                    # Convert prediction to price change
                    if pred_class == 2:  # Up
                        price_change = np.random.normal(0.02, 0.01)  # ~2% up
                    elif pred_class == 0:  # Down
                        price_change = np.random.normal(-0.02, 0.01)  # ~2% down
                    else:  # Flat
                        price_change = np.random.normal(0.0, 0.005)  # ~0% change
                    
                    next_price = current_price * (1 + price_change)
                    next_features = current_features  # Simplified: reuse features
                    
                except Exception as e:
                    # Fallback: random walk
                    price_change = np.random.normal(0, 0.01)
                    next_price = current_price * (1 + price_change)
                    next_features = self.feature_data[min(current_idx, len(self.feature_data) - 1)]
            
            prices.append(next_price)
            features.append(next_features)
            current_price = next_price
        
        return prices, features
    
    def simulate_random_playout(self, state: TradingState, max_steps: int = 10) -> float:
        """
        Perform random playout from given state
        
        Args:
            state: Starting state for simulation
            max_steps: Maximum steps to simulate
            
        Returns:
            Total reward from the simulation
        """
        current_state = copy.deepcopy(state)
        initial_portfolio_value = current_state.portfolio_value
        total_reward = 0.0
        
        # Simulate future price path
        remaining_steps = min(max_steps, current_state.max_steps - current_state.step)
        if remaining_steps <= 0:
            return 0.0
        
        try:
            start_idx = min(current_state.step, len(self.price_data) - 1)
            future_prices, future_features = self.simulate_future_path(start_idx, remaining_steps)
            
            for i, (next_price, next_features) in enumerate(zip(future_prices, future_features)):
                if current_state.is_terminal():
                    break
                
                # Choose random action
                possible_actions = current_state.get_possible_actions()
                action = random.choice(possible_actions)
                
                # Apply action
                previous_portfolio_value = current_state.portfolio_value
                current_state = current_state.apply_action(action, next_price, next_features)
                
                # Calculate reward
                reward = current_state.get_reward(previous_portfolio_value)
                total_reward += reward
            
        except Exception as e:
            # Fallback to simple random simulation
            for _ in range(remaining_steps):
                if current_state.is_terminal():
                    break
                
                possible_actions = current_state.get_possible_actions()
                action = random.choice(possible_actions)
                
                # Simple price simulation
                price_change = np.random.normal(0, 0.01)
                next_price = current_state.current_price * (1 + price_change)
                next_features = current_state.current_features
                
                previous_portfolio_value = current_state.portfolio_value
                current_state = current_state.apply_action(action, next_price, next_features)
                
                reward = current_state.get_reward(previous_portfolio_value)
                total_reward += reward
        
        return total_reward
    
    def search(self, initial_state: TradingState) -> str:
        """
        Perform MCTS to find the best action
        
        Args:
            initial_state: Starting state for the search
            
        Returns:
            Best action to take
        """
        root = MCTSNode(initial_state)
        
        for simulation in range(self.n_simulations):
            # Selection: traverse tree using UCB
            node = root
            path = [node]
            
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(self.c)
                path.append(node)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                
                # Simulate next state
                try:
                    if node.state.step + 1 < len(self.price_data):
                        next_price = self.price_data[node.state.step + 1]
                        next_features = self.feature_data[node.state.step + 1]
                    else:
                        # Use prediction for future state
                        _, probabilities = self.predictor.predict(
                            node.state.current_features.reshape(1, -1),
                            model_type='xgboost',
                            target_horizon=1
                        )
                        pred_class = np.random.choice([0, 1, 2], p=probabilities[0])
                        
                        if pred_class == 2:
                            price_change = 0.02
                        elif pred_class == 0:
                            price_change = -0.02
                        else:
                            price_change = 0.0
                        
                        next_price = node.state.current_price * (1 + price_change)
                        next_features = node.state.current_features
                except:
                    # Fallback
                    next_price = node.state.current_price * (1 + np.random.normal(0, 0.01))
                    next_features = node.state.current_features
                
                next_state = node.state.apply_action(action, next_price, next_features)
                node = node.expand(action, next_state)
                path.append(node)
            
            # Simulation: random playout
            reward = self.simulate_random_playout(node.state)
            
            # Backpropagation: update all nodes in path
            for node in path:
                node.update(reward)
        
        # Return best action
        return root.get_best_action()

class MCTSTradingAgent:
    """
    Complete trading agent using MCTS for decision making
    """
    
    def __init__(self, predictor, initial_cash: float = 10000, transaction_cost: float = 0.001,
                 mcts_simulations: int = 500, mcts_c: float = 1.414):
        """
        Initialize trading agent
        
        Args:
            predictor: Trained ML predictor
            initial_cash: Starting cash amount
            transaction_cost: Transaction cost percentage
            mcts_simulations: Number of MCTS simulations per decision
            mcts_c: UCB exploration parameter
        """
        self.predictor = predictor
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.mcts_simulations = mcts_simulations
        self.mcts_c = mcts_c
        
        # Trading history
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        
    def reset(self, initial_cash: float = None):
        """Reset agent for new trading session"""
        if initial_cash is not None:
            self.initial_cash = initial_cash
        
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
    
    def make_decision(self, current_price: float, current_features: np.ndarray, 
                     price_data: np.ndarray, feature_data: np.ndarray, 
                     current_step: int, max_steps: int,
                     current_position: float = 0, current_cash: float = None) -> str:
        """
        Make trading decision using MCTS
        
        Args:
            current_price: Current stock price
            current_features: Current feature vector
            price_data: Historical price data
            feature_data: Historical feature data
            current_step: Current time step
            max_steps: Maximum steps in episode
            current_position: Current position (-1, 0, 1)
            current_cash: Current cash amount
            
        Returns:
            Trading action ('buy', 'sell', 'hold', 'short', 'cover')
        """
        if current_cash is None:
            current_cash = self.initial_cash
        
        # Create current state
        state = TradingState(
            current_price=current_price,
            current_features=current_features,
            position=current_position,
            cash=current_cash,
            portfolio_value=0,  # Will be calculated in __post_init__
            step=current_step,
            max_steps=max_steps,
            transaction_cost=self.transaction_cost
        )
        
        # Initialize MCTS
        mcts = MonteCarloTreeSearch(
            predictor=self.predictor,
            price_data=price_data,
            feature_data=feature_data,
            c=self.mcts_c,
            n_simulations=self.mcts_simulations
        )
        
        # Search for best action
        best_action = mcts.search(state)
        
        # Record decision
        self.action_history.append(best_action)
        self.portfolio_history.append(state.portfolio_value)
        
        return best_action

if __name__ == "__main__":
    print("Monte Carlo Tree Search for Trading - Implementation Complete")
    print("Classes implemented:")
    print("- TradingState: Represents trading environment state")
    print("- MCTSNode: Node in the MCTS tree")
    print("- MonteCarloTreeSearch: Core MCTS algorithm")
    print("- MCTSTradingAgent: Complete trading agent")