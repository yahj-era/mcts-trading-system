"""
Monte Carlo Tree Search for London Range Break Strategy
Adapts MCTS algorithm for London Range Break trading with range-based states
"""

import numpy as np
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import copy
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class LondonRangeTradingState:
    """
    Trading state specific to London Range Break strategy
    """
    current_price: float
    current_features: np.ndarray
    position: float  # -1 (short), 0 (flat), 1 (long)
    cash: float
    portfolio_value: float
    step: int
    max_steps: int
    
    # London Range Break specific
    pre_london_high: float
    pre_london_low: float
    range_points: float
    buy_break_level: float
    sell_break_level: float
    session_id: str
    time_since_london_open: int  # Hours since London session opened
    
    # Range state information
    price_in_range: bool
    above_range: bool
    below_range: bool
    range_quality: str  # 'good', 'poor', 'invalid'
    
    # Market conditions
    volatility_level: str  # 'low', 'medium', 'high'
    breakout_strength: float  # How far price moved beyond range
    
    # Risk management
    transaction_cost: float = 0.001
    max_drawdown: float = 0.05
    
    def __post_init__(self):
        """Calculate derived state information"""
        if hasattr(self, '_initialized'):
            return
            
        self.portfolio_value = self.cash + (self.position * self.current_price)
        
        # Update range position
        self._update_range_position()
        
        # Update market conditions
        self._update_market_conditions()
        
        self._initialized = True
    
    def _update_range_position(self):
        """Update price position relative to range"""
        if self.range_points > 0:
            self.price_in_range = (self.pre_london_low <= self.current_price <= self.pre_london_high)
            self.above_range = self.current_price > self.pre_london_high
            self.below_range = self.current_price < self.pre_london_low
        else:
            self.price_in_range = True
            self.above_range = False
            self.below_range = False
    
    def _update_market_conditions(self):
        """Update market condition indicators"""
        # Range quality assessment
        if 200 <= self.range_points <= 3000:
            self.range_quality = 'good'
        elif 100 <= self.range_points <= 5000:
            self.range_quality = 'fair'
        else:
            self.range_quality = 'poor'
        
        # Volatility assessment (simplified)
        if len(self.current_features) > 0:
            # Assume ATR is in features
            atr_index = 5  # Simplified - would need proper feature mapping
            if atr_index < len(self.current_features):
                atr = self.current_features[atr_index]
                if atr < 0.5:
                    self.volatility_level = 'low'
                elif atr < 1.5:
                    self.volatility_level = 'medium'
                else:
                    self.volatility_level = 'high'
            else:
                self.volatility_level = 'medium'
        else:
            self.volatility_level = 'medium'
        
        # Breakout strength
        if self.above_range:
            self.breakout_strength = (self.current_price - self.pre_london_high) / self.range_points
        elif self.below_range:
            self.breakout_strength = (self.pre_london_low - self.current_price) / self.range_points
        else:
            self.breakout_strength = 0.0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return (self.step >= self.max_steps or 
                self.time_since_london_open >= 8 or  # End after 8 hours
                self.portfolio_value <= self.cash * (1 - self.max_drawdown))
    
    def get_possible_actions(self) -> List[str]:
        """Get list of possible actions based on London Range Break logic"""
        actions = ['hold']
        
        # Only trade during London session (first 8 hours)
        if self.time_since_london_open >= 8:
            return actions
        
        # Range quality filter
        if self.range_quality == 'poor':
            return actions
        
        # Buy actions (only if price breaks above range)
        if (self.above_range and self.position <= 0 and 
            self.cash > self.current_price * (1 + self.transaction_cost)):
            actions.extend(['buy', 'buy_aggressive'])
        
        # Sell actions (only if price breaks below range)
        if (self.below_range and self.position >= 0 and
            self.cash > self.current_price * (1 + self.transaction_cost)):
            actions.extend(['sell', 'sell_aggressive'])
        
        # Close position actions
        if self.position > 0:
            actions.extend(['close_long', 'close_long_partial'])
        elif self.position < 0:
            actions.extend(['close_short', 'close_short_partial'])
        
        return actions
    
    def get_state_signature(self) -> str:
        """Get unique signature for this state for MCTS node identification"""
        range_pos = 'in_range'
        if self.above_range:
            range_pos = 'above_range'
        elif self.below_range:
            range_pos = 'below_range'
        
        position_type = 'flat'
        if self.position > 0:
            position_type = 'long'
        elif self.position < 0:
            position_type = 'short'
        
        return f"{self.session_id}_{self.step}_{range_pos}_{position_type}_{self.range_quality}_{self.volatility_level}"

class LondonRangeBreakMCTS:
    """
    Monte Carlo Tree Search optimized for London Range Break strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MCTS for London Range Break
        
        Args:
            config (Dict): MCTS configuration
        """
        self.exploration_constant = config.get('exploration_constant', 1.414)
        self.max_iterations = config.get('max_iterations', 1000)
        self.max_depth = config.get('max_depth', 20)
        self.rollout_depth = config.get('rollout_depth', 10)
        
        # London Range Break specific parameters
        self.range_break_reward = config.get('range_break_reward', 2.0)
        self.false_breakout_penalty = config.get('false_breakout_penalty', -1.0)
        self.time_decay_factor = config.get('time_decay_factor', 0.95)
        
        # MCTS tree storage
        self.tree = defaultdict(lambda: {
            'visits': 0,
            'value': 0.0,
            'children': {},
            'parent': None,
            'action': None,
            'state': None
        })
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'average_depth': 0,
            'best_rewards': [],
            'exploration_paths': []
        }
    
    def search(self, initial_state: LondonRangeTradingState, 
              ml_predictor: Optional[Any] = None) -> Tuple[str, float]:
        """
        Perform MCTS search to find optimal action
        
        Args:
            initial_state (LondonRangeTradingState): Starting state
            ml_predictor (optional): ML model for state evaluation
            
        Returns:
            Tuple[str, float]: (best_action, confidence)
        """
        logger.info(f"Starting MCTS search from state: {initial_state.get_state_signature()}")
        
        root_id = initial_state.get_state_signature()
        self.tree[root_id]['state'] = initial_state
        
        # Perform MCTS iterations
        for iteration in range(self.max_iterations):
            # Selection and Expansion
            leaf_id, path = self._select_and_expand(root_id)
            
            # Simulation (rollout)
            reward = self._simulate(self.tree[leaf_id]['state'], ml_predictor)
            
            # Backpropagation
            self._backpropagate(path, reward)
            
            # Early termination if confidence is high
            if iteration > 100 and iteration % 100 == 0:
                best_action, confidence = self._get_best_action(root_id)
                if confidence > 0.8:
                    logger.info(f"Early termination at iteration {iteration}, confidence: {confidence:.3f}")
                    break
        
        # Get final best action
        best_action, confidence = self._get_best_action(root_id)
        
        # Update search statistics
        self._update_search_stats(root_id)
        
        logger.info(f"MCTS search completed. Best action: {best_action}, confidence: {confidence:.3f}")
        return best_action, confidence
    
    def _select_and_expand(self, root_id: str) -> Tuple[str, List[str]]:
        """Select path through tree and expand if needed"""
        path = [root_id]
        current_id = root_id
        
        # Selection phase - traverse to leaf
        while (self.tree[current_id]['children'] and 
               self.tree[current_id]['visits'] > 0):
            current_id = self._select_child(current_id)
            path.append(current_id)
        
        # Expansion phase - add new child if possible
        current_state = self.tree[current_id]['state']
        if not current_state.is_terminal():
            possible_actions = current_state.get_possible_actions()
            
            # Find unexplored actions
            unexplored_actions = [
                action for action in possible_actions
                if action not in self.tree[current_id]['children']
            ]
            
            if unexplored_actions:
                # Choose action to expand (prioritize breakout actions)
                action = self._choose_expansion_action(unexplored_actions, current_state)
                new_state = self._simulate_action(current_state, action)
                new_id = new_state.get_state_signature()
                
                # Add new node to tree
                self.tree[new_id]['state'] = new_state
                self.tree[new_id]['parent'] = current_id
                self.tree[new_id]['action'] = action
                self.tree[current_id]['children'][action] = new_id
                
                path.append(new_id)
                current_id = new_id
        
        return current_id, path
    
    def _select_child(self, node_id: str) -> str:
        """Select child using UCB1 with London Range Break enhancements"""
        node = self.tree[node_id]
        best_score = float('-inf')
        best_child = None
        
        for action, child_id in node['children'].items():
            child = self.tree[child_id]
            
            if child['visits'] == 0:
                # Unvisited child - prioritize
                ucb_score = float('inf')
            else:
                # UCB1 calculation
                exploitation = child['value'] / child['visits']
                exploration = self.exploration_constant * math.sqrt(
                    math.log(node['visits']) / child['visits']
                )
                
                # London Range Break specific bonuses
                strategy_bonus = self._get_strategy_bonus(action, node['state'])
                
                ucb_score = exploitation + exploration + strategy_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child_id
        
        return best_child
    
    def _get_strategy_bonus(self, action: str, state: LondonRangeTradingState) -> float:
        """Get strategy-specific bonus for action selection"""
        bonus = 0.0
        
        # Bonus for breakout actions in good range conditions
        if state.range_quality == 'good':
            if action in ['buy', 'buy_aggressive'] and state.above_range:
                bonus += 0.2 * state.breakout_strength
            elif action in ['sell', 'sell_aggressive'] and state.below_range:
                bonus += 0.2 * state.breakout_strength
        
        # Penalty for actions in poor conditions
        if state.range_quality == 'poor':
            if action in ['buy', 'sell', 'buy_aggressive', 'sell_aggressive']:
                bonus -= 0.3
        
        # Time-based adjustments
        if state.time_since_london_open > 6:  # Late in session
            if action == 'hold':
                bonus += 0.1  # Prefer holding late in session
        
        # Volatility adjustments
        if state.volatility_level == 'high':
            if action in ['buy_aggressive', 'sell_aggressive']:
                bonus -= 0.2  # Be more cautious in high volatility
        
        return bonus
    
    def _choose_expansion_action(self, actions: List[str], state: LondonRangeTradingState) -> str:
        """Choose which action to expand first (prioritize promising actions)"""
        # Prioritize breakout actions in good conditions
        if state.range_quality == 'good':
            if state.above_range and 'buy' in actions:
                return 'buy'
            elif state.below_range and 'sell' in actions:
                return 'sell'
        
        # Otherwise, random selection
        return random.choice(actions)
    
    def _simulate_action(self, state: LondonRangeTradingState, action: str) -> LondonRangeTradingState:
        """Simulate taking an action from current state"""
        new_state = copy.deepcopy(state)
        new_state.step += 1
        new_state.time_since_london_open = min(8, state.time_since_london_open + 0.25)  # 15min increment
        
        # Simulate action effects
        if action == 'buy':
            if new_state.position <= 0:
                new_state.position = 1.0
                new_state.cash -= new_state.current_price * (1 + new_state.transaction_cost)
        
        elif action == 'buy_aggressive':
            if new_state.position <= 0:
                new_state.position = 1.5  # Larger position
                new_state.cash -= new_state.current_price * 1.5 * (1 + new_state.transaction_cost)
        
        elif action == 'sell':
            if new_state.position >= 0:
                new_state.position = -1.0
                new_state.cash += new_state.current_price * (1 - new_state.transaction_cost)
        
        elif action == 'sell_aggressive':
            if new_state.position >= 0:
                new_state.position = -1.5  # Larger position
                new_state.cash += new_state.current_price * 1.5 * (1 - new_state.transaction_cost)
        
        elif action == 'close_long':
            if new_state.position > 0:
                new_state.cash += new_state.position * new_state.current_price * (1 - new_state.transaction_cost)
                new_state.position = 0.0
        
        elif action == 'close_short':
            if new_state.position < 0:
                new_state.cash -= abs(new_state.position) * new_state.current_price * (1 + new_state.transaction_cost)
                new_state.position = 0.0
        
        elif action == 'close_long_partial':
            if new_state.position > 0:
                close_amount = new_state.position * 0.5
                new_state.cash += close_amount * new_state.current_price * (1 - new_state.transaction_cost)
                new_state.position -= close_amount
        
        elif action == 'close_short_partial':
            if new_state.position < 0:
                close_amount = abs(new_state.position) * 0.5
                new_state.cash -= close_amount * new_state.current_price * (1 + new_state.transaction_cost)
                new_state.position += close_amount
        
        # Update portfolio value
        new_state.portfolio_value = new_state.cash + (new_state.position * new_state.current_price)
        
        # Simulate price movement (simplified)
        new_state.current_price = self._simulate_price_movement(new_state)
        
        # Update state properties
        new_state._update_range_position()
        new_state._update_market_conditions()
        
        return new_state
    
    def _simulate_price_movement(self, state: LondonRangeTradingState) -> float:
        """Simulate price movement (simplified model)"""
        # Simple random walk with bias based on breakout
        base_volatility = 0.001  # 0.1% base volatility
        
        # Increase volatility during breakouts
        if state.above_range or state.below_range:
            volatility = base_volatility * (1 + state.breakout_strength)
        else:
            volatility = base_volatility * 0.5  # Lower volatility in range
        
        # Add slight bias in breakout direction
        bias = 0.0
        if state.above_range and state.range_quality == 'good':
            bias = volatility * 0.3  # Slight upward bias
        elif state.below_range and state.range_quality == 'good':
            bias = -volatility * 0.3  # Slight downward bias
        
        price_change = np.random.normal(bias, volatility)
        new_price = state.current_price * (1 + price_change)
        
        return max(new_price, 0.01)  # Ensure positive price
    
    def _simulate(self, state: LondonRangeTradingState, ml_predictor: Optional[Any] = None) -> float:
        """Perform rollout simulation from given state"""
        current_state = copy.deepcopy(state)
        total_reward = 0.0
        
        for step in range(self.rollout_depth):
            if current_state.is_terminal():
                break
            
            # Get possible actions
            actions = current_state.get_possible_actions()
            if not actions:
                break
            
            # Choose action (use ML predictor if available, otherwise random)
            if ml_predictor and random.random() < 0.7:  # Use ML 70% of the time
                action = self._ml_guided_action_selection(current_state, actions, ml_predictor)
            else:
                action = self._random_policy(current_state, actions)
            
            # Take action
            new_state = self._simulate_action(current_state, action)
            
            # Calculate reward
            reward = self._calculate_reward(current_state, new_state, action)
            total_reward += reward * (self.time_decay_factor ** step)
            
            current_state = new_state
        
        # Add final state value
        final_reward = self._evaluate_final_state(current_state)
        total_reward += final_reward
        
        return total_reward
    
    def _ml_guided_action_selection(self, state: LondonRangeTradingState, 
                                  actions: List[str], ml_predictor: Any) -> str:
        """Use ML predictor to guide action selection"""
        try:
            # Get ML predictions for current state
            features = state.current_features.reshape(1, -1)
            predictions = ml_predictor.predict_proba(features)[0]
            
            # Map predictions to actions
            if len(predictions) >= 2:
                bullish_prob = predictions[1]  # Assuming binary classification
                
                if bullish_prob > 0.6 and state.above_range:
                    preferred_actions = ['buy', 'buy_aggressive']
                elif bullish_prob < 0.4 and state.below_range:
                    preferred_actions = ['sell', 'sell_aggressive']
                else:
                    preferred_actions = ['hold']
                
                # Choose from preferred actions if available
                available_preferred = [a for a in preferred_actions if a in actions]
                if available_preferred:
                    return random.choice(available_preferred)
        
        except Exception as e:
            logger.warning(f"ML guided action selection failed: {str(e)}")
        
        # Fallback to random policy
        return self._random_policy(state, actions)
    
    def _random_policy(self, state: LondonRangeTradingState, actions: List[str]) -> str:
        """Random action selection with strategy-aware weights"""
        weights = []
        
        for action in actions:
            weight = 1.0
            
            # Higher weight for breakout actions in good conditions
            if state.range_quality == 'good':
                if action in ['buy', 'buy_aggressive'] and state.above_range:
                    weight = 3.0
                elif action in ['sell', 'sell_aggressive'] and state.below_range:
                    weight = 3.0
            
            # Lower weight for risky actions in poor conditions
            if state.range_quality == 'poor':
                if action in ['buy_aggressive', 'sell_aggressive']:
                    weight = 0.3
            
            # Prefer hold in uncertain conditions
            if action == 'hold' and state.range_quality == 'poor':
                weight = 2.0
            
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(actions)
        
        weights = [w / total_weight for w in weights]
        return np.random.choice(actions, p=weights)
    
    def _calculate_reward(self, old_state: LondonRangeTradingState, 
                         new_state: LondonRangeTradingState, action: str) -> float:
        """Calculate reward for state transition"""
        # Base reward from portfolio value change
        portfolio_change = new_state.portfolio_value - old_state.portfolio_value
        base_reward = portfolio_change / old_state.portfolio_value  # Percentage change
        
        # Strategy-specific rewards
        strategy_reward = 0.0
        
        # Successful breakout reward
        if action in ['buy', 'buy_aggressive'] and old_state.above_range:
            if new_state.current_price > old_state.current_price:
                strategy_reward += self.range_break_reward * old_state.breakout_strength
        
        elif action in ['sell', 'sell_aggressive'] and old_state.below_range:
            if new_state.current_price < old_state.current_price:
                strategy_reward += self.range_break_reward * old_state.breakout_strength
        
        # False breakout penalty
        if action in ['buy', 'buy_aggressive'] and old_state.above_range:
            if new_state.current_price < old_state.pre_london_high:
                strategy_reward += self.false_breakout_penalty
        
        elif action in ['sell', 'sell_aggressive'] and old_state.below_range:
            if new_state.current_price > old_state.pre_london_low:
                strategy_reward += self.false_breakout_penalty
        
        # Time-based rewards
        time_reward = 0.0
        if old_state.time_since_london_open < 2:  # Early in session
            if action in ['buy', 'sell'] and old_state.range_quality == 'good':
                time_reward = 0.1
        
        # Risk management penalty
        risk_penalty = 0.0
        if new_state.portfolio_value < old_state.portfolio_value * 0.95:  # 5% loss
            risk_penalty = -0.5
        
        total_reward = base_reward + strategy_reward + time_reward + risk_penalty
        return total_reward
    
    def _evaluate_final_state(self, state: LondonRangeTradingState) -> float:
        """Evaluate final state value"""
        # Portfolio performance
        portfolio_return = (state.portfolio_value / state.cash) - 1
        
        # Position adjustment (prefer flat positions at end)
        position_penalty = abs(state.position) * 0.1
        
        # Time efficiency (prefer completing trades earlier)
        time_efficiency = max(0, (8 - state.time_since_london_open) / 8 * 0.2)
        
        final_value = portfolio_return - position_penalty + time_efficiency
        return final_value
    
    def _backpropagate(self, path: List[str], reward: float):
        """Backpropagate reward through the path"""
        for node_id in reversed(path):
            node = self.tree[node_id]
            node['visits'] += 1
            node['value'] += reward
            
            # Decay reward for parent nodes
            reward *= 0.95
    
    def _get_best_action(self, root_id: str) -> Tuple[str, float]:
        """Get best action based on visit counts and values"""
        root = self.tree[root_id]
        
        if not root['children']:
            return 'hold', 0.0
        
        best_action = None
        best_score = float('-inf')
        total_visits = sum(self.tree[child_id]['visits'] for child_id in root['children'].values())
        
        for action, child_id in root['children'].items():
            child = self.tree[child_id]
            
            if child['visits'] > 0:
                # Combine average value and visit frequency
                avg_value = child['value'] / child['visits']
                visit_weight = child['visits'] / total_visits
                
                # Weighted score
                score = 0.7 * avg_value + 0.3 * visit_weight
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        # Calculate confidence based on visit distribution
        if best_action and total_visits > 0:
            best_child_visits = self.tree[root['children'][best_action]]['visits']
            confidence = best_child_visits / total_visits
        else:
            confidence = 0.0
            best_action = 'hold'
        
        return best_action, confidence
    
    def _update_search_stats(self, root_id: str):
        """Update search statistics"""
        self.search_stats['total_searches'] += 1
        
        # Calculate average depth
        total_depth = 0
        node_count = 0
        
        def calculate_depth(node_id, current_depth):
            nonlocal total_depth, node_count
            node = self.tree[node_id]
            total_depth += current_depth
            node_count += 1
            
            for child_id in node['children'].values():
                calculate_depth(child_id, current_depth + 1)
        
        calculate_depth(root_id, 0)
        
        if node_count > 0:
            avg_depth = total_depth / node_count
            self.search_stats['average_depth'] = avg_depth
    
    def get_search_statistics(self) -> Dict:
        """Get MCTS search statistics"""
        return {
            'total_searches': self.search_stats['total_searches'],
            'average_depth': self.search_stats['average_depth'],
            'tree_size': len(self.tree),
            'best_rewards': self.search_stats['best_rewards'][-10:] if self.search_stats['best_rewards'] else []
        }
    
    def clear_tree(self):
        """Clear the MCTS tree to free memory"""
        self.tree.clear()
        logger.info("MCTS tree cleared")