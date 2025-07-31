"""
ML-Enhanced London Range Break Strategy
Integrates Machine Learning with London Range Break strategy for probability analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import joblib
import os

# ML libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import our custom modules
from csv_data_loader import CSVDataLoader
from london_range_strategy import LondonRangeBreakStrategy, TradeType

logger = logging.getLogger(__name__)

class MLLondonRangeBreakStrategy(LondonRangeBreakStrategy):
    """
    ML-Enhanced London Range Break Strategy
    
    Extends the base London Range Break strategy with ML probability analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ML configuration
        self.ml_config = config.get('ml_config', {})
        self.use_ml_filter = config.get('use_ml_filter', True)
        self.ml_confidence_threshold = config.get('ml_confidence_threshold', 0.6)
        self.ml_models_path = config.get('ml_models_path', 'models/')
        
        # ML models
        self.breakout_success_model = None
        self.direction_model = None
        self.profit_model = None
        self.feature_scaler = None
        self.feature_names = []
        
        # ML state
        self.ml_predictions = {}
        self.ml_probabilities = {}
        
        # Ensure models directory exists
        os.makedirs(self.ml_models_path, exist_ok=True)
    
    def train_ml_models(self, data_loader: CSVDataLoader, 
                       train_data: pd.DataFrame,
                       validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Train ML models for London Range Break strategy
        
        Args:
            data_loader (CSVDataLoader): Data loader with processed features
            train_data (pd.DataFrame): Training data
            validation_data (pd.DataFrame, optional): Validation data
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        logger.info("Training ML models for London Range Break strategy...")
        
        # Prepare training data with London Range Break features
        processed_data = data_loader.prepare_london_range_features(train_data)
        
        # Create target variables
        processed_data = self._create_target_variables(processed_data)
        
        # Get features and targets
        features, feature_names = self._prepare_ml_features(processed_data)
        targets = self._prepare_ml_targets(processed_data)
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | 
                      np.isnan(targets['breakout_success']) |
                      np.isnan(targets['direction']) |
                      np.isnan(targets['profit_category']))
        
        features = features[valid_mask]
        for key in targets:
            targets[key] = targets[key][valid_mask]
        
        logger.info(f"Training data shape: {features.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
        # Scale features
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        self.feature_names = feature_names
        
        # Split data if no validation set provided
        if validation_data is None:
            X_train, X_val, y_train_dict, y_val_dict = self._split_ml_data(
                features_scaled, targets, test_size=0.2
            )
        else:
            X_train, X_val = features_scaled, self._prepare_validation_data(validation_data, data_loader)
            y_train_dict, y_val_dict = targets, self._prepare_validation_targets(validation_data)
        
        # Train models
        performance_metrics = {}
        
        # 1. Breakout Success Model (XGBoost)
        logger.info("Training breakout success prediction model...")
        self.breakout_success_model = self._train_breakout_success_model(
            X_train, y_train_dict['breakout_success'],
            X_val, y_val_dict['breakout_success']
        )
        performance_metrics['breakout_success_accuracy'] = self._evaluate_model(
            self.breakout_success_model, X_val, y_val_dict['breakout_success']
        )
        
        # 2. Direction Model (Random Forest)
        logger.info("Training direction prediction model...")
        self.direction_model = self._train_direction_model(
            X_train, y_train_dict['direction'],
            X_val, y_val_dict['direction']
        )
        performance_metrics['direction_accuracy'] = self._evaluate_model(
            self.direction_model, X_val, y_val_dict['direction']
        )
        
        # 3. Profit Category Model (Gradient Boosting)
        logger.info("Training profit prediction model...")
        self.profit_model = self._train_profit_model(
            X_train, y_train_dict['profit_category'],
            X_val, y_val_dict['profit_category']
        )
        performance_metrics['profit_accuracy'] = self._evaluate_model(
            self.profit_model, X_val, y_val_dict['profit_category']
        )
        
        # Save models
        self._save_models()
        
        logger.info("ML model training completed!")
        logger.info(f"Performance metrics: {performance_metrics}")
        
        return performance_metrics
    
    def _create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML training"""
        data = data.copy()
        
        # Forward price movements for different horizons
        data['price_1h'] = data['close'].shift(-4)  # 1 hour ahead (15min * 4)
        data['price_4h'] = data['close'].shift(-16)  # 4 hours ahead
        data['price_8h'] = data['close'].shift(-32)  # 8 hours ahead
        
        # Breakout success (price continues in breakout direction)
        data['buy_breakout_success'] = np.where(
            (data['fresh_buy_breakout'] == 1) & 
            (data['price_4h'] > data['close'] * 1.001),  # 0.1% profit
            1, 0
        )
        
        data['sell_breakout_success'] = np.where(
            (data['fresh_sell_breakout'] == 1) & 
            (data['price_4h'] < data['close'] * 0.999),  # 0.1% profit
            1, 0
        )
        
        data['breakout_success'] = np.maximum(
            data['buy_breakout_success'], 
            data['sell_breakout_success']
        )
        
        # Direction prediction (1 hour ahead)
        data['future_direction'] = np.where(
            data['price_1h'] > data['close'], 1, 0
        )
        
        # Profit categories (based on 4-hour performance)
        data['profit_pct'] = (data['price_4h'] - data['close']) / data['close'] * 100
        data['profit_category'] = pd.cut(
            data['profit_pct'],
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=[0, 1, 2, 3, 4]  # Loss, Small Loss, Neutral, Small Profit, Large Profit
        ).astype(float)
        
        return data
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for ML models"""
        # Select relevant features for London Range Break strategy
        feature_columns = [
            # Range features
            'pre_london_range', 'range_quality', 'price_in_range_pct',
            'volatility_vs_range', 'price_vs_range_high', 'price_vs_range_low',
            
            # Technical indicators
            'rsi_14', 'atr_14', 'volatility_20', 'bb_position',
            'macd', 'macd_signal', 'macd_histogram',
            
            # Price action
            'price_change', 'price_range_pct', 'body_size_pct',
            'upper_shadow', 'lower_shadow',
            
            # Moving averages
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
            
            # Time features
            'hour', 'day_of_week', 'is_london_session', 'is_pre_london',
            'hours_since_london_open',
            
            # Breakout features
            'buy_breakout', 'sell_breakout'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        features = data[available_features].values
        
        return features, available_features
    
    def _prepare_ml_targets(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare target variables for ML models"""
        return {
            'breakout_success': data['breakout_success'].values,
            'direction': data['future_direction'].values,
            'profit_category': data['profit_category'].values
        }
    
    def _split_ml_data(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                      test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Split data for training and validation"""
        # Use stratified split based on breakout success
        X_train, X_val, _, _ = train_test_split(
            features, targets['breakout_success'],
            test_size=test_size, stratify=targets['breakout_success'],
            random_state=42
        )
        
        # Get corresponding indices
        train_indices = np.isin(features, X_train).all(axis=1)
        val_indices = ~train_indices
        
        y_train_dict = {key: targets[key][train_indices] for key in targets}
        y_val_dict = {key: targets[key][val_indices] for key in targets}
        
        return X_train, X_val, y_train_dict, y_val_dict
    
    def _train_breakout_success_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train XGBoost model for breakout success prediction"""
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        return calibrated_model
    
    def _train_direction_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train Random Forest model for direction prediction"""
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        return calibrated_model
    
    def _train_profit_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train Gradient Boosting model for profit prediction"""
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        try:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Model accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            return auc
        except:
            logger.info(f"Model accuracy: {accuracy:.3f}")
            return accuracy
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.breakout_success_model, 
                       os.path.join(self.ml_models_path, 'breakout_success_model.pkl'))
            joblib.dump(self.direction_model, 
                       os.path.join(self.ml_models_path, 'direction_model.pkl'))
            joblib.dump(self.profit_model, 
                       os.path.join(self.ml_models_path, 'profit_model.pkl'))
            joblib.dump(self.feature_scaler, 
                       os.path.join(self.ml_models_path, 'feature_scaler.pkl'))
            
            # Save feature names
            with open(os.path.join(self.ml_models_path, 'feature_names.txt'), 'w') as f:
                for feature in self.feature_names:
                    f.write(f"{feature}\n")
            
            logger.info(f"Models saved to {self.ml_models_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            self.breakout_success_model = joblib.load(
                os.path.join(self.ml_models_path, 'breakout_success_model.pkl'))
            self.direction_model = joblib.load(
                os.path.join(self.ml_models_path, 'direction_model.pkl'))
            self.profit_model = joblib.load(
                os.path.join(self.ml_models_path, 'profit_model.pkl'))
            self.feature_scaler = joblib.load(
                os.path.join(self.ml_models_path, 'feature_scaler.pkl'))
            
            # Load feature names
            with open(os.path.join(self.ml_models_path, 'feature_names.txt'), 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            logger.info("ML models loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load ML models: {str(e)}")
            return False
    
    def _get_ml_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get ML predictions for current market state"""
        if (self.breakout_success_model is None or 
            self.direction_model is None or 
            self.profit_model is None):
            return {}
        
        try:
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Get predictions
            breakout_prob = self.breakout_success_model.predict_proba(features_scaled)[0, 1]
            direction_prob = self.direction_model.predict_proba(features_scaled)[0, 1]
            profit_pred = self.profit_model.predict(features_scaled)[0]
            profit_proba = self.profit_model.predict_proba(features_scaled)[0]
            
            return {
                'breakout_success_probability': breakout_prob,
                'bullish_probability': direction_prob,
                'bearish_probability': 1 - direction_prob,
                'profit_category': profit_pred,
                'profit_probabilities': profit_proba
            }
            
        except Exception as e:
            logger.error(f"Error getting ML predictions: {str(e)}")
            return {}
    
    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """
        Generate ML-enhanced trading signals
        
        Args:
            data (pd.DataFrame): OHLCV data up to current time
            current_time (datetime): Current timestamp
            
        Returns:
            List[Dict]: List of enhanced signals with ML probabilities
        """
        # Get base signals from parent class
        base_signals = super().generate_signals(data, current_time)
        
        if not base_signals or not self.use_ml_filter:
            return base_signals
        
        enhanced_signals = []
        
        for signal in base_signals:
            # Get current market features
            try:
                # Add London Range Break features to current data
                from csv_data_loader import CSVDataLoader
                loader = CSVDataLoader()
                loader.data = data
                processed_data = loader.prepare_london_range_features()
                
                # Get features for current timestamp
                current_features, _ = self._prepare_ml_features(processed_data.tail(1))
                
                if len(current_features) > 0:
                    # Get ML predictions
                    ml_predictions = self._get_ml_predictions(current_features[-1])
                    
                    if ml_predictions:
                        # Enhance signal with ML predictions
                        enhanced_signal = signal.copy()
                        enhanced_signal.update(ml_predictions)
                        
                        # Calculate enhanced confidence
                        enhanced_confidence = self._calculate_enhanced_confidence(
                            signal, ml_predictions
                        )
                        enhanced_signal['ml_confidence'] = enhanced_confidence
                        enhanced_signal['original_confidence'] = signal['confidence']
                        enhanced_signal['confidence'] = enhanced_confidence
                        
                        # Apply ML filter
                        if enhanced_confidence >= self.ml_confidence_threshold:
                            enhanced_signals.append(enhanced_signal)
                            logger.info(f"ML-enhanced signal approved: {enhanced_confidence:.3f}")
                        else:
                            logger.info(f"ML filter rejected signal: {enhanced_confidence:.3f}")
                    else:
                        # If ML predictions fail, use original signal
                        enhanced_signals.append(signal)
                else:
                    enhanced_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error in ML enhancement: {str(e)}")
                enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def _calculate_enhanced_confidence(self, signal: Dict, ml_predictions: Dict) -> float:
        """Calculate enhanced confidence using ML predictions"""
        base_confidence = signal['confidence']
        
        if not ml_predictions:
            return base_confidence
        
        # Weight factors
        breakout_weight = 0.4
        direction_weight = 0.3
        profit_weight = 0.3
        
        # Breakout success probability
        breakout_score = ml_predictions.get('breakout_success_probability', 0.5)
        
        # Direction alignment
        signal_type = signal['signal_type']
        if signal_type == 'buy':
            direction_score = ml_predictions.get('bullish_probability', 0.5)
        else:
            direction_score = ml_predictions.get('bearish_probability', 0.5)
        
        # Profit expectation (higher categories are better)
        profit_category = ml_predictions.get('profit_category', 2)
        profit_score = min(1.0, profit_category / 4.0)  # Normalize to 0-1
        
        # Combine scores
        ml_confidence = (
            breakout_weight * breakout_score +
            direction_weight * direction_score +
            profit_weight * profit_score
        )
        
        # Blend with base confidence
        enhanced_confidence = 0.6 * ml_confidence + 0.4 * base_confidence
        
        return enhanced_confidence
    
    def get_ml_analysis(self, data: pd.DataFrame) -> Dict:
        """Get detailed ML analysis for current market state"""
        if not self.breakout_success_model:
            return {}
        
        try:
            # Prepare current features
            from csv_data_loader import CSVDataLoader
            loader = CSVDataLoader()
            loader.data = data
            processed_data = loader.prepare_london_range_features()
            
            current_features, feature_names = self._prepare_ml_features(processed_data.tail(1))
            
            if len(current_features) == 0:
                return {}
            
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(current_features[-1])
            
            # Get feature importance (for XGBoost model)
            feature_importance = {}
            if hasattr(self.breakout_success_model.base_estimator, 'feature_importances_'):
                importances = self.breakout_success_model.base_estimator.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            analysis = {
                'ml_predictions': ml_predictions,
                'feature_importance': feature_importance,
                'market_regime': self._classify_market_regime(ml_predictions),
                'trading_recommendation': self._get_trading_recommendation(ml_predictions)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {str(e)}")
            return {}
    
    def _classify_market_regime(self, ml_predictions: Dict) -> str:
        """Classify current market regime based on ML predictions"""
        if not ml_predictions:
            return "unknown"
        
        breakout_prob = ml_predictions.get('breakout_success_probability', 0.5)
        bullish_prob = ml_predictions.get('bullish_probability', 0.5)
        
        if breakout_prob > 0.7:
            if bullish_prob > 0.6:
                return "bullish_breakout"
            elif bullish_prob < 0.4:
                return "bearish_breakout"
            else:
                return "neutral_breakout"
        elif breakout_prob < 0.3:
            return "range_bound"
        else:
            return "uncertain"
    
    def _get_trading_recommendation(self, ml_predictions: Dict) -> str:
        """Get trading recommendation based on ML predictions"""
        if not ml_predictions:
            return "hold"
        
        breakout_prob = ml_predictions.get('breakout_success_probability', 0.5)
        bullish_prob = ml_predictions.get('bullish_probability', 0.5)
        profit_category = ml_predictions.get('profit_category', 2)
        
        if breakout_prob > 0.7 and profit_category >= 3:
            if bullish_prob > 0.6:
                return "strong_buy"
            elif bullish_prob < 0.4:
                return "strong_sell"
        elif breakout_prob > 0.6:
            if bullish_prob > 0.55:
                return "buy"
            elif bullish_prob < 0.45:
                return "sell"
        
        return "hold"