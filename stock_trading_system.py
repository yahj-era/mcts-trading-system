"""
Stock Trading System with Monte Carlo Tree Search
A comprehensive ML-based trading system with MCTS decision making
Compatible with Google Colab
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam

# Visualization and metrics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection and preprocessing from Yahoo Finance"""
    
    def __init__(self):
        self.data = None
        self.features = None
        
    def fetch_data(self, symbol, period="2y", interval="1d"):
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        try:
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period, interval=interval)
            logger.info(f"Fetched {len(self.data)} data points for {symbol}")
            return self.data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def engineer_features(self, data=None):
        """
        Engineer technical indicators and features
        
        Args:
            data (pd.DataFrame): OHLCV data, uses self.data if None
        """
        if data is None:
            data = self.data.copy()
        
        if data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Basic price features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Price_Change'] = data['Close'] - data['Open']
        data['Price_Change_Pct'] = data['Price_Change'] / data['Open']
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()
            data[f'Price_SMA_{window}_Ratio'] = data['Close'] / data[f'SMA_{window}']
            data[f'Price_EMA_{window}_Ratio'] = data['Close'] / data[f'EMA_{window}']
        
        # Volatility indicators
        data['Volatility_10'] = data['Returns'].rolling(window=10).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data['RSI_14'] = calculate_rsi(data['Close'])
        data['RSI_7'] = calculate_rsi(data['Close'], 7)
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
        
        # Volume indicators
        data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_10']
        
        # Momentum indicators
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5)
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10)
        
        # Support and Resistance levels
        data['Resistance'] = data['High'].rolling(window=20).max()
        data['Support'] = data['Low'].rolling(window=20).min()
        data['Support_Resistance_Ratio'] = (data['Close'] - data['Support']) / (data['Resistance'] - data['Support'])
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
        
        # Target variables for different prediction horizons
        for horizon in [1, 3, 5]:
            data[f'Future_Return_{horizon}'] = data['Returns'].shift(-horizon)
            data[f'Future_Price_{horizon}'] = data['Close'].shift(-horizon)
            
            # Classification targets (up/down/flat)
            future_returns = data[f'Future_Return_{horizon}']
            data[f'Target_Class_{horizon}'] = np.where(
                future_returns > 0.01, 2,  # Up (>1%)
                np.where(future_returns < -0.01, 0, 1)  # Down (<-1%), Flat (else)
            )
        
        # Drop rows with NaN values
        data = data.dropna()
        
        self.features = data
        logger.info(f"Engineered {len(data.columns)} features")
        return data

class MLPredictor:
    """Machine Learning models for price prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def prepare_features(self, data, target_horizon=1):
        """Prepare features and targets for ML models"""
        
        # Select feature columns (exclude targets and basic OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + \
                      [col for col in data.columns if 'Future_' in col or 'Target_' in col]
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = data[feature_cols].values
        y_reg = data[f'Future_Return_{target_horizon}'].values
        y_class = data[f'Target_Class_{target_horizon}'].values
        
        # Remove rows with NaN targets
        valid_idx = ~(np.isnan(y_reg) | np.isnan(y_class))
        X = X[valid_idx]
        y_reg = y_reg[valid_idx]
        y_class = y_class[valid_idx]
        
        return X, y_reg, y_class
    
    def train_xgboost(self, data, target_horizon=1, test_size=0.2):
        """Train XGBoost model for classification"""
        
        X, y_reg, y_class = self.prepare_features(data, target_horizon)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=test_size, random_state=42, stratify=y_class
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        
        # Store model and scaler
        self.models[f'xgboost_{target_horizon}'] = model
        self.scalers[f'xgboost_{target_horizon}'] = scaler
        
        return model, scaler, accuracy
    
    def build_lstm_model(self, sequence_length, n_features, n_classes=3):
        """Build LSTM model architecture"""
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_data(self, data, sequence_length=20, target_horizon=1):
        """Prepare sequential data for LSTM"""
        
        X, y_reg, y_class = self.prepare_features(data, target_horizon)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y_class[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_lstm(self, data, target_horizon=1, sequence_length=20, test_size=0.2, epochs=50):
        """Train LSTM model"""
        
        X_seq, y_seq = self.prepare_lstm_data(data, sequence_length, target_horizon)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build and train model
        model = self.build_lstm_model(sequence_length, X_train.shape[-1])
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        logger.info(f"LSTM Accuracy: {accuracy:.4f}")
        
        # Store model and scaler
        self.models[f'lstm_{target_horizon}'] = model
        self.scalers[f'lstm_{target_horizon}'] = scaler
        
        return model, scaler, accuracy, history
    
    def build_transformer_model(self, sequence_length, n_features, n_classes=3, d_model=64, n_heads=4):
        """Build Transformer model architecture"""
        
        inputs = tf.keras.Input(shape=(sequence_length, n_features))
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=n_heads, 
            key_dim=d_model
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization()(inputs + attention_output)
        
        # Feed forward
        ffn_output = Dense(d_model * 2, activation='relu')(attention_output)
        ffn_output = Dense(d_model)(ffn_output)
        
        # Add & Norm
        ffn_output = LayerNormalization()(attention_output + ffn_output)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        
        # Classification head
        outputs = Dense(n_classes, activation='softmax')(pooled)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_transformer(self, data, target_horizon=1, sequence_length=20, test_size=0.2, epochs=50):
        """Train Transformer model"""
        
        X_seq, y_seq = self.prepare_lstm_data(data, sequence_length, target_horizon)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build and train model
        model = self.build_transformer_model(sequence_length, X_train.shape[-1])
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        logger.info(f"Transformer Accuracy: {accuracy:.4f}")
        
        # Store model and scaler
        self.models[f'transformer_{target_horizon}'] = model
        self.scalers[f'transformer_{target_horizon}'] = scaler
        
        return model, scaler, accuracy, history
    
    def predict(self, features, model_type='xgboost', target_horizon=1):
        """Make predictions using trained models"""
        
        model_key = f'{model_type}_{target_horizon}'
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained yet")
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Scale features
        if model_type in ['lstm', 'transformer']:
            if len(features.shape) == 2:
                features_scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
            else:
                features_scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        else:
            features_scaled = scaler.transform(features)
        
        # Get predictions
        if model_type in ['lstm', 'transformer']:
            predictions = model.predict(features_scaled)
            probabilities = predictions
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            predicted_classes = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
        
        return predicted_classes, probabilities

if __name__ == "__main__":
    # Example usage
    print("Stock Trading System with MCTS - Data Collection and Feature Engineering")
    
    # Initialize data collector
    collector = DataCollector()
    
    # Fetch sample data
    data = collector.fetch_data('AAPL', period='1y')
    
    if data is not None:
        print(f"Fetched {len(data)} data points")
        
        # Engineer features
        features = collector.engineer_features()
        print(f"Created {len(features.columns)} features")
        print(f"Feature columns: {features.columns.tolist()}")
        
        # Initialize ML predictor
        predictor = MLPredictor()
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        xgb_model, xgb_scaler, xgb_accuracy = predictor.train_xgboost(features)
        
        print("\nData collection and feature engineering completed successfully!")