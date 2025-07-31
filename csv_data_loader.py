"""
CSV Data Loader for OHLCV Trading Data
Handles loading and preprocessing of CSV data for trading systems
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataLoader:
    """Handles loading and preprocessing OHLCV data from CSV files"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.processed_data = None
        
    def load_csv_data(self, csv_path: str, 
                      datetime_col: str = 'datetime',
                      open_col: str = 'open',
                      high_col: str = 'high', 
                      low_col: str = 'low',
                      close_col: str = 'close',
                      volume_col: str = 'volume') -> pd.DataFrame:
        """
        Load OHLCV data from CSV file
        
        Args:
            csv_path (str): Path to CSV file
            datetime_col (str): Name of datetime column
            open_col (str): Name of open price column
            high_col (str): Name of high price column
            low_col (str): Name of low price column
            close_col (str): Name of close price column
            volume_col (str): Name of volume column
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        try:
            logger.info(f"Loading data from {csv_path}")
            
            # Load CSV data
            data = pd.read_csv(csv_path)
            
            # Standardize column names
            column_mapping = {
                datetime_col: 'datetime',
                open_col: 'open',
                high_col: 'high',
                low_col: 'low', 
                close_col: 'close',
                volume_col: 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Convert datetime column to datetime type
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            else:
                # If no datetime column, try to parse index
                data.index = pd.to_datetime(data.index)
                
            # Ensure we have required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Add volume column if missing (set to 0 for forex data)
            if 'volume' not in data.columns:
                data['volume'] = 0
                logger.info("Volume column not found, setting to 0 (common for forex data)")
            
            # Sort by datetime
            data = data.sort_index()
            
            # Remove any duplicate timestamps
            data = data[~data.index.duplicated(keep='first')]
            
            # Forward fill any missing values
            data = data.fillna(method='ffill').dropna()
            
            logger.info(f"Loaded {len(data)} rows of data from {data.index[0]} to {data.index[-1]}")
            logger.info(f"Columns: {list(data.columns)}")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        
        Args:
            data (pd.DataFrame, optional): Data to process, uses self.data if None
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        # Price-based features
        data['price_change'] = data['close'].pct_change()
        data['price_range'] = data['high'] - data['low']
        data['price_range_pct'] = data['price_range'] / data['close']
        data['body_size'] = abs(data['close'] - data['open'])
        data['body_size_pct'] = data['body_size'] / data['close']
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            data[f'price_vs_sma_{period}'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']
        
        # Volatility indicators
        data['atr_14'] = self._calculate_atr(data, 14)
        data['volatility_20'] = data['close'].rolling(window=20).std()
        
        # RSI
        data['rsi_14'] = self._calculate_rsi(data['close'], 14)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        
        # London session indicators (assuming UTC timezone)
        london_start = 8  # 8 AM UTC (London market open)
        london_end = 16   # 4 PM UTC (London market close)
        data['is_london_session'] = ((data['hour'] >= london_start) & 
                                    (data['hour'] < london_end)).astype(int)
        
        # Pre-London session (3 AM - 8 AM UTC)
        pre_london_start = 3
        pre_london_end = 8
        data['is_pre_london'] = ((data['hour'] >= pre_london_start) & 
                                (data['hour'] < pre_london_end)).astype(int)
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_london_range_features(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features specifically for London Range Break strategy
        
        Args:
            data (pd.DataFrame, optional): Data to process, uses self.data if None
            
        Returns:
            pd.DataFrame: Data with London Range Break features
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        # Add basic technical indicators first
        data = self.add_technical_indicators(data)
        
        # London Range Break specific features
        data['session_date'] = data.index.date
        
        # Calculate daily pre-London range (3 AM - 8 AM UTC)
        def calculate_daily_range(group):
            pre_london_data = group[(group.index.hour >= 3) & (group.index.hour < 8)]
            if len(pre_london_data) > 0:
                high = pre_london_data['high'].max()
                low = pre_london_data['low'].min()
                range_points = (high - low) / 0.0001  # Assuming 0.0001 as point size for XAUUSD
                return pd.Series({
                    'pre_london_high': high,
                    'pre_london_low': low,
                    'pre_london_range': range_points,
                    'pre_london_midpoint': (high + low) / 2
                })
            else:
                return pd.Series({
                    'pre_london_high': np.nan,
                    'pre_london_low': np.nan,
                    'pre_london_range': np.nan,
                    'pre_london_midpoint': np.nan
                })
        
        # Group by date and calculate daily ranges
        daily_ranges = data.groupby('session_date').apply(calculate_daily_range)
        
        # Merge back to main dataframe
        data = data.merge(daily_ranges, left_on='session_date', right_index=True, how='left')
        
        # Forward fill the daily range values
        data[['pre_london_high', 'pre_london_low', 'pre_london_range', 'pre_london_midpoint']] = \
            data[['pre_london_high', 'pre_london_low', 'pre_london_range', 'pre_london_midpoint']].fillna(method='ffill')
        
        # Calculate break levels
        data['buy_break_level'] = data['pre_london_high'] + 0.0001  # 1 point offset
        data['sell_break_level'] = data['pre_london_low'] - 0.0001  # 1 point offset
        
        # Price position relative to range
        data['price_vs_range_high'] = (data['close'] - data['pre_london_high']) / data['pre_london_high']
        data['price_vs_range_low'] = (data['close'] - data['pre_london_low']) / data['pre_london_low']
        data['price_in_range_pct'] = ((data['close'] - data['pre_london_low']) / 
                                     (data['pre_london_high'] - data['pre_london_low']))
        
        # Range quality indicators
        data['range_quality'] = np.where(
            (data['pre_london_range'] >= 100) & (data['pre_london_range'] <= 5000),
            1, 0
        )
        
        # Volatility relative to range
        data['volatility_vs_range'] = data['atr_14'] / (data['pre_london_high'] - data['pre_london_low'])
        
        # Time since London open
        london_open_hour = 8
        data['hours_since_london_open'] = np.where(
            data['hour'] >= london_open_hour,
            data['hour'] - london_open_hour,
            data['hour'] + (24 - london_open_hour)
        )
        
        # Breakout signals
        data['buy_breakout'] = (data['close'] > data['buy_break_level']).astype(int)
        data['sell_breakout'] = (data['close'] < data['sell_break_level']).astype(int)
        
        # Previous bar breakout for signal confirmation
        data['prev_buy_breakout'] = data['buy_breakout'].shift(1)
        data['prev_sell_breakout'] = data['sell_breakout'].shift(1)
        
        # Fresh breakout (first bar breaking the level)
        data['fresh_buy_breakout'] = ((data['buy_breakout'] == 1) & 
                                     (data['prev_buy_breakout'] == 0)).astype(int)
        data['fresh_sell_breakout'] = ((data['sell_breakout'] == 1) & 
                                      (data['prev_sell_breakout'] == 0)).astype(int)
        
        return data
    
    def get_training_data(self, 
                         target_column: str = 'future_return',
                         lookback_periods: int = 1,
                         feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML models
        
        Args:
            target_column (str): Name of target column
            lookback_periods (int): Number of periods to look back for features
            feature_columns (List[str], optional): Specific features to use
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run prepare_london_range_features first.")
        
        data = self.processed_data.copy()
        
        # Create future returns if not exist
        if target_column not in data.columns:
            if target_column == 'future_return':
                data['future_return'] = data['close'].shift(-1) / data['close'] - 1
            elif target_column == 'future_direction':
                data['future_direction'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Select feature columns
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'session_date', 
                           'future_return', 'future_direction']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Create feature matrix
        features = data[feature_columns].values
        targets = data[target_column].values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
        
        return features, targets
    
    def get_feature_names(self, feature_columns: Optional[List[str]] = None) -> List[str]:
        """Get list of feature names"""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        if feature_columns is None:
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'session_date', 
                           'future_return', 'future_direction']
            feature_columns = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        return feature_columns
    
    def split_data_by_time(self, data: pd.DataFrame, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/validation/test sets
        
        Args:
            data (pd.DataFrame): Data to split
            train_ratio (float): Ratio for training data
            val_ratio (float): Ratio for validation data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test data
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data