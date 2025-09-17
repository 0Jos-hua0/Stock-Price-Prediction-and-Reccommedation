import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedStockDataProcessor:
    """
    Enhanced data processor for stock market data with feature engineering
    and sequence generation for LSTM-CNN models.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        scale_features: bool = True,
        scale_target: bool = True,
        feature_scaler_type: str = 'standard',  # 'minmax', 'standard', 'robust', or None
        target_scaler_type: str = 'standard',
        add_technical_indicators: bool = True,
        add_time_features: bool = True,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'close',
        drop_na: bool = True,
        fill_method: str = 'ffill',  # 'ffill', 'bfill', 'mean', 'median', or 'drop'
    ):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Number of time steps to look back for each sample
            prediction_horizon: Number of time steps to predict ahead
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
            scale_features: Whether to scale feature columns
            scale_target: Whether to scale target column
            feature_scaler_type: Type of scaler for features ('minmax', 'standard', 'robust')
            target_scaler_type: Type of scaler for target
            add_technical_indicators: Whether to add technical indicators
            add_time_features: Whether to add time-based features
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            drop_na: Whether to drop rows with NaN values
            fill_method: Method to fill NaN values ('ffill', 'bfill', 'mean', 'median', 'drop')
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.feature_scaler_type = feature_scaler_type
        self.target_scaler_type = target_scaler_type
        self.add_technical_indicators = add_technical_indicators
        self.add_time_features = add_time_features
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.drop_na = drop_na
        self.fill_method = fill_method
        
        # Initialize scalers
        self.feature_scaler = self._get_scaler(feature_scaler_type) if scale_features else None
        self.target_scaler = self._get_scaler(target_scaler_type) if scale_target else None
        
        # Store column names after feature engineering
        self.feature_columns_processed = None
    
    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler based on type."""
        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns for technical indicators. Skipping...")
            return df
        
        # Add moving averages
        df['sma_5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        # Add exponential moving averages
        df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # Add RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # Add MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_bbm'] = bollinger.bollinger_mavg()
        
        # Add ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close'],
            window=14
        ).average_true_range()
        
        # Add OBV (On-Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], 
            volume=df['volume']
        ).on_balance_volume()
        
        # Add Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Add ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).adx()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the DataFrame."""
        if 'date' not in df.columns and df.index.name == 'date':
            df = df.reset_index()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Time features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
            df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
            df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
            
            # Cyclical encoding for periodic features
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            
            # Remove the original columns if they exist
            df = df.drop(columns=['day_of_week', 'month'], errors='ignore')
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        if self.drop_na:
            return df.dropna()
        
        if self.fill_method == 'ffill':
            return df.ffill()
        elif self.fill_method == 'bfill':
            return df.bfill()
        elif self.fill_method == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif self.fill_method == 'median':
            return df.fillna(df.median(numeric_only=True))
        
        return df
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
        target: np.ndarray = None,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences of data for time series prediction.
        
        Args:
            data: Input features (n_samples, n_features)
            target: Target values (n_samples,)
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X, y) where X has shape (n_sequences, sequence_length, n_features)
            and y has shape (n_sequences,)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        X, y = [], []
        
        for i in range(len(data) - sequence_length - self.prediction_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            if target is not None:
                y.append(target[i + sequence_length + self.prediction_horizon - 1])
        
        X = np.array(X)
        
        if target is not None:
            return X, np.array(y)
        return X, None
    
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Preprocess the input DataFrame.
        
        Args:
            df: Input DataFrame with stock data
            is_training: Whether this is training data (affects fitting scalers)
            
        Returns:
            Tuple of (processed DataFrame, target values)
        """
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Ensure date is the index
        if 'date' in df.columns and df.index.name != 'date':
            df = df.set_index('date')
        
        # Add technical indicators if specified
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Add time features if specified
        if self.add_time_features:
            df = self._add_time_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Define feature and target columns
        if self.feature_columns is None:
            # Use all numeric columns except the target as features
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            self.feature_columns = [col for col in numeric_cols if col != self.target_column]
        
        # Ensure all specified columns exist
        missing_cols = [col for col in self.feature_columns + [self.target_column] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Scale features if specified
        if self.scale_features and self.feature_scaler is not None:
            if is_training:
                X = self.feature_scaler.fit_transform(X)
            else:
                X = self.feature_scaler.transform(X)
        
        # Scale target if specified
        if self.scale_target and self.target_scaler is not None:
            y = y.reshape(-1, 1)
            if is_training:
                y = self.target_scaler.fit_transform(y)
            else:
                y = self.target_scaler.transform(y)
            y = y.ravel()
        
        # Store processed feature columns
        self.feature_columns_processed = self.feature_columns.copy()
        
        return X, y
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Preprocess the data
        X, y = self.preprocess_data(df, is_training=True)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, 
            test_size=self.val_size,
            random_state=self.random_state,
            shuffle=False  # Important for time series data
        )
        
        return X_train, X_val, y_train, y_val
    
    def prepare_test_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare test data.
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            Tuple of (X_test, y_test)
        """
        # Preprocess the data (without fitting scalers)
        X, y = self.preprocess_data(df, is_training=False)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        return X_seq, y_seq
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform the target values if they were scaled."""
        if not self.scale_target or self.target_scaler is None:
            return y
        
        y = y.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y).ravel()
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        if not self.feature_columns_processed:
            logger.warning("No processed feature columns found")
            return {}
        
        return dict(zip(self.feature_columns_processed, model.feature_importances_))
