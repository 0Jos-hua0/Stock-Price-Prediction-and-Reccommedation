import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.indicators = ['sma_20', 'sma_50', 'rsi_14', 'macd']
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def process_historical_data(self, data: Dict) -> pd.DataFrame:
        """Process raw historical stock data into a cleaned DataFrame with technical indicators."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Ensure required columns exist
            if not all(col in df.columns for col in self.required_columns):
                raise ValueError("Missing required columns in the input data")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing historical data: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Forward fill for missing values
        df.ffill(inplace=True)
        
        # If there are still missing values at the beginning, backfill them
        df.bfill(inplace=True)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the data for model training."""
        try:
            # Select features and target
            features = df[self.indicators].values
            target = df['close'].pct_change().shift(-1).values  # Next day's return
            
            # Remove the last row as it won't have a target
            features = features[:-1]
            target = target[:-1]
            
            # Create sequences
            X, y = [], []
            for i in range(len(features) - sequence_length):
                X.append(features[i:(i + sequence_length)])
                y.append(target[i + sequence_length - 1])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
