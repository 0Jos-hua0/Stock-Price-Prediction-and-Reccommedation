import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

from app.core.data_enhanced_processor import EnhancedStockDataProcessor
from config import (
    DATA_DIR, 
    TICKERS, 
    TRAIN_CONFIG, 
    DATA_PROCESSING,
    LOGS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to handle data processing pipeline."""
    
    def __init__(self):
        """Initialize the data processor with configuration."""
        # Get the project root directory (two levels up from backend directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Set the data directories
        self.raw_dir = os.path.join(project_root, 'data', 'raw')
        self.processed_dir = os.path.join(project_root, 'data', 'processed')
        
        # Print debug information
        print(f"Project root: {project_root}")
        print(f"Raw data directory: {self.raw_dir}")
        print(f"Processed data directory: {self.processed_dir}")
        
        # Check if raw directory exists
        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
            
        # Print contents of raw directory for debugging
        print("\nContents of raw data directory:")
        try:
            for item in os.listdir(self.raw_dir):
                item_path = os.path.join(self.raw_dir, item)
                print(f"- {item} (file)" if os.path.isfile(item_path) else f"- {item} (directory)")
        except Exception as e:
            print(f"Error listing raw directory contents: {e}")
        
        # Create processed directory structure
        for subdir in ['train', 'val', 'test']:
            dir_path = os.path.join(self.processed_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created/validated directory: {dir_path}")
        
        # Set training parameters
        self.sequence_length = TRAIN_CONFIG['sequence_length']
        self.prediction_horizon = TRAIN_CONFIG['prediction_horizon']
        self.test_size = TRAIN_CONFIG['test_size']
        self.val_size = TRAIN_CONFIG['val_size']
        
        # Initialize data processor
        self.processor = EnhancedStockDataProcessor(
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            test_size=self.test_size,
            val_size=self.val_size,
            scale_features=DATA_PROCESSING['scale_features'],
            scale_target=DATA_PROCESSING['scale_target'],
            feature_scaler_type=DATA_PROCESSING['feature_scaler'],
            target_scaler_type=DATA_PROCESSING['target_scaler'],
            add_technical_indicators=DATA_PROCESSING['add_technical_indicators'],
            add_time_features=DATA_PROCESSING['add_time_features'],
            drop_na=DATA_PROCESSING['drop_na'],
            fill_method=DATA_PROCESSING['fill_method']
        )
    
    def load_raw_data(self, ticker: str) -> pd.DataFrame:
        """Load raw data for a given ticker."""
        import glob
        
        # Look for CSV files in the raw directory
        pattern = os.path.join(self.raw_dir, f"{ticker}_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No data files found for ticker: {ticker}")
        
        # Sort files by modification time (newest first) and get the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        file_path = files[0]
        
        print(f"\nLoading data from: {file_path}")
        
        # Read the CSV file with proper handling of the header
        try:
            # First, read the file to check its structure
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                
            # Check if the first line is a proper header
            if ',' not in first_line or first_line == '':
                # If first line is empty or malformed, skip it and use the second line as header
                df = pd.read_csv(file_path, skiprows=1, header=None, 
                               names=['date', 'close', 'high', 'low', 'open', 'volume', 'ticker'])
                print("Used default column names as the header was malformed")
            else:
                # Read normally if header is present
                df = pd.read_csv(file_path)
                
            # Clean up column names (strip whitespace and convert to lowercase)
            df.columns = df.columns.str.strip().str.lower()
            
            # Ensure date column is in the correct format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # Ensure all required columns are present
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {file_path}: {', '.join(missing_columns)}")
                
            # Convert numeric columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with missing values
            df = df.dropna()
            
            print(f"Successfully loaded {len(df)} rows of data for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            print(f"First line: {first_line}")
            print(f"Second line: {second_line}")
            raise
    
    def process_ticker_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Process data for a single ticker and return train/val/test splits."""
        logger.info(f"Processing data for {ticker}")
        
        try:
            # Load raw data
            df = self.load_raw_data(ticker)
            
            # Process data
            X, y = self.processor.preprocess_data(df, is_training=True)
            
            # Create sequences
            X_seq, y_seq = self.processor.create_sequences(X, y)
            
            # Split into train/val/test
            X_train, X_val, X_test, y_train, y_val, y_test = self.processor.train_val_test_split(
                X_seq, y_seq
            )
            
            # Create DataFrames for each split
            train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
            train_df['target'] = y_train
            
            val_df = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
            val_df['target'] = y_val
            
            test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
            test_df['target'] = y_test
            
            return {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            return None
    
    def save_processed_data(self, ticker: str, data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to disk."""
        for split, df in data.items():
            if df is not None:
                file_path = os.path.join(self.processed_dir, split, f"{ticker}.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {split} data for {ticker} to {file_path}")
    
    def process_all_tickers(self, tickers: List[str]) -> None:
        """Process data for all tickers."""
        for ticker in tqdm(tickers, desc="Processing tickers"):
            try:
                processed_data = self.process_ticker_data(ticker)
                if processed_data:
                    self.save_processed_data(ticker, processed_data)
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {str(e)}")

def main():
    """Main function to run the data processing pipeline."""
    logger.info("Starting data processing pipeline...")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process all tickers
    processor.process_all_tickers(TICKERS)
    
    logger.info("Data processing complete!")

if __name__ == "__main__":
    main()
