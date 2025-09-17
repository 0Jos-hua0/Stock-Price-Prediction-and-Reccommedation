import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class StockDataLoader:
    """A class to handle loading and managing stock market data."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Root directory for storing data (default: project root/data)
        """
        if data_dir is None:
            # Go up two levels from this file's directory to reach project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.data_dir = os.path.join(project_root, 'data')
        else:
            self.data_dir = data_dir
            
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: str = '2015-01-01',
        end_date: Optional[str] = None,
        interval: str = '1d',
        save_raw: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval ('1d', '1wk', '1mo')
            save_raw: Whether to save the raw data
            
        Returns:
            DataFrame with stock data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        try:
            # Fetch data using yfinance
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
                
            # Reset index to make date a column
            df = df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Save raw data
            if save_raw:
                raw_path = os.path.join(self.raw_dir, f"{ticker}_{start_date}_{end_date}.csv")
                df.to_csv(raw_path, index=False)
                logger.info(f"Saved raw data to {raw_path}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def load_processed_data(
        self, 
        ticker: str, 
        data_type: str = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load processed data from disk.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data ('train', 'val', 'test')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with processed data
        """
        file_path = os.path.join(self.processed_dir, data_type, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No processed data found for {ticker} ({data_type})")
            
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Filter by date if specified
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
        return df
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        ticker: str, 
        data_type: str
    ) -> None:
        """
        Save processed data to disk.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker symbol
            data_type: Type of data ('train', 'val', 'test')
        """
        save_dir = os.path.join(self.processed_dir, data_type)
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"{ticker}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {data_type} data to {file_path}")


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = StockDataLoader()
    
    # Example: Fetch and save data for multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in tickers:
        try:
            df = data_loader.fetch_stock_data(ticker)
            print(f"Fetched {len(df)} rows for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
