import os
import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from tqdm import tqdm

from app.data.data_loader import StockDataLoader
from config import TICKERS, DATA_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_stock_data(
    tickers: List[str],
    start_date: str = '2015-01-01',
    end_date: str = None,
    interval: str = '1d',
    save_raw: bool = True
) -> None:
    """
    Download stock data for multiple tickers.
    
    Args:
        tickers: List of stock tickers to download
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: today)
        interval: Data interval ('1d', '1wk', '1mo')
        save_raw: Whether to save the raw data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data_loader = StockDataLoader(DATA_DIR)
    
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            df = data_loader.fetch_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                save_raw=save_raw
            )
            logger.info(f"Successfully downloaded {len(df)} rows for {ticker}")
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")

def create_metadata_file():
    """Create a metadata file with information about the dataset."""
    metadata = {
        'creation_date': datetime.now().strftime('%Y-%m-%d'),
        'data_source': 'Yahoo Finance',
        'tickers': TICKERS,
        'time_updated': datetime.now().isoformat(),
        'description': 'Stock market data for training and evaluation',
        'data_processing': 'Raw OHLCV data with basic feature engineering',
        'license': 'CC0 1.0 Universal (Public Domain)',
    }
    
    metadata_path = DATA_DIR / 'raw' / 'METADATA.md'
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            if isinstance(value, list):
                value = ', '.join(value)
            f.write(f'## {key.replace("_", " ").title()}\n{value}\n\n')
    
    logger.info(f"Created metadata file at {metadata_path}")

if __name__ == "__main__":
    # Download data for all tickers
    download_stock_data(
        tickers=TICKERS,
        start_date='2015-01-01',
        interval='1d',
        save_raw=True
    )
    
    # Create metadata file
    create_metadata_file()
    
    logger.info("Data download and processing complete!")
