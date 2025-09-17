import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """A class to handle data fetching from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the Yahoo Finance data fetcher."""
        self.base_interval_map = {
            '1m': '1d',
            '2m': '1d',
            '5m': '1d',
            '15m': '1d',
            '30m': '1d',
            '60m': '1d',
            '90m': '1d',
            '1h': '1d',
            '1d': '1mo',
            '5d': '1mo',
            '1wk': '1y',
            '1mo': '5y',
            '3mo': '10y',
        }
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = '1y', 
        interval: str = '1d',
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Data period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Download start date string (YYYY-MM-DD)
            end: Download end date string (YYYY-MM-DD)
            
        Returns:
            DataFrame containing historical data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # If start and end are provided, use them; otherwise, use period
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval, prepost=False)
            else:
                # For intraday data, we need to adjust the period
                if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                    # For intraday, we're limited to recent data
                    base_period = self.base_interval_map.get(interval, '1d')
                    df = ticker.history(period=base_period, interval=interval, prepost=False)
                else:
                    df = ticker.history(period=period, interval=interval, prepost=False)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} with period={period}, interval={interval}")
                return None
                
            # Clean up the data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = [col.lower() for col in df.columns]  # Convert to lowercase column names
            df.index.name = 'date'
            
            # Forward fill any missing values
            df = df.ffill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'logo_url': info.get('logo_url', '')
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
