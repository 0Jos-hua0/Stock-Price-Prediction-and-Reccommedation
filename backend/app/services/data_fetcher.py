import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import time
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """A class to handle data fetching from Polygon.io API."""
    
    BASE_URL = "https://api.polygon.io/"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the data fetcher with an API key."""
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon.io API key not provided and not found in environment variables.")
        
        self.session = requests.Session()
        self.session.params = {'apiKey': self.api_key}
    
    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> Dict:
        """Make a GET request to the Polygon.io API."""
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    def get_historical_data(
        self, 
        symbol: str, 
        from_date: Optional[Union[str, datetime]] = None, 
        to_date: Optional[Union[str, datetime]] = None,
        timespan: str = 'day',
        limit: int = 5000
    ) -> Dict:
        """
        Get historical stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            from_date: Start date (default: 1 year ago)
            to_date: End date (default: today)
            timespan: The timespan for the data ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            limit: Maximum number of results to return (max 50000)
            
        Returns:
            Dictionary containing historical data
        """
        # Set default date range if not provided
        if to_date is None:
            to_date = datetime.now()
        elif isinstance(to_date, str):
            to_date = datetime.strptime(to_date, '%Y-%m-%d')
            
        if from_date is None:
            from_date = to_date - timedelta(days=365)
        elif isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d')
        
        # Format dates for API
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        # Build the endpoint
        endpoint = f"v2/aggs/ticker/{symbol.upper()}/range/1/{timespan}/{from_str}/{to_str}"
        
        # Set up parameters
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': min(limit, 50000)  # Polygon's max limit is 50000
        }
        
        try:
            # Make the request
            data = self._make_request(endpoint, params=params)
            
            # Handle pagination if there are more results
            while 'next_url' in data and len(data.get('results', [])) < limit:
                next_url = data['next_url']
                next_data = self._make_request(next_url.replace(self.BASE_URL, ''))
                if 'results' in next_data:
                    data['results'].extend(next_data['results'])
                if 'next_url' not in next_data:
                    break
                
                # Respect rate limits (5 requests per minute for free tier)
                time.sleep(0.2)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information for a given symbol."""
        endpoint = f"v1/meta/symbols/{symbol.upper()}/company"
        return self._make_request(endpoint)
    
    def get_dividends(self, symbol: str) -> Dict:
        """Get dividend information for a given symbol."""
        endpoint = f"v2/reference/dividends/{symbol.upper()}"
        return self._make_request(endpoint)
    
    def get_splits(self, symbol: str) -> Dict:
        """Get stock split information for a given symbol."""
        endpoint = f"v2/reference/splits/{symbol.upper()}"
        return self._make_request(endpoint)
    
    def get_ticker_news(
        self, 
        symbol: Optional[str] = None, 
        limit: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict:
        """
        Get news articles for a given symbol or all tickers.
        
        Args:
            symbol: Optional stock ticker symbol
            limit: Maximum number of news articles to return (max 1000)
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing news articles
        """
        endpoint = "v2/reference/news"
        
        params = {
            'limit': min(limit, 1000)  # Polygon's max limit is 1000
        }
        
        if symbol:
            params['ticker'] = symbol.upper()
        if from_date:
            params['published_utc.gte'] = from_date
        if to_date:
            params['published_utc.lte'] = to_date
        
        return self._make_request(endpoint, params=params)
    
    def get_market_status(self) -> Dict:
        """Get the current market status (open/closed)."""
        endpoint = "v1/marketstatus/now"
        return self._make_request(endpoint)
    
    def close(self):
        """Close the session."""
        self.session.close()
