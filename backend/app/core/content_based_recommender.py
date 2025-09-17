import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import logging
from datetime import datetime, timedelta
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    Content-based recommendation system for stocks with yfinance integration.
    This class implements content-based filtering to recommend similar stocks
    based on their features, financial metrics, and market data.
    """
    
    def __init__(
        self,
        n_recommendations: int = 5,
        use_yfinance: bool = True,
        use_text_features: bool = True,
        use_numeric_features: bool = True,
        min_similarity: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize the content-based recommender.
        
        Args:
            n_recommendations: Number of recommendations to return
            use_text_features: Whether to use text-based features (e.g., company description)
            use_numeric_features: Whether to use numerical features (e.g., financial ratios)
            min_similarity: Minimum similarity score to include in recommendations
            random_state: Random seed for reproducibility
        """
        self.n_recommendations = n_recommendations
        self.use_text_features = use_text_features
        self.use_numeric_features = use_numeric_features
        self.min_similarity = min_similarity
        self.random_state = random_state
        self.use_yfinance = use_yfinance
        
        # Initialize yfinance Ticker objects cache
        self._ticker_cache = {}
        self._market_data = {}
        self._last_fetch_time = None
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) if use_text_features else None
        self.scaler = MinMaxScaler() if use_numeric_features else None
        
        # Store data
        self.stock_data = None
        self.feature_matrix = None
        self.stock_indices = {}
        
    def _preprocess_text(self, text_series: pd.Series) -> np.ndarray:
        """Preprocess and vectorize text data using TF-IDF."""
        if not self.use_text_features or self.tfidf_vectorizer is None:
            return None
            
        # Fill missing values with empty string
        text_series = text_series.fillna('')
        return self.tfidf_vectorizer.fit_transform(text_series)
    
    def _preprocess_numeric(self, numeric_df: pd.DataFrame) -> np.ndarray:
        """Preprocess and scale numerical features."""
        if not self.use_numeric_features or self.scaler is None:
            return None
            
        # Fill missing values with column means
        numeric_df = numeric_df.fillna(numeric_df.mean())
        return self.scaler.fit_transform(numeric_df)
    
    def _calculate_similarity(
        self, 
        features: np.ndarray, 
        stock_idx: int
    ) -> np.ndarray:
        """
        Calculate similarity scores between a target stock and all other stocks.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            stock_idx: Index of the target stock
            
        Returns:
            Array of similarity scores
        """
        if features is None or features.shape[0] == 0:
            return None
            
        # Calculate cosine similarity
        similarity = cosine_similarity(
            features[stock_idx:stock_idx+1], 
            features
        )[0]
        
        return similarity
    
    def _fetch_yfinance_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch stock data using yfinance."""
        current_time = datetime.now()
        
        # Check if we need to refresh the cache (every hour)
        if (self._last_fetch_time is None or 
            (current_time - self._last_fetch_time) > timedelta(hours=1)):
            self._market_data = {}
            
        # Filter out symbols already in cache
        symbols_to_fetch = [s for s in symbols if s not in self._market_data]
        
        if symbols_to_fetch:
            try:
                # Fetch data in batches to avoid rate limiting
                batch_size = 10
                for i in range(0, len(symbols_to_fetch), batch_size):
                    batch = symbols_to_fetch[i:i + batch_size]
                    tickers = yf.Tickers(' '.join(batch))
                    
                    for symbol in batch:
                        ticker = tickers.tickers[symbol]
                        try:
                            info = ticker.info
                            # Get historical data for the last year
                            hist = ticker.history(period='1y')
                            
                            # Calculate some basic metrics
                            if not hist.empty:
                                returns = hist['Close'].pct_change().dropna()
                                volatility = returns.std() * np.sqrt(252)  # Annualized
                                
                                self._market_data[symbol] = {
                                    'symbol': symbol,
                                    'name': info.get('shortName', symbol),
                                    'sector': info.get('sector', 'Unknown'),
                                    'industry': info.get('industry', 'Unknown'),
                                    'description': info.get('longBusinessSummary', ''),
                                    'market_cap': info.get('marketCap', 0),
                                    'pe_ratio': info.get('trailingPE', 0),
                                    'forward_pe': info.get('forwardPE', 0),
                                    'price': info.get('currentPrice', 0),
                                    'volume': info.get('volume', 0),
                                    'avg_volume': info.get('averageVolume', 0),
                                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                                    'beta': info.get('beta', 0),
                                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                                    'volatility': volatility,
                                    'shares_outstanding': info.get('sharesOutstanding', 0),
                                    'revenue': info.get('totalRevenue', 0),
                                    'profit_margin': info.get('profitMargins', 0),
                                    'return_on_equity': info.get('returnOnEquity', 0),
                                    'debt_to_equity': info.get('debtToEquity', 0),
                                    'price_to_book': info.get('priceToBook', 0),
                                    'free_cash_flow': info.get('freeCashflow', 0),
                                    'revenue_growth': info.get('revenueGrowth', 0),
                                    'earnings_growth': info.get('earningsGrowth', 0)
                                }
                        except Exception as e:
                            logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                            continue
                
                self._last_fetch_time = current_time
                
            except Exception as e:
                logger.error(f"Error in _fetch_yfinance_data: {str(e)}")
                raise
        
        # Return DataFrame with requested symbols
        data = [self._market_data.get(symbol) for symbol in symbols 
               if symbol in self._market_data and self._market_data[symbol] is not None]
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def _enrich_with_yfinance(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance stock data with yfinance information."""
        if not self.use_yfinance or stock_data.empty:
            return stock_data
            
        try:
            # Get unique symbols
            symbols = stock_data['symbol'].unique().tolist()
            
            # Fetch data from yfinance
            yf_data = self._fetch_yfinance_data(symbols)
            
            if yf_data.empty:
                return stock_data
                
            # Merge with existing data
            merged = stock_data.merge(
                yf_data,
                on='symbol',
                how='left',
                suffixes=('', '_yf')
            )
            
            # Fill missing values with yfinance data
            for col in yf_data.columns:
                if col != 'symbol' and col in merged.columns:
                    merged[col] = merged[col].fillna(merged.get(f"{col}_yf", None))
                    if f"{col}_yf" in merged.columns:
                        del merged[f"{col}_yf"]
            
            return merged
            
        except Exception as e:
            logger.error(f"Error in _enrich_with_yfinance: {str(e)}")
            return stock_data
    
    def fit(self, stock_data: Union[pd.DataFrame, List[str], str]) -> 'ContentBasedRecommender':
        """
        Fit the recommender with stock data.
        
        Args:
            stock_data: Can be one of the following:
                       - DataFrame containing stock information with columns:
                         - symbol: Stock symbol (required)
                         - name: Company name
                         - sector: Industry sector
                         - description: Company description (for text features)
                         - Other numeric features (e.g., market_cap, pe_ratio, etc.)
                       - List of stock symbols (will be fetched using yfinance)
                       - Single stock symbol as string (will be fetched using yfinance)
        """
        # Handle different input types
        if isinstance(stock_data, str):
            stock_data = [stock_data]
            
        if isinstance(stock_data, list):
            # Convert list of symbols to DataFrame
            stock_data = pd.DataFrame({'symbol': stock_data})
        
        # Ensure we have the required columns
        if 'symbol' not in stock_data.columns:
            raise ValueError("Input DataFrame must contain 'symbol' column")
        
        # Enrich with yfinance data if enabled
        if self.use_yfinance:
            stock_data = self._enrich_with_yfinance(stock_data)
        
        # Ensure we have at least name and sector for basic recommendations
        if 'name' not in stock_data.columns:
            stock_data['name'] = stock_data['symbol']
        if 'sector' not in stock_data.columns:
            stock_data['sector'] = 'Unknown'
        if stock_data.empty:
            raise ValueError("Stock data cannot be empty")
            
        # Store a copy of the data
        self.stock_data = stock_data.copy()
        
        # Create a mapping from symbol to index
        self.stock_indices = {}
        for idx, row in self.stock_data.iterrows():
            symbol = row['symbol']
            if pd.notna(symbol) and symbol not in self.stock_indices:
                self.stock_indices[symbol] = len(self.stock_indices)
        
        # Process text features if enabled
        text_features = None
        if self.use_text_features and 'description' in stock_data.columns:
            # Ensure we have strings, not NaN
            descriptions = stock_data['description'].fillna('').astype(str)
            if not descriptions.empty:
                text_features = self._preprocess_text(descriptions)
        
        # Process numeric features if enabled
        numeric_features = None
        if self.use_numeric_features:
            # Select numeric columns (excluding text and categorical columns)
            numeric_cols = stock_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Exclude potential ID columns or other non-feature columns
            numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'sector']]
            
            if numeric_cols:
                numeric_features = self._preprocess_numeric(stock_data[numeric_cols])
        
        # Combine features
        if text_features is not None and numeric_features is not None:
            # Combine text and numeric features
            from scipy.sparse import hstack
            self.feature_matrix = hstack([text_features, numeric_features])
        elif text_features is not None:
            self.feature_matrix = text_features
        elif numeric_features is not None:
            self.feature_matrix = numeric_features
        else:
            raise ValueError("No valid features available. Check your data and settings.")
        
        return self
    
    def recommend(
        self, 
        stock_symbol: Union[str, List[str]],
        n_recommendations: Optional[int] = None,
        exclude_self: bool = True,
        min_similarity: Optional[float] = None,
        refresh_data: bool = False
    ) -> List[Dict]:
        """
        Get recommendations for a given stock.
        
        Args:
            stock_symbol: Symbol of the target stock
            n_recommendations: Number of recommendations to return (overrides initialization)
            exclude_self: Whether to exclude the target stock from recommendations
            min_similarity: Minimum similarity score (overrides initialization)
            
        Returns:
            List of recommended stocks with similarity scores
        """
        if self.feature_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Handle list of symbols
        if isinstance(stock_symbol, list):
            return self.get_diverse_recommendations(
                stock_symbol, 
                n_recommendations=n_recommendations or self.n_recommendations,
                min_similarity=min_similarity or self.min_similarity
            )
            
        if stock_symbol not in self.stock_indices:
            # Try to fetch the stock if not found and yfinance is enabled
            if self.use_yfinance and refresh_data:
                logger.info(f"Stock {stock_symbol} not found in dataset, fetching from yfinance...")
                try:
                    # Add the new stock to our dataset
                    new_data = self._enrich_with_yfinance(pd.DataFrame([{'symbol': stock_symbol}]))
                    if not new_data.empty:
                        self.stock_data = pd.concat([self.stock_data, new_data], ignore_index=True)
                        # Re-fit with the new data
                        self.fit(self.stock_data)
                except Exception as e:
                    logger.error(f"Error fetching data for {stock_symbol}: {str(e)}")
            
            if stock_symbol not in self.stock_indices:
                raise ValueError(f"Stock symbol '{stock_symbol}' not found in the dataset and could not be fetched")
            
        # Get parameters
        n = n_recommendations or self.n_recommendations
        min_sim = min_similarity if min_similarity is not None else self.min_similarity
        
        # Get stock index
        stock_idx = self.stock_indices[stock_symbol]
        
        # Calculate similarity scores
        similarity_scores = self._calculate_similarity(self.feature_matrix, stock_idx)
        
        if similarity_scores is None:
            return []
        
        # Get indices of top similar stocks
        top_indices = np.argsort(similarity_scores)[::-1]
        
        # Prepare recommendations
        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= n + (1 if exclude_self else 0):
                break
                
            current_symbol = self.stock_data.iloc[idx]['symbol']
            score = similarity_scores[idx]
            
            # Skip if below minimum similarity
            if score < min_sim:
                continue
                
            # Skip self if needed
            if exclude_self and current_symbol == stock_symbol:
                continue
                
            # Add to recommendations
            stock_info = self.stock_data.iloc[idx].to_dict()
            recommendations.append({
                'symbol': stock_info['symbol'],
                'name': stock_info.get('name', ''),
                'sector': stock_info.get('sector', ''),
                'similarity_score': float(score),
                'price': stock_info.get('price', None),
                'change_percent': stock_info.get('change_percent', None)
            })
        
        return recommendations
    
    def get_similar_stocks(
        self, 
        stock_symbols: Union[str, List[str]],
        n_recommendations: int = 5,
        min_similarity: float = 0.3,
        refresh_data: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Get similar stocks for multiple target stocks.
        
        Args:
            stock_symbols: Single stock symbol or list of stock symbols
            n_recommendations: Number of recommendations per stock
            min_similarity: Minimum similarity score
            refresh_data: Whether to refresh data from yfinance if stock not found
            
        Returns:
            Dictionary mapping each stock symbol to its recommendations
        """
        if isinstance(stock_symbols, str):
            stock_symbols = [stock_symbols]
            
        results = {}
        
        for symbol in stock_symbols:
            try:
                if symbol in self.stock_indices or (self.use_yfinance and refresh_data):
                    results[symbol] = self.recommend(
                        symbol,
                        n_recommendations=n_recommendations,
                        min_similarity=min_similarity,
                        refresh_data=refresh_data
                    )
            except Exception as e:
                logger.warning(f"Could not get recommendations for {symbol}: {str(e)}")
                results[symbol] = []
        
        return results
    
    def get_diverse_recommendations(
        self, 
        stock_symbols: Union[str, List[str]],
        n_recommendations: int = 5,
        min_similarity: float = 0.3,
        strategy: str = 'similarity'  # 'similarity', 'diverse', 'sector_diverse'
    ) -> List[Dict]:
        """
        Get diverse recommendations across multiple stocks.
        
        Args:
            stock_symbols: List of stock symbols
            n_recommendations: Total number of recommendations to return
            min_similarity: Minimum similarity score
            
        Returns:
            List of diverse recommendations
        """
        if isinstance(stock_symbols, str):
            stock_symbols = [stock_symbols]
            
        # Get recommendations for each stock
        all_recommendations = []
        for symbol in stock_symbols:
            if symbol in self.stock_indices:
                try:
                    recs = self.recommend(
                        symbol,
                        n_recommendations=n_recommendations * 2,  # Get extra for diversity
                        min_similarity=min_similarity
                    )
                    all_recommendations.extend(recs)
                except Exception as e:
                    logger.warning(f"Could not get recommendations for {symbol}: {str(e)}")
        
        if not all_recommendations:
            return []
        
        # Sort by similarity score and remove duplicates
        unique_recommendations = {}
        for rec in sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True):
            if rec['symbol'] not in unique_recommendations and rec['symbol'] not in stock_symbols:
                unique_recommendations[rec['symbol']] = rec
            if len(unique_recommendations) >= n_recommendations * 2:  # Get extra for diversity
                break
        
        if not unique_recommendations:
            return []
            
        recommendations = list(unique_recommendations.values())
        
        if strategy == 'similarity':
            # Sort by highest similarity
            recommendations.sort(key=lambda x: -x['similarity_score'])
        elif strategy == 'diverse':
            # Sort by diversity (lower similarity to input stocks)
            # This is a simple implementation - could be enhanced with more sophisticated diversity metrics
            recommendations.sort(key=lambda x: x['similarity_score'])
        elif strategy == 'sector_diverse':
            # Try to get recommendations from different sectors
            sector_counts = {}
            for symbol in stock_symbols:
                if symbol in self.stock_indices:
                    sector = self.stock_data.iloc[self.stock_indices[symbol]].get('sector', 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Sort by sector diversity and then by similarity
            recommendations.sort(
                key=lambda x: (
                    -sector_counts.get(x.get('sector', 'Unknown'), 0),  # Less common sectors first
                    -x['similarity_score']  # Then by similarity
                )
            )
        
        return recommendations[:n_recommendations]
    
    def get_sector_recommendations(
        self, 
        sector: str,
        n_recommendations: int = 5,
        exclude_symbols: List[str] = None,
        sort_by: str = 'market_cap'  # 'market_cap', 'pe_ratio', 'dividend_yield', 'volatility'
    ) -> List[Dict]:
        """
        Get recommendations within the same sector.
        
        Args:
            sector: Target sector
            n_recommendations: Number of recommendations to return
            exclude_symbols: List of symbols to exclude from results
            
        Returns:
            List of recommended stocks in the same sector
        """
        if self.stock_data is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
            
        # Filter stocks by sector
        sector_stocks = self.stock_data[
            self.stock_data['sector'].str.lower() == sector.lower()
        ]
        
        if exclude_symbols:
            sector_stocks = sector_stocks[~sector_stocks['symbol'].isin(exclude_symbols)]
        
        if sector_stocks.empty:
            return []
        
        # Determine sort column and order
        sort_col = sort_by if sort_by in sector_stocks.columns else 'market_cap'
        ascending = sort_by in ['pe_ratio', 'volatility']  # Lower is better for these
        
        # Sort and get top N
        top_stocks = sector_stocks.sort_values(
            by=sort_col, 
            ascending=ascending,
            na_position='last'  # Put NA values at the end
        ).head(n_recommendations)
        
        return [
            {
                'symbol': row['symbol'],
                'name': row.get('name', ''),
                'sector': row.get('sector', ''),
                'price': row.get('price', None),
                'change_percent': row.get('change_percent', None)
            }
            for _, row in top_stocks.iterrows()
        ]
