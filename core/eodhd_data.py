"""
Real financial data fetching from EODHD API with robust error handling.
Replaces synthetic data generation with actual market data.
"""

import os
import time
import pickle
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
from torch_geometric.data import Data

# Load environment variables
load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")

logger = logging.getLogger(__name__)


class EODHDClient:
    """Enhanced EODHD client with retry logic and rate limit handling."""
    
    BASE_URL = "https://eodhd.com/api"
    CACHE_DIR = Path("cache")
    
    # Rate limiting settings
    MAX_REQUESTS_PER_MINUTE = 50  # EODHD limit
    REQUEST_DELAY = 1.5  # seconds between requests
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("EODHD API key is required. Set EODHD_API_KEY in .env file")
            
        self.api_key = api_key
        self.session = requests.Session()
        self.CACHE_DIR.mkdir(exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start_time = time.time()
        
        logger.info(f"EODHD client initialized with cache at {self.CACHE_DIR}")
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid exceeding API limits."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start_time > 60:
            self.request_count = 0
            self.minute_start_time = current_time
        
        # Check if we need to wait
        if self.request_count >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_count = 0
                self.minute_start_time = time.time()
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _get_cache_path(self, symbol: str, exchange: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        filename = f"{symbol}.{exchange}_{start_date}_{end_date}.pkl"
        return self.CACHE_DIR / filename
    
    def get_eod_data(
        self,
        symbol: str,
        exchange: str = "US",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch EOD data with retry logic and error handling.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code (default: US)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Default date range if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')
        
        # Check cache first
        cache_path = self._get_cache_path(symbol, exchange, start_date, end_date)
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.debug(f"Loaded {symbol} from cache")
                    return df
            except Exception as e:
                logger.warning(f"Cache read failed for {symbol}: {e}")
        
        # Fetch from API with retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                self._wait_for_rate_limit()
                
                params = {
                    'api_token': self.api_key,
                    'fmt': 'json',
                    'from': start_date,
                    'to': end_date
                }
                
                url = f"{self.BASE_URL}/eod/{symbol}.{exchange}"
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.TIMEOUT
                )
                
                if response.status_code == 429:  # Rate limit exceeded
                    wait_time = (attempt + 1) * 60
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Process data
                df = pd.DataFrame(data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()
                
                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        logger.warning(f"Missing column {col} for {symbol}")
                        df[col] = np.nan
                
                # Cache successful result
                if use_cache and not df.empty:
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(df, f)
                        logger.debug(f"Cached data for {symbol}")
                    except Exception as e:
                        logger.warning(f"Cache write failed: {e}")
                
                logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
                return df
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout fetching {symbol} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt == self.MAX_RETRIES - 1:
                    return pd.DataFrame()
                time.sleep(5 * (attempt + 1))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {symbol}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    return pd.DataFrame()
                time.sleep(5 * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        exchange: str = "US",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with progress tracking.
        
        Args:
            symbols: List of stock symbols
            exchange: Exchange code
            start_date: Start date
            end_date: End date
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"Fetching data for {total} symbols...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.get_eod_data(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    results[symbol] = df
                else:
                    logger.warning(f"No data for {symbol}")
                
                if progress_callback:
                    progress_callback(i + 1, total, symbol)
                    
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)}/{total} symbols")
        return results


class RealFinancialDataProcessor:
    """Process real financial data from EODHD for RALEC-GNN."""
    
    # Asset universe matching original research
    ASSET_UNIVERSE = {
        'tech_mega': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'finance': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB'],
        'industrials': ['CAT', 'BA', 'HON', 'UPS', 'GE'],
        'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD'],
        'sector_etfs': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE'],
        'broad_market': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
        'volatility': ['VXX']
    }
    
    def __init__(self, api_key: str = None):
        """Initialize with EODHD API key."""
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("EODHD_API_KEY must be set in environment or passed as parameter")
            
        self.client = EODHDClient(self.api_key)
        self.symbols = self._get_all_symbols()
        logger.info(f"Initialized with {len(self.symbols)} symbols")
    
    def _get_all_symbols(self) -> List[str]:
        """Get flat list of all symbols."""
        all_symbols = []
        for category, symbols in self.ASSET_UNIVERSE.items():
            all_symbols.extend(symbols)
        return list(set(all_symbols))  # Remove duplicates
    
    def fetch_and_prepare_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch and prepare real financial data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to fetch (uses default universe if None)
            
        Returns:
            Combined DataFrame with all symbols
        """
        symbols = symbols or self.symbols
        
        # Fetch data for all symbols
        logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        def progress_callback(current, total, symbol):
            if current % 10 == 0:
                logger.info(f"Progress: {current}/{total} - Last: {symbol}")
        
        data_dict = self.client.fetch_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
        
        # Combine into single DataFrame
        combined_data = []
        
        for symbol, df in data_dict.items():
            if df.empty:
                continue
                
            df = df.copy()
            df['symbol'] = symbol
            df['category'] = self._get_category(symbol)
            
            # Add returns
            df['return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Add technical features
            df['volatility'] = df['return'].rolling(20).std() * np.sqrt(252)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['high_low_ratio'] = df['high'] / df['low'] - 1
            
            combined_data.append(df)
        
        if not combined_data:
            raise ValueError("No data fetched successfully")
        
        # Combine all data
        full_df = pd.concat(combined_data, axis=0)
        full_df = full_df.sort_index()
        
        # Forward fill missing values (holidays, etc)
        full_df = full_df.groupby('symbol').ffill()
        
        logger.info(f"Prepared data shape: {full_df.shape}")
        logger.info(f"Date range: {full_df.index.min()} to {full_df.index.max()}")
        logger.info(f"Symbols included: {full_df['symbol'].nunique()}")
        
        return full_df
    
    def _get_category(self, symbol: str) -> str:
        """Get category for a symbol."""
        for category, symbols in self.ASSET_UNIVERSE.items():
            if symbol in symbols:
                return category
        return 'other'
    
    def create_graph_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 20,
        prediction_horizon: int = 5
    ) -> List[List[Data]]:
        """
        Create graph sequences from real financial data.
        
        Args:
            df: Combined DataFrame with all symbols
            sequence_length: Number of time steps per sequence
            prediction_horizon: Days ahead to predict
            
        Returns:
            List of graph sequences
        """
        # Get unique dates and symbols
        dates = df.index.unique().sort_values()
        symbols = df['symbol'].unique()
        
        # Create sequences
        sequences = []
        
        for i in range(len(dates) - sequence_length - prediction_horizon):
            sequence = []
            
            for j in range(sequence_length):
                date = dates[i + j]
                
                # Get data for this date
                date_data = df.loc[date]
                
                # Create node features
                node_features = []
                valid_symbols = []
                
                for symbol in symbols:
                    if symbol in date_data.index:
                        row = date_data.loc[symbol]
                        
                        features = [
                            row.get('return', 0),
                            row.get('volatility', 0),
                            row.get('volume_ratio', 1),
                            row.get('high_low_ratio', 0),
                            np.sign(row.get('return', 0)),  # Direction
                        ]
                        
                        # Add if valid
                        if not any(np.isnan(features)):
                            node_features.append(features)
                            valid_symbols.append(symbol)
                
                if len(valid_symbols) < 10:  # Skip if too few valid symbols
                    break
                
                # Create edge index (fully connected for now)
                num_nodes = len(valid_symbols)
                edge_index = []
                for src in range(num_nodes):
                    for dst in range(num_nodes):
                        if src != dst:
                            edge_index.append([src, dst])
                
                # Create graph
                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                
                # Add regime label (simplified - based on market volatility)
                if j == sequence_length - 1:
                    # Calculate market regime
                    future_date = dates[i + j + prediction_horizon]
                    if future_date in df.index:
                        future_returns = df.loc[future_date, 'return'].mean()
                        future_vol = df.loc[future_date, 'volatility'].mean()
                        
                        # Simple regime classification
                        if future_vol > 0.25:  # High volatility = crisis
                            y = torch.tensor([2])
                        elif future_returns > 0.001:  # Positive returns = bull
                            y = torch.tensor([0])
                        else:  # Normal/bear
                            y = torch.tensor([1])
                    else:
                        y = torch.tensor([1])  # Default to normal
                else:
                    y = None
                
                graph = Data(x=x, edge_index=edge_index, y=y)
                sequence.append(graph)
            
            if len(sequence) == sequence_length:
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")
        return sequences
    
    def prepare_train_test_data(
        self,
        sequences: List[List[Data]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[List[List[Data]], List[List[Data]], List[List[Data]]]:
        """
        Split sequences into train, validation, and test sets.
        
        Args:
            sequences: List of graph sequences
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            
        Returns:
            Train, validation, and test sequences
        """
        n_sequences = len(sequences)
        train_end = int(n_sequences * train_ratio)
        val_end = int(n_sequences * (train_ratio + val_ratio))
        
        train_sequences = sequences[:train_end]
        val_sequences = sequences[train_end:val_end]
        test_sequences = sequences[val_end:]
        
        logger.info(f"Data split: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")
        
        return train_sequences, val_sequences, test_sequences


def get_real_financial_data(
    lookback_days: int = 2000,
    sequence_length: int = 20,
    prediction_horizon: int = 5
) -> Tuple[List[List[Data]], List[List[Data]], List[List[Data]]]:
    """
    Main function to get real financial data for RALEC-GNN.
    
    Args:
        lookback_days: Days of historical data to fetch
        sequence_length: Length of each sequence
        prediction_horizon: Days ahead to predict
        
    Returns:
        Train, validation, and test data
    """
    # Initialize processor
    processor = RealFinancialDataProcessor()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Fetch and prepare data
    logger.info(f"Fetching real financial data from {start_date} to {end_date}")
    df = processor.fetch_and_prepare_data(start_date, end_date)
    
    # Create graph sequences
    sequences = processor.create_graph_sequences(
        df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Split data
    train_data, val_data, test_data = processor.prepare_train_test_data(sequences)
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test data fetching
    logging.basicConfig(level=logging.INFO)
    
    try:
        train, val, test = get_real_financial_data(lookback_days=500)
        print(f"Successfully loaded real data!")
        print(f"Train: {len(train)} sequences")
        print(f"Val: {len(val)} sequences")
        print(f"Test: {len(test)} sequences")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure EODHD_API_KEY is set in .env file")