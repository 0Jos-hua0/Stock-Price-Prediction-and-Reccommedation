"""Configuration settings for the stock market analysis project."""
from pathlib import Path

# Project directories
import os
# Use the project root directory
BASE_DIR = Path(__file__).parent.parent  # Points to backend directory
DATA_DIR = BASE_DIR / '..' / 'data'  # Go up one level to project root, then into data
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Data settings
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'PG'
]

# Training parameters
TRAIN_CONFIG = {
    'sequence_length': 30,          # Number of time steps in each sequence
    'prediction_horizon': 5,        # Number of days to predict ahead
    'test_size': 0.2,               # Percentage of data for testing
    'val_size': 0.1,                # Percentage of training data for validation
    'batch_size': 32,               # Batch size for training
    'epochs': 100,                  # Maximum number of epochs
    'patience': 15,                 # Early stopping patience
    'learning_rate': 0.001,         # Learning rate
    'dropout_rate': 0.3,            # Dropout rate
    'l2_reg': 0.001,               # L2 regularization factor
}

# Model architecture
MODEL_CONFIG = {
    'num_blocks': 2,                # Number of LSTM-CNN blocks
    'lstm_units': [64, 32],         # Number of units in each LSTM layer
    'cnn_filters': [64, 128],       # Number of filters in each CNN layer
    'kernel_sizes': [3, 5],         # Kernel sizes for CNN layers
    'dense_units': [128, 64],       # Number of units in dense layers
    'use_attention': True,          # Whether to use attention mechanism
    'use_residual': True,           # Whether to use residual connections
}

# Data processing
DATA_PROCESSING = {
    'scale_features': True,         # Whether to scale features
    'scale_target': True,           # Whether to scale target variable
    'feature_scaler': 'standard',   # 'minmax', 'standard', or 'robust'
    'target_scaler': 'standard',    # 'minmax', 'standard', or 'robust'
    'add_technical_indicators': True,  # Whether to add technical indicators
    'add_time_features': True,      # Whether to add time-based features
    'fill_method': 'ffill',         # Method to handle missing values
    'drop_na': True,               # Whether to drop rows with NA values
}

# API Keys (store sensitive keys in environment variables)
API_KEYS = {
    'alpha_vantage': None,  # Set in environment variables
    'finnhub': None,        # Set in environment variables
    'yfinance': None        # Not needed for yfinance
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'stock_analysis.log',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    },
}
