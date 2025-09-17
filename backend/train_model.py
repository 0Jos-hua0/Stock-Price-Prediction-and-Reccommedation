import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.lstm_cnn_model import LSTMCnnModel
from app.core.data_enhanced_processor import EnhancedStockDataProcessor
from app.core.content_based_recommender import ContentBasedRecommender

# Constants
TICKER = 'AAPL'  # Example stock ticker
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 5  # Predict 5 days ahead
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    
    # Reset index to make date a column
    df = df.reset_index()
    
    # Ensure we have the required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Rename columns to lowercase
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Add additional features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    
    return df

def prepare_training_data(data_processor, df: pd.DataFrame) -> tuple:
    """Prepare training and validation data using the EnhancedStockDataProcessor."""
    logger.info("Preparing training data...")
    
    # Preprocess and create sequences
    X, y = data_processor.preprocess_data(df, is_training=True)
    
    # Create sequences
    X_seq, y_seq = data_processor._create_sequences(X, y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, 
        test_size=TEST_SIZE,
        random_state=42,
        shuffle=False  # Important for time series data
    )
    
    # Further split training data into train and validation sets
    val_size = int(len(X_train) * VAL_SIZE)
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_lstm_cnn_model(X_train, X_val, y_train, y_val, input_shape: tuple) -> LSTMCnnModel:
    """Train the LSTM-CNN model."""
    logger.info("Initializing LSTM-CNN model...")
    
    # Initialize model
    model = LSTMCnnModel(
        input_shape=input_shape,
        num_blocks=2,
        lstm_units=[64, 32],
        cnn_filters=[64, 128],
        kernel_sizes=[3, 5],
        dense_units=[128, 64],
        dropout_rate=0.3,
        l2_reg=0.001,
        use_attention=True,
        use_residual=True
    )
    
    # Compile model
    model.compile(learning_rate=0.001)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=PATIENCE // 2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("Training model...")
    history = model.train(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, data_processor):
    """Evaluate the trained model on test data."""
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = data_processor.inverse_transform_target(y_test)
    y_pred_inv = data_processor.inverse_transform_target(y_pred)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual', alpha=0.7)
    plt.plot(y_pred_inv, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, 'predictions.png')
    plt.savefig(plot_path)
    logger.info(f"Prediction plot saved to {plot_path}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def prepare_recommendation_data() -> pd.DataFrame:
    """Prepare sample data for content-based recommendations."""
    # In a real application, this would come from a database or API
    # For demonstration, we'll create a sample dataset
    
    # List of tech stocks for our example
    tech_stocks = [
        ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 'Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'),
        ('MSFT', 'Microsoft Corporation', 'Technology', 'Software - Infrastructure', 'Microsoft develops, licenses, and supports software, services, devices, and solutions worldwide.'),
        ('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Content & Information', 'Alphabet provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'),
        ('AMZN', 'Amazon.com, Inc.', 'Consumer Cyclical', 'Internet Retail', 'Amazon engages in the retail sale of consumer products and subscriptions in North America and internationally.'),
        ('META', 'Meta Platforms, Inc.', 'Communication Services', 'Internet Content & Information', 'Meta Platforms develops products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality headsets, and wearables.'),
        ('TSLA', 'Tesla, Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'Tesla designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems.'),
        ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 'NVIDIA provides graphics, and compute and networking solutions in the United States, China, and internationally.'),
        ('PYPL', 'PayPal Holdings, Inc.', 'Financial Services', 'Credit Services', 'PayPal operates a technology platform that enables digital payments on behalf of merchants and consumers worldwide.'),
        ('ADBE', 'Adobe Inc.', 'Technology', 'Software - Infrastructure', 'Adobe operates as a diversified software company worldwide.'),
        ('INTC', 'Intel Corporation', 'Technology', 'Semiconductors', 'Intel designs, manufactures, and sells computer components and related products.')
    ]
    
    # Create DataFrame
    df = pd.DataFrame(tech_stocks, columns=['symbol', 'name', 'sector', 'industry', 'description'])
    
    # Add some random financial metrics for demonstration
    np.random.seed(42)
    df['market_cap'] = np.random.uniform(50, 1000, len(df))  # In billions
    df['pe_ratio'] = np.random.uniform(10, 50, len(df))
    df['dividend_yield'] = np.random.uniform(0, 3, len(df))
    df['price'] = np.random.uniform(100, 500, len(df))
    df['change_percent'] = np.random.uniform(-5, 5, len(df))
    
    return df

def demonstrate_content_based_recommendations():
    """Demonstrate content-based recommendations."""
    logger.info("Preparing content-based recommendation data...")
    
    # Prepare sample data
    stock_data = prepare_recommendation_data()
    
    # Initialize recommender
    recommender = ContentBasedRecommender(
        n_recommendations=3,
        use_text_features=True,
        use_numeric_features=True,
        min_similarity=0.1
    )
    
    # Fit the recommender
    recommender.fit(stock_data)
    
    # Get recommendations for a few stocks
    target_stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    for stock in target_stocks:
        print(f"\nRecommendations for {stock}:")
        recommendations = recommender.recommend(stock)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} ({rec['symbol']}) - Similarity: {rec['similarity_score']:.2f}")
    
    # Get diverse recommendations across multiple stocks
    print("\nDiverse recommendations:")
    diverse_recs = recommender.get_diverse_recommendations(target_stocks, n_recommendations=3)
    
    for i, rec in enumerate(diverse_recs, 1):
        print(f"{i}. {rec['name']} ({rec['symbol']}) - Similarity: {rec['similarity_score']:.2f}")

def main():
    """Main function to run the training and evaluation pipeline."""
    try:
        # Fetch and prepare data
        df = fetch_stock_data(TICKER, START_DATE, END_DATE)
        
        # Initialize data processor
        data_processor = EnhancedStockDataProcessor(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            scale_features=True,
            scale_target=True,
            feature_scaler_type='standard',
            target_scaler_type='standard',
            add_technical_indicators=True,
            add_time_features=True,
            target_column='close',
            drop_na=True,
            fill_method='ffill'
        )
        
        # Prepare training data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(data_processor, df)
        
        # Get input shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Train model
        model, history = train_lstm_cnn_model(X_train, X_val, y_train, y_val, input_shape)
        
        # Save the final model
        model_path = os.path.join(MODEL_DIR, 'final_model.h5')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save the data processor
        processor_path = os.path.join(MODEL_DIR, 'data_processor.pkl')
        joblib.dump(data_processor, processor_path)
        logger.info(f"Data processor saved to {processor_path}")
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, data_processor)
        
        # Demonstrate content-based recommendations
        demonstrate_content_based_recommendations()
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
