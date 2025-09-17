import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.lstm_cnn_model import LSTMCnnModel
from app.core.data_enhanced_processor import EnhancedStockDataProcessor

# Constants
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 5  # Predict 5 days ahead
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
MODEL_DIR = 'saved_models'
DATA_DIR = os.path.join('..', 'data', 'processed')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed_data(ticker: str, data_type: str = 'train') -> np.ndarray:
    """Load processed data for a specific ticker and data type (train/val/test)."""
    file_path = os.path.join(DATA_DIR, data_type, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data not found for {ticker} ({data_type})")
    
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    # Reshape for LSTM input (samples, timesteps, features)
    # We need to know the number of features from the data
    n_samples = len(X)
    n_features = X.shape[1] // SEQUENCE_LENGTH
    
    if n_samples == 0:
        return np.array([]), np.array([])
    
    X_reshaped = X.reshape(n_samples, SEQUENCE_LENGTH, n_features)
    
    return X_reshaped, y

def load_all_data() -> tuple:
    """Load all processed data for all tickers."""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    for ticker in TICKERS:
        try:
            # Load train data
            X_train, y_train = load_processed_data(ticker, 'train')
            if len(X_train) > 0:
                X_train_list.append(X_train)
                y_train_list.append(y_train)
            
            # Load validation data
            X_val, y_val = load_processed_data(ticker, 'val')
            if len(X_val) > 0:
                X_val_list.append(X_val)
                y_val_list.append(y_val)
            
            # Load test data
            X_test, y_test = load_processed_data(ticker, 'test')
            if len(X_test) > 0:
                X_test_list.append(X_test)
                y_test_list.append(y_test)
                
            logger.info(f"Loaded data for {ticker}: train={len(X_train) if len(X_train) > 0 else 0}, "
                       f"val={len(X_val) if len(X_val) > 0 else 0}, test={len(X_test) if len(X_test) > 0 else 0}")
            
        except Exception as e:
            logger.warning(f"Error loading data for {ticker}: {str(e)}")
    
    # Concatenate data from all tickers
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    
    X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.array([])
    y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.array([])
    
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])
    
    # Shuffle the training data
    if len(X_train) > 0:
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    logger.info(f"Total samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_lstm_cnn_model(X_train, X_val, y_train, y_val, input_shape: tuple) -> tuple:
    """Train the LSTM-CNN model."""
    logger.info("Initializing LSTM-CNN model...")
    
    # Initialize model
    model = LSTMCnnModel(
        input_shape=input_shape,
        lstm_units=[64, 32],  # Using list format as expected by the class
        cnn_filters=[64, 128],  # Using list format as expected by the class
        kernel_sizes=[3, 5],    # Using list format as expected by the class
        dense_units=[128, 64],  # Using list format as expected by the class
        dropout_rate=0.2,
        l2_reg=0.001
    )
    
    # Compile the model
    model.model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=PATIENCE//2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
    ]
    
    logger.info("Starting model training...")
    
    # Train the model
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Model training completed.")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    if len(X_test) == 0 or len(y_test) == 0:
        logger.warning("No test data available for evaluation.")
        return
    
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test MAE: {mae:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test R²: {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Stock Prices (First 100 Samples)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, 'predictions_plot.png')
    plt.savefig(plot_path)
    logger.info(f"Prediction plot saved to {plot_path}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def evaluate_model(model, X_test, y_test, data_processor=None):
    """Evaluate the trained model on test data."""
    if len(X_test) == 0 or len(y_test) == 0:
        logger.warning("No test data available for evaluation.")
        return
    
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.model.predict(X_test)
    
    # If data_processor is provided, inverse transform the predictions
    if data_processor and hasattr(data_processor, 'inverse_transform_target'):
        y_test = data_processor.inverse_transform_target(y_test)
        y_pred = data_processor.inverse_transform_target(y_pred)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test MAE: {mae:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test R²: {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Stock Prices (First 100 Samples)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, 'predictions_plot.png')
    plt.savefig(plot_path)
    logger.info(f"Prediction plot saved to {plot_path}")
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
        # Load all processed data
        logger.info("Loading processed data...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_all_data()
        
        if len(X_train) == 0:
            raise ValueError("No training data available. Please check the data processing step.")
        
        # Get input shape from training data
        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(f"Input shape: {input_shape}")
        
        # Train model
        logger.info("Starting model training...")
        model, history = train_lstm_cnn_model(X_train, X_val, y_train, y_val, input_shape)
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, 'stock_prediction_model.h5')
        model.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(MODEL_DIR, 'training_history.pkl')
        joblib.dump(history.history, history_path)
        logger.info(f"Training history saved to {history_path}")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save metrics
        if metrics:
            metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Model metrics saved to {metrics_path}")
        
        # Demonstrate content-based recommendations
        demonstrate_content_based_recommendations()
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
