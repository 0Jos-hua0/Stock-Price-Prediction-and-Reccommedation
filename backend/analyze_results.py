import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary classes
from app.models.lstm_cnn_model import LSTMCnnModel
from app.core.data_enhanced_processor import EnhancedStockDataProcessor
from process_data import DataProcessor

# Initialize the data processor
data_processor = DataProcessor()

# Load and prepare the data
print("Loading and preparing data...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
X_train_all, X_val_all, X_test_all = [], [], []
y_train_all, y_val_all, y_test_all = [], [], []

for ticker in tickers:
    try:
        # Load raw data
        df = data_processor.load_raw_data(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping...")
            continue
            
        # Initialize the enhanced processor
        processor = EnhancedStockDataProcessor(
            sequence_length=30,
            prediction_horizon=1,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            scale_features=True,
            scale_target=True
        )
        
        # Process the data
        X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_training_data(df)
        
        # Collect data from all tickers
        X_train_all.append(X_train)
        X_val_all.append(X_val)
        X_test_all.append(X_test)
        y_train_all.append(y_train)
        y_val_all.append(y_val)
        y_test_all.append(y_test)
        
        print(f"Processed {ticker}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        continue

# Concatenate data from all tickers
X_train = np.concatenate(X_train_all, axis=0) if X_train_all else np.array([])
X_val = np.concatenate(X_val_all, axis=0) if X_val_all else np.array([])
X_test = np.concatenate(X_test_all, axis=0) if X_test_all else np.array([])
y_train = np.concatenate(y_train_all, axis=0) if y_train_all else np.array([])
y_val = np.concatenate(y_val_all, axis=0) if y_val_all else np.array([])
y_test = np.concatenate(y_test_all, axis=0) if y_test_all else np.array([])

print(f"Total samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Initialize and load the model
input_shape = (30, X_train.shape[2]) if len(X_train) > 0 else (30, 34)
model = LSTMCnnModel(
    input_shape=input_shape,
    lstm_units=[64, 32],
    cnn_filters=[64, 128],
    kernel_sizes=[3, 5],
    dense_units=[128, 64],
    dropout_rate=0.2,
    l2_reg=0.001
)

# Load the model weights
model_path = os.path.join('saved_models', 'best_model.h5')
model.model.load_weights(model_path)

# Load the training history
try:
    history = joblib.load(os.path.join('saved_models', 'training_history.pkl'))
    print("Loaded training history.")
except FileNotFoundError:
    print("Warning: Training history not found. Some plots will be skipped.")
    history = {}

# Make predictions on test data
if len(X_test) > 0:
    y_pred = model.model.predict(X_test)
else:
    print("No test data available for predictions.")
    y_pred = np.array([])

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test R²: {r2:.6f}")

# Plot training & validation loss values
plt.figure(figsize=(12, 10))

# Plot loss
plt.subplot(2, 2, 1)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Plot MAE
plt.subplot(2, 2, 2)
plt.plot(history['mae'], label='Train')
plt.plot(history['val_mae'], label='Validation')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

# Plot predictions vs actual (first 100 samples)
plt.subplot(2, 2, 3)
plt.plot(y_test[:100], label='Actual', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted (First 100 Samples)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# Scatter plot of predictions vs actuals
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Predictions vs Actuals')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig(os.path.join('saved_models', 'detailed_analysis.png'))
print(f"Detailed analysis plot saved to saved_models/detailed_analysis.png")

# Save metrics to a JSON file
metrics = {
    'mse': float(mse),
    'mae': float(mae),
    'rmse': float(rmse),
    'r2': float(r2)
}

with open(os.path.join('saved_models', 'final_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("Analysis complete. Check the saved_models directory for results.")
