import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir: str = 'saved_models'):
        self.model_dir = model_dir
        self.model: Optional[tf.keras.Model] = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 30
        self.batch_size = 32
        self.epochs = 100
        self.patience = 10
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build a CNN-LSTM model for stock price prediction."""
        model = Sequential([
            # First LSTM layer
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dense(1, activation='linear')  # Linear activation for regression
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> tf.keras.callbacks.History:
        """Train the model with early stopping and model checkpointing."""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        model_path = os.path.join(self.model_dir, 'best_model.h5')
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X)
    
    def save_model(self, filename: str = 'stock_prediction_model.h5'):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save.")
            
        model_path = os.path.join(self.model_dir, filename)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str = 'stock_prediction_model.h5') -> tf.keras.Model:
        """Load a trained model from a file."""
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
            
        metrics = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, metrics))
