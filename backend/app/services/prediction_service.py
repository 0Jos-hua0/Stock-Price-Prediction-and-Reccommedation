import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path: str = None):
        """Initialize the prediction service with a trained model."""
        self.model = None
        self.scaler = None
        self.sequence_length = 30  # Should match training sequence length
        self.prediction_horizon = 1
        self.model_path = model_path or os.path.join('saved_models', 'best_model.h5')
        self.scaler_path = os.path.join('saved_models', 'feature_scaler.pkl')
        self.load_model()

    def load_model(self):
        """Load the trained model and scaler."""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info(f"Successfully loaded model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Successfully loaded scaler from {self.scaler_path}")
            else:
                logger.warning(f"Scaler not found at {self.scaler_path}")

        except Exception as e:
            logger.error(f"Error loading model/scaler: {str(e)}")
            raise

    def prepare_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare input sequence for prediction."""
        # Select features and scale
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.sequence_length + 1):
            sequences.append(features[i:(i + self.sequence_length)])
        
        return np.array(sequences)

    def predict(self, historical_data: pd.DataFrame, days: int = 1) -> Dict:
        """
        Predict stock prices for the next 'days'.
        
        Args:
            historical_data: DataFrame with historical stock data
            days: Number of days to predict
            
        Returns:
            Dict containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        try:
            # Prepare the most recent sequence for prediction
            sequence = self.prepare_sequence(historical_data)
            if len(sequence) == 0:
                raise ValueError("Insufficient data to form a sequence")
            
            # Use the most recent sequence
            current_sequence = sequence[-1:]
            
            # Generate predictions
            predictions = []
            for _ in range(days):
                # Predict next step
                next_step = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_step[0][0])
                
                # Update sequence with the prediction
                if len(current_sequence[0]) > 1:
                    # Remove first step and append prediction
                    updated_sequence = np.append(current_sequence[0][1:], [next_step[0]], axis=0)
                    current_sequence = np.array([updated_sequence])
            
            # If we have a scaler, inverse transform the predictions
            if self.scaler and hasattr(self.scaler, 'inverse_transform'):
                # Create dummy array for inverse transform
                dummy = np.zeros((len(predictions), 5))  # Assuming 5 features
                dummy[:, 3] = predictions  # Assuming 'close' is at index 3
                predictions = self.scaler.inverse_transform(dummy)[:, 3]
            
            # Generate prediction dates
            last_date = pd.to_datetime(historical_data.index[-1])
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            return {
                'predictions': predictions.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'last_sequence_date': last_date.strftime('%Y-%m-%d'),
                'model_loaded': True
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {
                'model_loaded': False,
                'model_path': self.model_path,
                'message': 'Model not loaded'
            }
        
        return {
            'model_loaded': True,
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'sequence_length': self.sequence_length,
            'scaler_loaded': self.scaler is not None,
            'last_updated': datetime.fromtimestamp(os.path.getmtime(self.model_path)).isoformat()
        }
