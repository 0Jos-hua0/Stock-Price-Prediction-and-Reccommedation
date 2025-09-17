import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    Flatten, Concatenate, BatchNormalization, LeakyReLU,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from typing import Tuple, Optional
import numpy as np

class LSTMCnnModel:
    """
    A hybrid LSTM-CNN model for time series forecasting.
    This model combines the sequence learning capabilities of LSTM 
    with the feature extraction power of CNN.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_blocks: int = 2,
        lstm_units: list = [64, 32],
        cnn_filters: list = [64, 128],
        kernel_sizes: list = [3, 5],
        dense_units: list = [128, 64],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        use_attention: bool = True,
        use_residual: bool = True,
        num_heads: int = 4,
        key_dim: int = 32
    ):
        """
        Initialize the LSTM-CNN model.
        
        Args:
            input_shape: Tuple of (timesteps, features)
            num_blocks: Number of LSTM-CNN blocks
            lstm_units: List of units for each LSTM layer
            cnn_filters: List of filters for each CNN layer
            kernel_sizes: List of kernel sizes for CNN layers
            dense_units: List of units for dense layers after LSTM-CNN blocks
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            use_attention: Whether to use self-attention mechanism
            use_residual: Whether to use residual connections
            num_heads: Number of attention heads
            key_dim: Dimension of the key space in attention
        """
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.model = self._build_model()
    
    def _attention_block(self, inputs):
        """Self-attention mechanism for time series data."""
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )(inputs, inputs)
        
        # Skip connection and layer normalization
        if self.use_residual:
            attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        else:
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
            
        return attention_output
    
    def _lstm_cnn_block(self, inputs, lstm_units, cnn_filters, kernel_size, block_num):
        """A single LSTM-CNN block with optional residual connection."""
        # LSTM layer
        lstm_out = LSTM(
            units=lstm_units,
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg),
            name=f'block{block_num}_lstm'
        )(inputs)
        
        # CNN layer for feature extraction
        conv_out = Conv1D(
            filters=cnn_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(self.l2_reg),
            name=f'block{block_num}_conv1d_{kernel_size}'
        )(lstm_out)
        
        # Batch normalization and activation
        conv_out = BatchNormalization()(conv_out)
        conv_out = LeakyReLU(alpha=0.1)(conv_out)
        
        # Max pooling
        pooled = MaxPooling1D(pool_size=2, padding='same')(conv_out)
        
        # Dropout
        pooled = Dropout(self.dropout_rate)(pooled)
        
        # Residual connection if input and output shapes match
        if self.use_residual and inputs.shape[-1] == pooled.shape[-1]:
            return tf.keras.layers.add([inputs, pooled])
        
        return pooled
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM-CNN model."""
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')
        x = inputs
        
        # Multiple LSTM-CNN blocks
        for i in range(self.num_blocks):
            x = self._lstm_cnn_block(
                x,
                lstm_units=self.lstm_units[i % len(self.lstm_units)],
                cnn_filters=self.cnn_filters[i % len(self.cnn_filters)],
                kernel_size=self.kernel_sizes[i % len(self.kernel_sizes)],
                block_num=i+1
            )
        
        # Optional self-attention layer
        if self.use_attention:
            x = self._attention_block(x)
        
        # Global average pooling to handle variable sequence lengths
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers for final prediction
        for i, units in enumerate(self.dense_units):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer (regression)
        outputs = Dense(1, activation='linear', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_cnn_model')
        
        return model
    
    def compile(self, learning_rate: float = 0.001) -> None:
        """Compile the model with Adam optimizer and MSE loss."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    def summary(self) -> None:
        """Print model summary."""
        return self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training data of shape (samples, timesteps, features)
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            verbose: Verbosity mode (0, 1, or 2)
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = []
            
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMCnnModel':
        """Load a saved model from a file."""
        model = cls(input_shape=(None, 1))  # Dummy input shape
        model.model = tf.keras.models.load_model(filepath)
        return model
