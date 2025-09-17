import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Create saved_models directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

try:
    # Load the training history
    history = joblib.load(os.path.join('saved_models', 'training_history.pkl'))
    print("Loaded training history.")
    
    # Plot training & validation loss values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    if 'mae' in history:
        plt.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('saved_models', 'training_metrics.png'))
    print(f"Training metrics plot saved to saved_models/training_metrics.png")
    
    # If we have test predictions, plot them
    if os.path.exists(os.path.join('saved_models', 'test_predictions.npy')):
        test_data = np.load(os.path.join('saved_models', 'test_predictions.npy'), allow_pickle=True).item()
        y_test = test_data.get('y_test')
        y_pred = test_data.get('y_pred')
        
        if y_test is not None and y_pred is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test[:200], label='Actual', alpha=0.7)
            plt.plot(y_pred[:200], label='Predicted', alpha=0.7)
            plt.title('Actual vs Predicted (First 200 Samples)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(os.path.join('saved_models', 'predictions_plot.png'))
            print(f"Predictions plot saved to saved_models/predictions_plot.png")
    
    plt.show()
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure you have run the training script first and that the model files exist.")
