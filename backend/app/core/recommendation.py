from typing import Dict, Any, Tuple
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StockRecommendation:
    def __init__(self):
        self.buy_threshold = 0.02  # 2% expected return for buy
        self.sell_threshold = -0.01  # -1% expected return for sell
        self.high_confidence_threshold = 0.7
    
    def generate_recommendation(
        self, 
        symbol: str,
        prediction: float,
        confidence: float,
        current_price: float,
        historical_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading recommendation based on model prediction and confidence.
        
        Args:
            symbol: Stock symbol
            prediction: Predicted price movement (e.g., percentage change)
            confidence: Model's confidence in the prediction (0-1)
            current_price: Current price of the stock
            historical_data: Optional historical data for additional context
            
        Returns:
            Dict containing recommendation details
        """
        try:
            # Determine the action based on prediction and thresholds
            if prediction > self.buy_threshold:
                action = "BUY"
                reason = "Predicted significant positive price movement"
            elif prediction < self.sell_threshold:
                action = "SELL"
                reason = "Predicted significant negative price movement"
            else:
                action = "HOLD"
                reason = "Expected price movement within neutral range"
            
            # Adjust confidence based on market conditions if historical data is provided
            if historical_data is not None:
                confidence = self._adjust_confidence(confidence, historical_data)
            
            # Determine confidence level
            confidence_level = "HIGH" if confidence >= self.high_confidence_threshold else "MEDIUM"
            
            # Calculate target price if it's a BUY or SELL recommendation
            target_price = None
            if action != "HOLD":
                target_price = current_price * (1 + prediction)
            
            return {
                "symbol": symbol.upper(),
                "action": action,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "prediction": prediction,
                "current_price": current_price,
                "target_price": target_price,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "model_confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            # Return a safe HOLD recommendation in case of errors
            return {
                "symbol": symbol.upper(),
                "action": "HOLD",
                "confidence": 0.5,
                "confidence_level": "LOW",
                "prediction": 0.0,
                "current_price": current_price,
                "target_price": None,
                "reason": f"Error in generating recommendation: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "model_confidence": 0.0
            }
    
    def _adjust_confidence(self, confidence: float, historical_data: Dict[str, Any]) -> float:
        """
        Adjust confidence based on historical data and market conditions.
        
        Args:
            confidence: Original model confidence
            historical_data: Historical price and volume data
            
        Returns:
            Adjusted confidence value
        """
        try:
            # Example: Reduce confidence during high volatility
            # Calculate 20-day volatility as standard deviation of returns
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # If volatility is high, reduce confidence
            if volatility > 0.02:  # 2% daily volatility is considered high
                confidence *= 0.9  # Reduce confidence by 10%
            
            # Ensure confidence stays within [0, 1]
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error adjusting confidence: {str(e)}")
            return confidence
    
    def batch_recommendations(
        self, 
        symbol: str, 
        predictions: np.ndarray, 
        confidences: np.ndarray,
        current_price: float,
        historical_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for multiple predictions.
        
        Args:
            symbol: Stock symbol
            predictions: Array of predicted price movements
            confidences: Array of corresponding confidence values
            current_price: Current price of the stock
            historical_data: Optional historical data for context
            
        Returns:
            Dict containing aggregated recommendation
        """
        try:
            # Calculate weighted prediction based on confidence
            weighted_pred = np.average(predictions, weights=confidences)
            avg_confidence = np.mean(confidences)
            
            # Generate recommendation based on weighted prediction
            return self.generate_recommendation(
                symbol=symbol,
                prediction=weighted_pred,
                confidence=avg_confidence,
                current_price=current_price,
                historical_data=historical_data
            )
            
        except Exception as e:
            logger.error(f"Error in batch recommendations: {str(e)}")
            return self.generate_recommendation(
                symbol=symbol,
                prediction=0.0,
                confidence=0.0,
                current_price=current_price,
                historical_data=historical_data
            )
