from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

from app.services.prediction_service import PredictionService
from app.services.data_fetcher import DataFetcher
from app.core.data_enhanced_processor import EnhancedStockDataProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
prediction_service = PredictionService()
data_fetcher = DataFetcher()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify the API is running."""
    model_info = prediction_service.get_model_info()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_info['model_loaded'],
        "model_last_updated": model_info.get('last_updated')
    }

@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded prediction model."""
    return prediction_service.get_model_info()

@router.get("/predict/{symbol}")
async def predict_stock(
    symbol: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to predict (1-30)"),
    period: str = Query('1y', description="Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)"),
    interval: str = Query('1d', description="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)"),
    include_technical: bool = Query(True, description="Include technical indicators in the prediction")
) -> Dict[str, Any]:
    """
    Predict stock prices for the given symbol using our trained LSTM-CNN model.
    
    This endpoint fetches historical stock data, processes it with technical indicators,
    and uses our trained model to predict future price movements.
    
    Returns:
        Dictionary containing predictions, historical data, and model confidence metrics
    """
    try:
        logger.info(f"Starting prediction for {symbol} - {days} days")
        
        # 1. Fetch historical data
        logger.info(f"Fetching historical data for {symbol}...")
        df = data_fetcher.get_historical_data(
            symbol=symbol,
            period=period,
            interval=interval
        )
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data available for {symbol}. Please check the symbol and try again."
            )
        
        # 2. Process the data
        logger.info("Processing data with technical indicators...")
        processor = EnhancedStockDataProcessor(
            sequence_length=30,
            prediction_horizon=days,
            add_technical_indicators=include_technical
        )
        
        # Prepare data for prediction
        X, y = processor.prepare_data(df)
        
        if X is None or len(X) == 0:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data to make predictions. Try with a longer time period."
            )
        
        # 3. Make predictions
        logger.info("Generating predictions...")
        y_pred = prediction_service.model.predict(X, verbose=0)
        
        # Inverse transform predictions if scaler is available
        if hasattr(processor, 'target_scaler') and processor.target_scaler is not None:
            y_pred = processor.target_scaler.inverse_transform(y_pred)
        
        # 4. Format response
        last_date = df.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        # Calculate confidence intervals (simplified for this example)
        confidence = float(np.mean(y_pred) / 100)  # Simple confidence metric
        confidence = max(0.5, min(0.95, confidence))  # Keep within reasonable bounds
        
        # Prepare historical data for charting
        historical = df[['open', 'high', 'low', 'close', 'volume']].tail(90)  # Last 90 days
        
        response = {
            "symbol": symbol.upper(),
            "last_updated": datetime.utcnow().isoformat(),
            "prediction_days": days,
            "predictions": y_pred.flatten().tolist(),
            "prediction_dates": [d.strftime('%Y-%m-%d') for d in prediction_dates],
            "confidence": confidence,
            "historical": historical.reset_index().rename(columns={'index': 'date'}).to_dict(orient='records'),
            "model_info": {
                "name": "LSTM-CNN Hybrid Model",
                "version": "1.0.0",
                "last_trained": "2023-09-15"  # This should be dynamic in production
            }
        }
        
        logger.info(f"Successfully generated predictions for {symbol}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_stock for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating predictions: {str(e)}"
        )

@router.get("/recommend/{symbol}")
async def get_recommendation(
    symbol: str,
    period: str = Query('6mo', description="Analysis period (1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)")
) -> Dict[str, Any]:
    """
    Generate a trading recommendation for the given stock symbol.
    
    This endpoint analyzes the stock's technical indicators and price action
    to provide a trading recommendation (BUY, SELL, or HOLD) along with
    confidence scores and detailed technical analysis.
    
    Returns:
        Dictionary containing the recommendation, confidence score, and analysis
    """
    try:
        logger.info(f"Generating recommendation for {symbol}...")
        
        # 1. Fetch historical data
        df = data_fetcher.get_historical_data(
            symbol=symbol,
            period=period,
            interval='1d'  # Daily data for technical analysis
        )
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data available for {symbol}"
            )
        
        # 2. Calculate technical indicators
        # Note: In a real implementation, these would be calculated properly
        # For now, we'll use mock values
        
        # Mock technical analysis (replace with actual calculations)
        last_close = df['close'].iloc[-1]
        ma50 = df['close'].rolling(window=50).mean().iloc[-1]
        ma200 = df['close'].rolling(window=200).mean().iloc[-1]
        
        # Simple recommendation logic (example only)
        if last_close > ma50 > ma200:
            recommendation = "BUY"
            confidence = min(0.95, 0.7 + (last_close - ma50) / ma50 * 10)
        elif last_close < ma50 < ma200:
            recommendation = "SELL"
            confidence = min(0.95, 0.7 + (ma50 - last_close) / last_close * 10)
        else:
            recommendation = "HOLD"
            confidence = 0.6
        
        # 3. Prepare response
        response = {
            "symbol": symbol.upper(),
            "recommendation": recommendation,
            "confidence": round(float(confidence), 2),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "price": round(float(last_close), 2),
                "moving_averages": {
                    "ma50": round(float(ma50), 2),
                    "ma200": round(float(ma200), 2),
                    "trend": "Bullish" if ma50 > ma200 else "Bearish"
                },
                "rsi": 58.2,  # Would be calculated in a real implementation
                "macd": "Bullish" if last_close > ma50 else "Bearish",
                "support": round(float(df['close'].min() * 0.98), 2),  # Example
                "resistance": round(float(df['close'].max() * 1.02), 2)  # Example
            },
            "period_analyzed": {
                "start": df.index[0].strftime('%Y-%m-%d'),
                "end": df.index[-1].strftime('%Y-%m-%d'),
                "days": len(df)
            }
        }
        
        logger.info(f"Generated {recommendation} recommendation for {symbol} with {confidence*100:.1f}% confidence")
        return response
    except Exception as e:
        logger.error(f"Error in get_recommendation for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendation: {str(e)}"
        )
