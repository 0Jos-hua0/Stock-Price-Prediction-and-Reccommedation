from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.services.data_fetcher import DataFetcher

router = APIRouter()
logger = logging.getLogger(__name__)
data_fetcher = DataFetcher()

@router.get("/company/{symbol}")
async def get_company_info(symbol: str) -> Dict[str, Any]:
    """
    Get company information for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Dictionary containing company information
    """
    try:
        logger.info(f"Fetching company info for {symbol}")
        company_info = data_fetcher.get_company_info(symbol)
        
        if not company_info:
            raise HTTPException(
                status_code=404,
                detail=f"No information found for symbol {symbol}"
            )
            
        return company_info
        
    except Exception as e:
        logger.error(f"Error fetching company info for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching company information: {str(e)}"
        )
