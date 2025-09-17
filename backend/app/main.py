import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import routers
from app.api.prediction_api import router as prediction_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Stock Market Analysis API",
    description="API for stock market analysis and prediction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    prediction_router,
    prefix="/api/v1",
    tags=["predictions"]
)

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {
        "message": "Welcome to Stock Market Analysis API",
        "status": "operational",
        "version": "0.1.0"
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "detail": exc.detail,
        "status_code": exc.status_code
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    logger.info("Starting up Stock Market Analysis API...")
    
    # Verify required environment variables
    required_vars = ["POLYGON_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(
            f"The following required environment variables are missing: {', '.join(missing_vars)}. "
            "Some features may not work correctly."
        )
    else:
        logger.info("All required environment variables are set.")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    logger.info("Shutting down Stock Market Analysis API...")

# For running with uvicorn programmatically
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
