"""Main FastAPI application module."""
from datetime import datetime
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Import core configurations
from app.core.config import settings
from app.core.logger import get_logger, logger

# Import routers
from app.api.prediction_api import router as prediction_router
from app.api.company_api import router as company_router

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for stock market analysis, predictions, and trading recommendations",
    version=settings.VERSION,
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests."""
    logger.info(
        "Incoming request",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None,
            "path_params": request.path_params,
            "query_params": dict(request.query_params),
        },
    )
    
    try:
        response = await call_next(request)
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
            },
        )
        return response
    except Exception as e:
        logger.exception("Request failed")
        raise

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.error(
        "Request validation error",
        extra={
            "errors": exc.errors(),
            "body": exc.body,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Include API routers
app.include_router(
    prediction_router,
    prefix=f"{settings.API_PREFIX}/v1",
    tags=["predictions"]
)

app.include_router(
    company_router,
    prefix=f"{settings.API_PREFIX}/v1",
    tags=["company"]
)

# Health check and info endpoint
@app.get("/")
async def root():
    """Root endpoint with API information and health status."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "status": "operational",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json"
        },
        "endpoints": {
            "predictions": f"{settings.API_PREFIX}/v1/predict",
            "company": f"{settings.API_PREFIX}/v1/company"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    # TODO: Add database and external service health checks
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# HTTP Exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTP Exception",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    logger.info("Starting up Stock Market Analysis API...")
    
    # Verify required environment variables
    required_vars = ["POLYGON_API_KEY"]
    missing_vars = [var for var in required_vars if not getattr(settings, var, None)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.critical(error_msg)
        raise RuntimeError(error_msg)
    
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
