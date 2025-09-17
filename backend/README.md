# Stock Market Analysis Backend

A FastAPI-based backend service for stock market analysis, predictions, and trading recommendations.

## 🚀 Features

- **RESTful API** with OpenAPI documentation
- **Stock Data Fetching** from multiple sources (Yahoo Finance, Alpha Vantage)
- **Machine Learning** models for stock price prediction
- **Technical Analysis** with various indicators
- **Trading Recommendations** based on technical and fundamental analysis
- **Asynchronous** request handling
- **JWT Authentication** (coming soon)
- **Rate Limiting** (coming soon)

## 🛠 Tech Stack

- **Framework**: FastAPI
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: TensorFlow, scikit-learn
- **Technical Analysis**: TA-Lib, pandas_ta
- **Data Fetching**: yfinance, Alpha Vantage API
- **Documentation**: Swagger UI, ReDoc
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: black, isort, flake8, mypy

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-market-analysis.git
   cd stock-market-analysis/backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   POLYGON_API_KEY=your_api_key_here
   ```

5. **Run the development server**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API documentation**
   - Swagger UI: http://localhost:8000/api/docs
   - ReDoc: http://localhost:8000/api/redoc

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app tests/

# Run a specific test file
pytest tests/test_api.py -v
```

## 🧩 Project Structure

```
backend/
├── app/                      # Application source code
│   ├── api/                  # API routes
│   ├── core/                 # Core functionality
│   │   ├── config.py         # Application settings
│   │   └── logger.py         # Logging configuration
│   ├── models/               # Database models (coming soon)
│   ├── services/             # Business logic
│   │   ├── data_fetcher.py   # Data fetching services
│   │   └── prediction.py     # Prediction services
│   └── main.py               # FastAPI application
├── tests/                    # Test files
├── .env.example              # Example environment variables
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## 📚 API Documentation

Once the server is running, you can access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used
- [yfinance](https://github.com/ranaroussi/yfinance) - For fetching stock data
- [Alpha Vantage](https://www.alphavantage.co/) - For financial market data
- [TensorFlow](https://www.tensorflow.org/) - For machine learning capabilities
