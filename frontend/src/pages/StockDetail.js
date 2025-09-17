import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Divider,
  Tabs,
  Tab,
  Chip,
  CircularProgress,
  Paper,
  useMediaQuery,
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import {
  ArrowBack as ArrowBackIcon,
  StarBorder as StarBorderIcon,
  Star as StarIcon,
  ShowChart as ChartIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Article as ArticleIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  CompareArrows as CompareArrowsIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

// Mock data - in a real app, this would come from an API
const mockStockData = {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  currentPrice: 175.34,
  change: 2.15,
  changePercent: 1.24,
  marketCap: 2750000000000,
  peRatio: 28.76,
  volume: 75000000,
  avgVolume: 80000000,
  high52Week: 198.23,
  low52Week: 124.17,
  dividendYield: 0.52,
  about: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod.',
  sector: 'Technology',
  industry: 'Consumer Electronics',
  website: 'https://www.apple.com',
};

// Mock historical data for charts
const generateHistoricalData = (days = 30) => {
  const today = new Date();
  const data = [];
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    data.push({
      date: date.toISOString().split('T')[0],
      open: 170 + Math.random() * 10,
      high: 172 + Math.random() * 12,
      low: 168 + Math.random() * 8,
      close: 169 + Math.random() * 11,
      volume: Math.floor(1000000 + Math.random() * 1000000),
    });
  }
  
  return data;
};

const historicalData = generateHistoricalData(30);

const StockDetail = () => {
  const { symbol } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState(0);
  const [isFavorite, setIsFavorite] = useState(false);
  const [loading, setLoading] = useState(true);
  const [stockData, setStockData] = useState(null);
  const isMobile = useMediaQuery((theme) => theme.breakpoints.down('md'));

  useEffect(() => {
    // Simulate API call
    const fetchStockData = async () => {
      try {
        setLoading(true);
        // In a real app, you would fetch data from your API here
        // const response = await fetch(`/api/stocks/${symbol}`);
        // const data = await response.json();
        // setStockData(data);
        
        // Using mock data for now
        setTimeout(() => {
          setStockData({
            ...mockStockData,
            symbol: symbol.toUpperCase(),
            historicalData: historicalData,
          });
          setLoading(false);
        }, 800);
      } catch (error) {
        console.error('Error fetching stock data:', error);
        setLoading(false);
      }
    };

    fetchStockData();
  }, [symbol]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const toggleFavorite = () => {
    setIsFavorite(!isFavorite);
    // In a real app, you would update the user's favorites in the backend
  };

  if (loading || !stockData) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  // Prepare chart data
  const lineChartData = {
    labels: stockData.historicalData.map((item) => item.date),
    datasets: [
      {
        label: 'Closing Price',
        data: stockData.historicalData.map((item) => item.close),
        borderColor: 'rgb(25, 118, 210)',
        backgroundColor: 'rgba(25, 118, 210, 0.1)',
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const volumeChartData = {
    labels: stockData.historicalData.map((item) => item.date),
    datasets: [
      {
        label: 'Volume',
        data: stockData.historicalData.map((item) => item.volume),
        backgroundColor: 'rgba(25, 118, 210, 0.7)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
      },
      y: {
        grid: {
          borderDash: [3, 3],
        },
      },
    },
  };

  return (
    <Box>
      {/* Header with back button and stock info */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(-1)}
          sx={{ mr: 2 }}
        >
          Back
        </Button>
        <Box sx={{ flexGrow: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
            <Typography variant="h4" component="h1" sx={{ mr: 2 }}>
              {stockData.symbol}
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mr: 2 }}>
              {stockData.name}
            </Typography>
            <Chip
              label={stockData.sector}
              size="small"
              color="primary"
              variant="outlined"
              sx={{ borderRadius: 1, mr: 1 }}
            />
            <IconButton size="small" onClick={toggleFavorite} color={isFavorite ? 'warning' : 'default'}>
              {isFavorite ? <StarIcon /> : <StarBorderIcon />}
            </IconButton>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            <Typography variant="h4" sx={{ fontWeight: 'bold', mr: 2 }}>
              {stockData.currentPrice.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
            </Typography>
            <Chip
              icon={stockData.change >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
              label={`${stockData.change >= 0 ? '+' : ''}${stockData.change.toFixed(2)} (${stockData.changePercent.toFixed(2)}%)`}
              color={stockData.change >= 0 ? 'success' : 'error'}
              sx={{ borderRadius: 1, fontWeight: 'medium' }}
            />
          </Box>
        </Box>
        <Box>
          <Button variant="contained" color="primary" sx={{ mr: 1, textTransform: 'none' }}>
            Buy
          </Button>
          <Button variant="outlined" color="primary" sx={{ textTransform: 'none' }}>
            Sell
          </Button>
        </Box>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant={isMobile ? 'scrollable' : 'standard'}
          scrollButtons="auto"
          allowScrollButtonsMobile
        >
          <Tab icon={<ChartIcon />} label="Chart" />
          <Tab icon={<AssessmentIcon />} label="Statistics" />
          <Tab icon={<ArticleIcon />} label="News" />
          <Tab icon={<TimelineIcon />} label="Analysis" />
          <Tab icon={<CompareArrowsIcon />} label="Compare" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <Card sx={{ p: 2, mb: 3 }}>
              <Box sx={{ height: 400 }}>
                <Line data={lineChartData} options={chartOptions} />
              </Box>
            </Card>
            <Card sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Volume
              </Typography>
              <Box sx={{ height: 200 }}>
                <Bar data={volumeChartData} options={chartOptions} />
              </Box>
            </Card>
          </Grid>
          <Grid item xs={12} lg={4}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  About {stockData.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {stockData.about}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary">
                  Industry: {stockData.industry}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary">
                  Website:{' '}
                  <a href={stockData.website} target="_blank" rel="noopener noreferrer">
                    {stockData.website}
                  </a>
                </Typography>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Key Statistics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <StatItem label="Market Cap" value={`$${(stockData.marketCap / 1000000000).toFixed(2)}B`} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="P/E Ratio" value={stockData.peRatio} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="Volume" value={(stockData.volume / 1000000).toFixed(2) + 'M'} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="Avg Volume" value={(stockData.avgVolume / 1000000).toFixed(2) + 'M'} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="52 Week High" value={`$${stockData.high52Week}`} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="52 Week Low" value={`$${stockData.low52Week}`} />
                  </Grid>
                  <Grid item xs={6}>
                    <StatItem label="Dividend Yield" value={`${stockData.dividendYield}%`} />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Statistics
            </Typography>
            <Typography color="text.secondary">
              Detailed statistics coming soon...
            </Typography>
          </CardContent>
        </Card>
      )}

      {activeTab === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              News
            </Typography>
            <Typography color="text.secondary">
              Latest news about {stockData.name} will appear here...
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

// Helper component for statistics items
const StatItem = ({ label, value }) => (
  <Box sx={{ mb: 2 }}>
    <Typography variant="caption" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="body1">{value}</Typography>
  </Box>
);

export default StockDetail;
