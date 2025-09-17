import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Paper,
  Divider,
  Chip,
  CircularProgress,
  Alert,
  useMediaQuery,
} from '@mui/material';
import { Search as SearchIcon, ShowChart as ChartIcon } from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Mock data - in a real app, this would come from an API
const mockStocks = [
  { symbol: 'AAPL', name: 'Apple Inc.', price: 175.34, change: 2.15, changePercent: 1.24 },
  { symbol: 'MSFT', name: 'Microsoft', price: 315.76, change: -1.23, changePercent: -0.39 },
  { symbol: 'GOOGL', name: 'Alphabet', price: 135.21, change: 0.87, changePercent: 0.65 },
  { symbol: 'AMZN', name: 'Amazon', price: 174.99, change: 3.45, changePercent: 2.01 },
  { symbol: 'META', name: 'Meta', price: 300.45, change: -2.34, changePercent: -0.77 },
];

const mockMarketData = {
  sp500: { value: 4457.49, change: 12.34, changePercent: 0.28 },
  nasdaq: { value: 13787.92, change: 45.67, changePercent: 0.33 },
  dow: { value: 34261.42, change: -23.45, changePercent: -0.07 },
};

// Chart data
const chartData = {
  labels: Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }),
  datasets: [
    {
      label: 'Portfolio Value',
      data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 10000) + 90000),
      borderColor: 'rgb(25, 118, 210)',
      backgroundColor: 'rgba(25, 118, 210, 0.1)',
      tension: 0.4,
      fill: true,
    },
  ],
};

const chartOptions = {
  responsive: true,
  plugins: {
    legend: {
      display: false,
    },
    tooltip: {
      mode: 'index',
      intersect: false,
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

const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const isMobile = useMediaQuery((theme) => theme.breakpoints.down('sm'));

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/stock/${searchQuery.toUpperCase()}`);
    }
  };

  const filteredStocks = mockStocks.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Welcome back! Here's what's happening with your portfolio today.
        </Typography>
      </Box>

      {/* Search Bar */}
      <Paper
        component="form"
        onSubmit={handleSearch}
        sx={{
          p: '2px 4px',
          display: 'flex',
          alignItems: 'center',
          mb: 3,
          borderRadius: 2,
          boxShadow: 1,
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search stocks..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            sx: { border: 'none', '& fieldset': { border: 'none' } },
            startAdornment: <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />,
          }}
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          sx={{ ml: 1, borderRadius: 2, textTransform: 'none', px: 3 }}
        >
          Search
        </Button>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Market Overview */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Market Overview
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(mockMarketData).map(([key, value]) => (
            <Grid item xs={12} sm={4} key={key}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    {key.toUpperCase()}
                  </Typography>
                  <Typography variant="h6" component="div">
                    {value.value.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <Chip
                      size="small"
                      label={`${value.change > 0 ? '+' : ''}${value.change.toFixed(2)} (${value.changePercent.toFixed(2)}%)`}
                      color={value.change >= 0 ? 'success' : 'error'}
                      sx={{ borderRadius: 1, fontWeight: 'medium' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Portfolio Value Chart */}
      <Card sx={{ mb: 4, p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Portfolio Value</Typography>
          <Chip
            label="1M"
            size="small"
            color="primary"
            variant="outlined"
            sx={{ borderRadius: 1 }}
          />
        </Box>
        <Box sx={{ height: 300 }}>
          <Line data={chartData} options={chartOptions} />
        </Box>
      </Card>

      {/* Watchlist */}
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Watchlist</Typography>
          <Button size="small" color="primary">
            View All
          </Button>
        </Box>
        <Card>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : filteredStocks.length > 0 ? (
            <Box sx={{ overflowX: 'auto' }}>
              <Box sx={{ minWidth: 600 }}>
                {filteredStocks.map((stock, index) => (
                  <Box key={stock.symbol}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2,
                        '&:hover': { bgcolor: 'action.hover', cursor: 'pointer' },
                      }}
                      onClick={() => navigate(`/stock/${stock.symbol}`)}
                    >
                      <Box sx={{ width: 40, height: 40, bgcolor: 'primary.light', borderRadius: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', mr: 2 }}>
                        <ChartIcon sx={{ color: 'primary.contrastText' }} />
                      </Box>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="subtitle1">{stock.symbol}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {stock.name}
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'right' }}>
                        <Typography variant="subtitle1">
                          {stock.price.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
                        </Typography>
                        <Chip
                          size="small"
                          label={`${stock.change > 0 ? '+' : ''}${stock.change.toFixed(2)} (${stock.changePercent.toFixed(2)}%)`}
                          color={stock.change >= 0 ? 'success' : 'error'}
                          sx={{ borderRadius: 1, height: 20, fontSize: '0.7rem' }}
                        />
                      </Box>
                    </Box>
                    {index < filteredStocks.length - 1 && <Divider />}
                  </Box>
                ))}
              </Box>
            </Box>
          ) : (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography color="text.secondary">No stocks found matching your search.</Typography>
            </Box>
          )}
        </Card>
      </Box>
    </Box>
  );
};

export default Dashboard;
