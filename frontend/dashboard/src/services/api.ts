import axios from 'axios';
import { io, Socket } from 'socket.io-client';

// Configuración de la API
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

// Crear instancia de axios
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para manejar tokens
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Interceptor para manejar errores
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Redirigir al login si no está autorizado
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// WebSocket Manager
class WebSocketManager {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 5000;

  connect(): Socket {
    if (this.socket?.connected) {
      return this.socket;
    }

    this.socket = io(WS_BASE_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectInterval,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket conectado');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket desconectado:', reason);
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  getSocket(): Socket | null {
    return this.socket;
  }

  emit(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket no conectado');
    }
  }

  on(event: string, callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on(event, callback);
    }
  }

  off(event: string, callback?: (data: any) => void) {
    if (this.socket) {
      this.socket.off(event, callback);
    }
  }
}

export const wsManager = new WebSocketManager();

// API Endpoints
export const tradingAPI = {
  // Market Data
  getMarketData: (symbol: string) => api.get(`/api/v1/market/${symbol}`),
  getOrderBook: (symbol: string) => api.get(`/api/v1/market/${symbol}/orderbook`),
  getTicker: (symbol: string) => api.get(`/api/v1/market/${symbol}/ticker`),
  getCandles: (symbol: string, interval: string, limit?: number) => 
    api.get(`/api/v1/market/${symbol}/candles`, { params: { interval, limit } }),

  // Trading
  getPositions: () => api.get('/api/v1/trading/positions'),
  getOrders: () => api.get('/api/v1/trading/orders'),
  createOrder: (data: any) => api.post('/api/v1/trading/orders', data),
  cancelOrder: (orderId: string) => api.delete(`/api/v1/trading/orders/${orderId}`),
  
  // Strategies
  getStrategies: () => api.get('/api/v1/strategies'),
  getStrategyStatus: (strategyId: string) => api.get(`/api/v1/strategies/${strategyId}/status`),
  updateStrategy: (strategyId: string, data: any) => api.put(`/api/v1/strategies/${strategyId}`, data),
  toggleStrategy: (strategyId: string, active: boolean) => 
    api.post(`/api/v1/strategies/${strategyId}/${active ? 'start' : 'stop'}`),

  // Agents
  getAgentsStatus: () => api.get('/api/v1/agents/status'),
  getAgentAnalysis: (agentId: string) => api.get(`/api/v1/agents/${agentId}/analysis`),

  // Analytics
  getPerformanceMetrics: (period?: string) => 
    api.get('/api/v1/analytics/performance', { params: { period } }),
  getTradeHistory: (limit?: number) => 
    api.get('/api/v1/analytics/trades', { params: { limit } }),
  getPortfolioStats: () => api.get('/api/v1/analytics/portfolio'),

  // Settings
  getSettings: () => api.get('/api/v1/settings'),
  updateSettings: (data: any) => api.put('/api/v1/settings', data),
  testExchangeConnection: () => api.post('/api/v1/settings/test-connection'),
};

// WebSocket Events
export const wsEvents = {
  // Suscripciones
  subscribeToMarket: (symbol: string) => {
    wsManager.emit('subscribe', { channel: 'market', symbol });
  },
  
  unsubscribeFromMarket: (symbol: string) => {
    wsManager.emit('unsubscribe', { channel: 'market', symbol });
  },

  subscribeToTrades: () => {
    wsManager.emit('subscribe', { channel: 'trades' });
  },

  subscribeToSignals: () => {
    wsManager.emit('subscribe', { channel: 'signals' });
  },

  subscribeToAgents: () => {
    wsManager.emit('subscribe', { channel: 'agents' });
  },

  // Listeners
  onMarketUpdate: (callback: (data: any) => void) => {
    wsManager.on('market_update', callback);
  },

  onTradeUpdate: (callback: (data: any) => void) => {
    wsManager.on('trade_update', callback);
  },

  onSignalUpdate: (callback: (data: any) => void) => {
    wsManager.on('signal_update', callback);
  },

  onAgentUpdate: (callback: (data: any) => void) => {
    wsManager.on('agent_update', callback);
  },

  // Cleanup
  removeAllListeners: () => {
    ['market_update', 'trade_update', 'signal_update', 'agent_update'].forEach(event => {
      wsManager.off(event);
    });
  },
};