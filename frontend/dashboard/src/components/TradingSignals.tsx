import React from 'react';
import { ArrowUpCircle, ArrowDownCircle, Clock, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';

interface Signal {
  id: string;
  type: 'buy' | 'sell';
  symbol: string;
  price: number;
  confidence: number;
  strategy: string;
  timestamp: Date;
  status: 'active' | 'executed' | 'expired';
}

const TradingSignals: React.FC = () => {
  // Señales simuladas
  const signals: Signal[] = [
    {
      id: '1',
      type: 'buy',
      symbol: 'BTC/USDT',
      price: 67500,
      confidence: 0.85,
      strategy: 'Momentum',
      timestamp: new Date(),
      status: 'active',
    },
    {
      id: '2',
      type: 'sell',
      symbol: 'ETH/USDT',
      price: 3470,
      confidence: 0.72,
      strategy: 'Mean Reversion',
      timestamp: new Date(Date.now() - 1000 * 60 * 5),
      status: 'active',
    },
    {
      id: '3',
      type: 'buy',
      symbol: 'SOL/USDT',
      price: 175.50,
      confidence: 0.91,
      strategy: 'AI Strategy',
      timestamp: new Date(Date.now() - 1000 * 60 * 10),
      status: 'executed',
    },
    {
      id: '4',
      type: 'sell',
      symbol: 'BNB/USDT',
      price: 435.20,
      confidence: 0.68,
      strategy: 'Arbitrage',
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      status: 'expired',
    },
  ];

  const getStatusColor = (status: Signal['status']) => {
    switch (status) {
      case 'active':
        return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400';
      case 'executed':
        return 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400';
      case 'expired':
        return 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-400';
      default:
        return '';
    }
  };

  const getStatusText = (status: Signal['status']) => {
    switch (status) {
      case 'active':
        return 'Activa';
      case 'executed':
        return 'Ejecutada';
      case 'expired':
        return 'Expirada';
      default:
        return '';
    }
  };

  const getStrategyIcon = (strategy: string) => {
    if (strategy.includes('AI')) return <Zap className="h-4 w-4" />;
    return null;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Señales de Trading
        </h3>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          Últimas 24h
        </span>
      </div>

      <div className="space-y-3">
        {signals.map((signal, index) => (
          <motion.div
            key={signal.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`p-4 rounded-lg border ${
              signal.status === 'active'
                ? 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/10'
                : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/30'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full ${
                    signal.type === 'buy'
                      ? 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400'
                      : 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400'
                  }`}
                >
                  {signal.type === 'buy' ? (
                    <ArrowUpCircle className="h-5 w-5" />
                  ) : (
                    <ArrowDownCircle className="h-5 w-5" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {signal.symbol}
                    </p>
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(
                        signal.status
                      )}`}
                    >
                      {getStatusText(signal.status)}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {signal.type === 'buy' ? 'Comprar' : 'Vender'} a $
                    {signal.price.toLocaleString()}
                  </p>
                  <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                    <span className="flex items-center">
                      {getStrategyIcon(signal.strategy)}
                      {signal.strategy}
                    </span>
                    <span className="flex items-center">
                      <Clock className="h-3 w-3 mr-1" />
                      {format(signal.timestamp, 'HH:mm', { locale: es })}
                    </span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Confianza
                </p>
                <p
                  className={`text-sm font-semibold ${
                    signal.confidence > 0.8
                      ? 'text-green-600 dark:text-green-400'
                      : signal.confidence > 0.6
                      ? 'text-yellow-600 dark:text-yellow-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}
                >
                  {(signal.confidence * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 text-center">
        <button className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium">
          Ver todas las señales →
        </button>
      </div>
    </div>
  );
};

export default TradingSignals;