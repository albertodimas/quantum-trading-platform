import React from 'react';
import { CheckCircle, XCircle, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';

interface Trade {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  pnl?: number;
  pnlPercent?: number;
  status: 'open' | 'closed' | 'cancelled';
  strategy: string;
  openTime: Date;
  closeTime?: Date;
}

const RecentTrades: React.FC = () => {
  // Trades simulados
  const trades: Trade[] = [
    {
      id: '1',
      symbol: 'BTC/USDT',
      type: 'buy',
      quantity: 0.15,
      entryPrice: 65000,
      exitPrice: 67500,
      pnl: 375,
      pnlPercent: 3.85,
      status: 'closed',
      strategy: 'Momentum',
      openTime: new Date(Date.now() - 1000 * 60 * 60 * 2),
      closeTime: new Date(Date.now() - 1000 * 60 * 30),
    },
    {
      id: '2',
      symbol: 'ETH/USDT',
      type: 'sell',
      quantity: 2.5,
      entryPrice: 3500,
      status: 'open',
      strategy: 'Mean Reversion',
      openTime: new Date(Date.now() - 1000 * 60 * 45),
    },
    {
      id: '3',
      symbol: 'SOL/USDT',
      type: 'buy',
      quantity: 10,
      entryPrice: 170,
      exitPrice: 165,
      pnl: -50,
      pnlPercent: -2.94,
      status: 'closed',
      strategy: 'AI Strategy',
      openTime: new Date(Date.now() - 1000 * 60 * 60 * 4),
      closeTime: new Date(Date.now() - 1000 * 60 * 60 * 2),
    },
    {
      id: '4',
      symbol: 'BNB/USDT',
      type: 'buy',
      quantity: 5,
      entryPrice: 425,
      status: 'cancelled',
      strategy: 'Arbitrage',
      openTime: new Date(Date.now() - 1000 * 60 * 60 * 3),
    },
  ];

  const getStatusIcon = (status: Trade['status']) => {
    switch (status) {
      case 'closed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'open':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'cancelled':
        return <XCircle className="h-5 w-5 text-gray-500" />;
      default:
        return null;
    }
  };

  const getStatusText = (status: Trade['status']) => {
    switch (status) {
      case 'closed':
        return 'Cerrada';
      case 'open':
        return 'Abierta';
      case 'cancelled':
        return 'Cancelada';
      default:
        return '';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Operaciones Recientes
        </h3>
        <button className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium">
          Ver historial completo →
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead>
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Par / Estrategia
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Tipo
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Cantidad
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Entrada / Salida
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                P&L
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Estado
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Tiempo
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {trades.map((trade, index) => (
              <motion.tr
                key={trade.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              >
                <td className="px-4 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {trade.symbol}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {trade.strategy}
                    </div>
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      trade.type === 'buy'
                        ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400'
                        : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400'
                    }`}
                  >
                    {trade.type === 'buy' ? 'Compra' : 'Venta'}
                  </span>
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {trade.quantity}
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="text-sm">
                    <div className="text-gray-900 dark:text-white">
                      ${trade.entryPrice.toLocaleString()}
                    </div>
                    {trade.exitPrice && (
                      <div className="text-gray-500 dark:text-gray-400">
                        ${trade.exitPrice.toLocaleString()}
                      </div>
                    )}
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  {trade.pnl !== undefined ? (
                    <div>
                      <div
                        className={`text-sm font-medium ${
                          trade.pnl > 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                      </div>
                      <div
                        className={`text-xs ${
                          trade.pnl > 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {trade.pnlPercent! > 0 ? '+' : ''}
                        {trade.pnlPercent!.toFixed(2)}%
                      </div>
                    </div>
                  ) : (
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      -
                    </span>
                  )}
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {getStatusIcon(trade.status)}
                    <span className="ml-2 text-sm text-gray-900 dark:text-white">
                      {getStatusText(trade.status)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  <div>
                    {format(trade.openTime, 'dd/MM HH:mm', { locale: es })}
                  </div>
                  {trade.closeTime && (
                    <div className="text-xs">
                      Cerrada: {format(trade.closeTime, 'HH:mm', { locale: es })}
                    </div>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Resumen */}
      <div className="mt-4 grid grid-cols-4 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Total Trades</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {trades.length}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Win Rate</p>
          <p className="text-sm font-medium text-green-600 dark:text-green-400">
            66.7%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">P&L Total</p>
          <p className="text-sm font-medium text-green-600 dark:text-green-400">
            +$325.00
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Abiertas</p>
          <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
            {trades.filter((t) => t.status === 'open').length}
          </p>
        </div>
      </div>
    </div>
  );
};

export default RecentTrades;