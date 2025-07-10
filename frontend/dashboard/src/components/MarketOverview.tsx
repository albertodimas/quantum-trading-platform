import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'framer-motion';

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume24h: number;
  marketCap: number;
  sparkline: number[];
}

const MarketOverview: React.FC = () => {
  // Datos simulados
  const markets: MarketData[] = [
    {
      symbol: 'BTC',
      name: 'Bitcoin',
      price: 67543.21,
      change24h: 2.34,
      volume24h: 45234567890,
      marketCap: 1324567890123,
      sparkline: [65000, 66000, 65500, 66500, 67000, 67200, 67543],
    },
    {
      symbol: 'ETH',
      name: 'Ethereum',
      price: 3456.78,
      change24h: -1.23,
      volume24h: 23456789012,
      marketCap: 456789012345,
      sparkline: [3500, 3480, 3490, 3470, 3460, 3450, 3456],
    },
    {
      symbol: 'BNB',
      name: 'Binance Coin',
      price: 432.10,
      change24h: 0.56,
      volume24h: 1234567890,
      marketCap: 67890123456,
      sparkline: [430, 431, 429, 431, 432, 431, 432],
    },
    {
      symbol: 'SOL',
      name: 'Solana',
      price: 178.90,
      change24h: 5.67,
      volume24h: 2345678901,
      marketCap: 56789012345,
      sparkline: [169, 172, 174, 175, 177, 178, 179],
    },
  ];

  const MiniSparkline = ({ data }: { data: number[] }) => {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    const width = 80;
    const height = 30;
    
    const points = data
      .map((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * height;
        return `${x},${y}`;
      })
      .join(' ');

    return (
      <svg width={width} height={height} className="inline-block">
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={
            data[data.length - 1] > data[0]
              ? 'text-green-500'
              : 'text-red-500'
          }
        />
      </svg>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Resumen del Mercado
      </h3>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead>
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Moneda
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Precio
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                24h %
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Volumen 24h
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Gráfico 7d
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {markets.map((market, index) => (
              <motion.tr
                key={market.symbol}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              >
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
                      <span className="text-xs font-bold text-gray-600 dark:text-gray-300">
                        {market.symbol}
                      </span>
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {market.name}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {market.symbol}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    ${market.price.toLocaleString()}
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      market.change24h > 0
                        ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400'
                        : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400'
                    }`}
                  >
                    {market.change24h > 0 ? (
                      <TrendingUp className="w-3 h-3 mr-1" />
                    ) : (
                      <TrendingDown className="w-3 h-3 mr-1" />
                    )}
                    {Math.abs(market.change24h).toFixed(2)}%
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  ${(market.volume24h / 1e9).toFixed(2)}B
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <MiniSparkline data={market.sparkline} />
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MarketOverview;