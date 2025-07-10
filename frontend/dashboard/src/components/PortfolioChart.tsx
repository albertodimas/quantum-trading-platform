import React, { useState } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';

// Datos simulados
const generateData = () => {
  const data = [];
  const now = new Date();
  let balance = 100000;
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    // Simular cambios de balance
    const change = (Math.random() - 0.45) * 0.02 * balance;
    balance += change;
    
    data.push({
      date: date.toISOString(),
      balance: Math.round(balance * 100) / 100,
      pnl: Math.round(change * 100) / 100,
    });
  }
  
  return data;
};

const PortfolioChart: React.FC = () => {
  const [data] = useState(generateData());
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  const filteredData = data.slice(
    timeRange === '7d' ? -7 : timeRange === '30d' ? -30 : -90
  );

  const formatXAxis = (tickItem: string) => {
    return format(new Date(tickItem), 'dd MMM', { locale: es });
  };

  const formatTooltip = (value: number, name: string) => {
    if (name === 'balance') return `$${value.toLocaleString()}`;
    if (name === 'pnl') return `$${value.toLocaleString()}`;
    return value;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Evolución del Portfolio
        </h3>
        <div className="flex space-x-2">
          {(['7d', '30d', '90d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                timeRange === range
                  ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filteredData}>
            <defs>
              <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#0070f3" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#0070f3" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e5e7eb"
              className="dark:stroke-gray-700"
            />
            <XAxis
              dataKey="date"
              tickFormatter={formatXAxis}
              stroke="#6b7280"
              className="text-xs"
            />
            <YAxis
              stroke="#6b7280"
              className="text-xs"
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
              }}
              formatter={formatTooltip}
              labelFormatter={(label) =>
                format(new Date(label), 'dd MMMM yyyy', { locale: es })
              }
            />
            <Area
              type="monotone"
              dataKey="balance"
              stroke="#0070f3"
              fillOpacity={1}
              fill="url(#colorBalance)"
              strokeWidth={2}
              name="Balance"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Resumen */}
      <div className="mt-4 grid grid-cols-3 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Balance Inicial</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            ${filteredData[0]?.balance.toLocaleString() || '0'}
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Balance Actual</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            ${filteredData[filteredData.length - 1]?.balance.toLocaleString() || '0'}
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Retorno Total</p>
          <p className={`text-lg font-semibold ${
            filteredData.length > 1 && 
            filteredData[filteredData.length - 1].balance > filteredData[0].balance
              ? 'text-green-600 dark:text-green-400'
              : 'text-red-600 dark:text-red-400'
          }`}>
            {filteredData.length > 1
              ? `${(
                  ((filteredData[filteredData.length - 1].balance - filteredData[0].balance) /
                    filteredData[0].balance) *
                  100
                ).toFixed(2)}%`
              : '0%'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PortfolioChart;