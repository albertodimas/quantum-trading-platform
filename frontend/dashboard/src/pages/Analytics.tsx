import React, { useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Calendar, Download, Filter } from 'lucide-react';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';

const Analytics: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');

  // Datos simulados para los gráficos
  const performanceData = Array.from({ length: 30 }, (_, i) => ({
    date: format(new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000), 'dd MMM', { locale: es }),
    profit: Math.random() * 2000 - 500,
    trades: Math.floor(Math.random() * 20) + 5,
    winRate: 50 + Math.random() * 30,
  }));

  const strategyPerformance = [
    { name: 'Momentum', value: 35, profit: 12450, trades: 156 },
    { name: 'Mean Reversion', value: 25, profit: 8320, trades: 98 },
    { name: 'AI Strategy', value: 30, profit: 15670, trades: 45 },
    { name: 'Arbitrage', value: 10, profit: 5430, trades: 234 },
  ];

  const assetDistribution = [
    { name: 'BTC', value: 45, amount: 56250 },
    { name: 'ETH', value: 30, amount: 37500 },
    { name: 'BNB', value: 15, amount: 18750 },
    { name: 'SOL', value: 10, amount: 12500 },
  ];

  const COLORS = ['#0070f3', '#10b981', '#f59e0b', '#ef4444'];

  const tradingMetrics = {
    totalTrades: 533,
    winningTrades: 368,
    losingTrades: 165,
    avgWinAmount: 543.21,
    avgLossAmount: 234.56,
    profitFactor: 2.31,
    maxDrawdown: 8.45,
    sharpeRatio: 1.92,
    calmarRatio: 2.87,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Análisis y Estadísticas
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Visualiza el rendimiento detallado de tu sistema de trading
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button className="px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center">
            <Calendar className="h-4 w-4 mr-2" />
            {format(new Date(), 'dd MMM yyyy', { locale: es })}
          </button>
          <button className="px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </button>
          <button className="px-3 py-2 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors flex items-center">
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </button>
        </div>
      </div>

      {/* Selector de Tiempo */}
      <div className="flex space-x-2">
        {(['7d', '30d', '90d', '1y'] as const).map((range) => (
          <button
            key={range}
            onClick={() => setTimeRange(range)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              timeRange === range
                ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            {range === '7d' ? '7 días' : range === '30d' ? '30 días' : range === '90d' ? '90 días' : '1 año'}
          </button>
        ))}
      </div>

      {/* Métricas Principales */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Total Operaciones</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{tradingMetrics.totalTrades}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {tradingMetrics.winningTrades} ganadoras / {tradingMetrics.losingTrades} perdedoras
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Win Rate</p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {((tradingMetrics.winningTrades / tradingMetrics.totalTrades) * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Factor de beneficio: {tradingMetrics.profitFactor}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Drawdown Máximo</p>
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            -{tradingMetrics.maxDrawdown}%
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Calmar Ratio: {tradingMetrics.calmarRatio}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{tradingMetrics.sharpeRatio}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Rendimiento ajustado al riesgo
          </p>
        </div>
      </div>

      {/* Gráficos Principales */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Gráfico de P&L */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Evolución de Ganancias y Pérdidas
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-gray-700" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="profit"
                  stroke="#0070f3"
                  strokeWidth={2}
                  name="P&L ($)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Gráfico de Operaciones */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Volumen de Operaciones y Win Rate
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-gray-700" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis yAxisId="left" stroke="#6b7280" />
                <YAxis yAxisId="right" orientation="right" stroke="#6b7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="trades" fill="#10b981" name="Operaciones" />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="winRate"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="Win Rate (%)"
                  dot={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rendimiento por Estrategia */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Rendimiento por Estrategia
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={strategyPerformance}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {strategyPerformance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            {strategyPerformance.map((strategy, index) => (
              <div key={strategy.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                  <div
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-gray-700 dark:text-gray-300">{strategy.name}</span>
                </div>
                <div className="text-right">
                  <span className="font-medium text-gray-900 dark:text-white">
                    ${strategy.profit.toLocaleString()}
                  </span>
                  <span className="text-gray-500 dark:text-gray-400 ml-2">
                    ({strategy.trades} trades)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Distribución de Activos */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Distribución de Activos
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={assetDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                >
                  {assetDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            {assetDistribution.map((asset, index) => (
              <div key={asset.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                  <div
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-gray-700 dark:text-gray-300">{asset.name}</span>
                </div>
                <div className="text-right">
                  <span className="font-medium text-gray-900 dark:text-white">{asset.value}%</span>
                  <span className="text-gray-500 dark:text-gray-400 ml-2">
                    (${asset.amount.toLocaleString()})
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tabla de Rendimiento Detallado */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Rendimiento Detallado por Mes
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Mes
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Operaciones
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Win Rate
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Retorno
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Drawdown
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {['Enero', 'Febrero', 'Marzo', 'Abril'].map((month) => (
                <tr key={month} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                  <td className="px-4 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {month} 2024
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {Math.floor(Math.random() * 100) + 50}
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {(60 + Math.random() * 20).toFixed(1)}%
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm font-medium text-green-600 dark:text-green-400">
                    +${(Math.random() * 10000).toFixed(2)}
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-green-600 dark:text-green-400">
                    +{(Math.random() * 10).toFixed(2)}%
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-red-600 dark:text-red-400">
                    -{(Math.random() * 5).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Analytics;