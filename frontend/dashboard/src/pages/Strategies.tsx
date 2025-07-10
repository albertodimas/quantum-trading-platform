import React, { useState } from 'react';
import { Play, Pause, Settings, TrendingUp, Brain, BarChart3, Zap } from 'lucide-react';
import { Switch } from '@radix-ui/react-switch';

interface Strategy {
  id: string;
  name: string;
  type: string;
  description: string;
  performance: {
    winRate: number;
    totalTrades: number;
    pnl: number;
    sharpeRatio: number;
  };
  status: 'active' | 'paused' | 'inactive';
  allocation: number;
  icon: React.ReactNode;
}

const Strategies: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([
    {
      id: '1',
      name: 'Momentum Strategy',
      type: 'momentum',
      description: 'Identifica y sigue tendencias fuertes en el mercado usando indicadores técnicos avanzados.',
      performance: {
        winRate: 68.5,
        totalTrades: 156,
        pnl: 12450,
        sharpeRatio: 1.85,
      },
      status: 'active',
      allocation: 30,
      icon: <TrendingUp className="h-6 w-6" />,
    },
    {
      id: '2',
      name: 'Mean Reversion',
      type: 'mean_reversion',
      description: 'Aprovecha los movimientos extremos del mercado para operar reversiones hacia la media.',
      performance: {
        winRate: 72.3,
        totalTrades: 98,
        pnl: 8320,
        sharpeRatio: 1.92,
      },
      status: 'active',
      allocation: 25,
      icon: <BarChart3 className="h-6 w-6" />,
    },
    {
      id: '3',
      name: 'AI Strategy',
      type: 'ai',
      description: 'Utiliza múltiples agentes de IA para análisis técnico, sentimiento y gestión de riesgo.',
      performance: {
        winRate: 75.8,
        totalTrades: 45,
        pnl: 15670,
        sharpeRatio: 2.15,
      },
      status: 'active',
      allocation: 35,
      icon: <Brain className="h-6 w-6" />,
    },
    {
      id: '4',
      name: 'Arbitrage',
      type: 'arbitrage',
      description: 'Detecta y ejecuta oportunidades de arbitraje entre diferentes exchanges y pares.',
      performance: {
        winRate: 89.2,
        totalTrades: 234,
        pnl: 5430,
        sharpeRatio: 3.21,
      },
      status: 'paused',
      allocation: 10,
      icon: <Zap className="h-6 w-6" />,
    },
  ]);

  const toggleStrategy = (strategyId: string) => {
    setStrategies(strategies.map(s => 
      s.id === strategyId 
        ? { ...s, status: s.status === 'active' ? 'paused' : 'active' }
        : s
    ));
  };

  const getStatusColor = (status: Strategy['status']) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400';
      case 'paused':
        return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400';
      case 'inactive':
        return 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-400';
      default:
        return '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Estrategias de Trading
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Gestiona y monitorea tus estrategias automatizadas
          </p>
        </div>
        <button className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors">
          + Nueva Estrategia
        </button>
      </div>

      {/* Resumen de Performance */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Estrategias Activas</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {strategies.filter(s => s.status === 'active').length}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Win Rate Promedio</p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {(strategies.reduce((acc, s) => acc + s.performance.winRate, 0) / strategies.length).toFixed(1)}%
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">P&L Total</p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            ${strategies.reduce((acc, s) => acc + s.performance.pnl, 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio Promedio</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {(strategies.reduce((acc, s) => acc + s.performance.sharpeRatio, 0) / strategies.length).toFixed(2)}
          </p>
        </div>
      </div>

      {/* Lista de Estrategias */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {strategies.map((strategy) => (
          <div
            key={strategy.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start space-x-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400">
                  {strategy.icon}
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    {strategy.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {strategy.description}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(strategy.status)}`}>
                  {strategy.status === 'active' ? 'Activa' : strategy.status === 'paused' ? 'Pausada' : 'Inactiva'}
                </span>
                <button
                  onClick={() => toggleStrategy(strategy.id)}
                  className={`p-1 rounded-md transition-colors ${
                    strategy.status === 'active'
                      ? 'text-red-600 hover:bg-red-100 dark:text-red-400 dark:hover:bg-red-900/20'
                      : 'text-green-600 hover:bg-green-100 dark:text-green-400 dark:hover:bg-green-900/20'
                  }`}
                >
                  {strategy.status === 'active' ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <button className="p-1 text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700 rounded-md">
                  <Settings className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Métricas de Performance */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Win Rate</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {strategy.performance.winRate}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Total Trades</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {strategy.performance.totalTrades}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">P&L</p>
                <p className="text-lg font-semibold text-green-600 dark:text-green-400">
                  ${strategy.performance.pnl.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Sharpe Ratio</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {strategy.performance.sharpeRatio}
                </p>
              </div>
            </div>

            {/* Asignación de Capital */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Asignación de Capital
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {strategy.allocation}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${strategy.allocation}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Configuración de Asignación */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Configuración de Asignación
        </h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Rebalanceo Automático
            </span>
            <Switch className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600">
              <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
            </Switch>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Modo de Asignación
            </span>
            <select className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
              <option value="equal">Igual</option>
              <option value="performance">Por Performance</option>
              <option value="dynamic">Dinámico</option>
            </select>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Capital Total Asignado
            </span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              $50,000
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Strategies;