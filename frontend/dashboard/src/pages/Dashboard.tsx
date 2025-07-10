import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowUpRight,
  ArrowDownRight,
  TrendingUp,
  Users,
  Activity,
  DollarSign,
} from 'lucide-react';
import { Line, Bar } from 'recharts';
import { motion } from 'framer-motion';

import StatsCard from '../components/StatsCard';
import MarketOverview from '../components/MarketOverview';
import TradingSignals from '../components/TradingSignals';
import AgentStatus from '../components/AgentStatus';
import PortfolioChart from '../components/PortfolioChart';
import RecentTrades from '../components/RecentTrades';

const Dashboard: React.FC = () => {
  // Simular datos de estadísticas
  const stats = [
    {
      title: 'Balance Total',
      value: '$125,432.00',
      change: '+12.5%',
      trend: 'up' as const,
      icon: DollarSign,
      color: 'green',
    },
    {
      title: 'P&L Diario',
      value: '+$2,543.12',
      change: '+5.2%',
      trend: 'up' as const,
      icon: TrendingUp,
      color: 'blue',
    },
    {
      title: 'Operaciones Activas',
      value: '8',
      change: '+2',
      trend: 'up' as const,
      icon: Activity,
      color: 'purple',
    },
    {
      title: 'Win Rate',
      value: '68.5%',
      change: '+3.2%',
      trend: 'up' as const,
      icon: Users,
      color: 'orange',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard de Trading
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Resumen en tiempo real del sistema de trading cuántico
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <StatsCard {...stat} />
          </motion.div>
        ))}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Portfolio Chart - 2 columnas */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <PortfolioChart />
        </motion.div>

        {/* Agent Status - 1 columna */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <AgentStatus />
        </motion.div>

        {/* Market Overview - 2 columnas */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2"
        >
          <MarketOverview />
        </motion.div>

        {/* Trading Signals - 1 columna */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
        >
          <TradingSignals />
        </motion.div>

        {/* Recent Trades - 3 columnas */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
          className="lg:col-span-3"
        >
          <RecentTrades />
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;