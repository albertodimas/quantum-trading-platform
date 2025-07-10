import React from 'react';
import { Brain, TrendingUp, AlertTriangle, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'idle' | 'error';
  lastAction: string;
  confidence: number;
  icon: React.ReactNode;
}

const AgentStatus: React.FC = () => {
  const agents: Agent[] = [
    {
      id: '1',
      name: 'Análisis Técnico',
      type: 'technical',
      status: 'active',
      lastAction: 'Analizando BTC/USDT',
      confidence: 0.85,
      icon: <TrendingUp className="h-5 w-5" />,
    },
    {
      id: '2',
      name: 'Sentimiento',
      type: 'sentiment',
      status: 'active',
      lastAction: 'Procesando noticias',
      confidence: 0.72,
      icon: <Brain className="h-5 w-5" />,
    },
    {
      id: '3',
      name: 'Gestión de Riesgo',
      type: 'risk',
      status: 'idle',
      lastAction: 'Última revisión hace 5m',
      confidence: 0.90,
      icon: <Shield className="h-5 w-5" />,
    },
    {
      id: '4',
      name: 'Alertas',
      type: 'alerts',
      status: 'active',
      lastAction: 'Monitoreando mercado',
      confidence: 0.95,
      icon: <AlertTriangle className="h-5 w-5" />,
    },
  ];

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400';
      case 'idle':
        return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400';
      case 'error':
        return 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400';
      default:
        return 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-400';
    }
  };

  const getStatusText = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return 'Activo';
      case 'idle':
        return 'En espera';
      case 'error':
        return 'Error';
      default:
        return 'Desconocido';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Estado de Agentes IA
      </h3>

      <div className="space-y-4">
        {agents.map((agent, index) => (
          <motion.div
            key={agent.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50"
          >
            <div className="flex items-center space-x-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400">
                {agent.icon}
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {agent.name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {agent.lastAction}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="text-right">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Confianza
                </p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {(agent.confidence * 100).toFixed(0)}%
                </p>
              </div>
              <span
                className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                  agent.status
                )}`}
              >
                {getStatusText(agent.status)}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Resumen del Orquestador */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-white">
              Orquestador Principal
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Consenso del sistema: Alcista moderado
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            <span className="text-sm text-green-600 dark:text-green-400">
              Operativo
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentStatus;