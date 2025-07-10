import React, { useState } from 'react';
import { Plus, Minus, TrendingUp, TrendingDown } from 'lucide-react';

const Trading: React.FC = () => {
  const [selectedPair, setSelectedPair] = useState('BTC/USDT');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Terminal de Trading
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Ejecuta operaciones manualmente o con asistencia de IA
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Panel de Órdenes */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Nueva Orden
          </h3>

          {/* Selector de Par */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Par de Trading
            </label>
            <select
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="BTC/USDT">BTC/USDT</option>
              <option value="ETH/USDT">ETH/USDT</option>
              <option value="BNB/USDT">BNB/USDT</option>
              <option value="SOL/USDT">SOL/USDT</option>
            </select>
          </div>

          {/* Tabs Compra/Venta */}
          <div className="flex mb-4">
            <button
              onClick={() => setSide('buy')}
              className={`flex-1 py-2 text-sm font-medium rounded-l-md transition-colors ${
                side === 'buy'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              Comprar
            </button>
            <button
              onClick={() => setSide('sell')}
              className={`flex-1 py-2 text-sm font-medium rounded-r-md transition-colors ${
                side === 'sell'
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              Vender
            </button>
          </div>

          {/* Tipo de Orden */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Tipo de Orden
            </label>
            <div className="flex space-x-2">
              <button
                onClick={() => setOrderType('market')}
                className={`flex-1 py-2 px-3 text-sm rounded-md transition-colors ${
                  orderType === 'market'
                    ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400 border border-primary-300 dark:border-primary-700'
                    : 'bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                }`}
              >
                Mercado
              </button>
              <button
                onClick={() => setOrderType('limit')}
                className={`flex-1 py-2 px-3 text-sm rounded-md transition-colors ${
                  orderType === 'limit'
                    ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400 border border-primary-300 dark:border-primary-700'
                    : 'bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                }`}
              >
                Límite
              </button>
            </div>
          </div>

          {/* Precio (solo para orden límite) */}
          {orderType === 'limit' && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Precio
              </label>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="0.00"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>
          )}

          {/* Cantidad */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Cantidad
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Botón de Ejecución */}
          <button
            className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors ${
              side === 'buy'
                ? 'bg-green-500 hover:bg-green-600'
                : 'bg-red-500 hover:bg-red-600'
            }`}
          >
            {side === 'buy' ? 'Comprar' : 'Vender'} {selectedPair.split('/')[0]}
          </button>

          {/* Recomendación IA */}
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
            <p className="text-sm text-blue-800 dark:text-blue-400">
              <strong>Recomendación IA:</strong> Momento neutral. Esperar confirmación de tendencia.
            </p>
          </div>
        </div>

        {/* Libro de Órdenes */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Libro de Órdenes
          </h3>

          {/* Órdenes de Venta */}
          <div className="mb-4">
            <div className="space-y-1">
              {[...Array(5)].map((_, i) => (
                <div key={`sell-${i}`} className="flex justify-between text-sm">
                  <span className="text-red-600 dark:text-red-400">
                    {(67500 + (i + 1) * 10).toFixed(2)}
                  </span>
                  <span className="text-gray-600 dark:text-gray-400">
                    {(Math.random() * 5).toFixed(4)}
                  </span>
                  <span className="text-gray-900 dark:text-white">
                    ${((67500 + (i + 1) * 10) * (Math.random() * 5)).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Precio Actual */}
          <div className="py-2 border-y border-gray-200 dark:border-gray-700 mb-4">
            <div className="text-center">
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                67,500.00
              </span>
              <span className="ml-2 text-sm text-green-600 dark:text-green-400">
                +2.34%
              </span>
            </div>
          </div>

          {/* Órdenes de Compra */}
          <div className="space-y-1">
            {[...Array(5)].map((_, i) => (
              <div key={`buy-${i}`} className="flex justify-between text-sm">
                <span className="text-green-600 dark:text-green-400">
                  {(67500 - (i + 1) * 10).toFixed(2)}
                </span>
                <span className="text-gray-600 dark:text-gray-400">
                  {(Math.random() * 5).toFixed(4)}
                </span>
                <span className="text-gray-900 dark:text-white">
                  ${((67500 - (i + 1) * 10) * (Math.random() * 5)).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Posiciones Abiertas */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Posiciones Abiertas
          </h3>

          <div className="space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    BTC/USDT
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Long • 0.15 BTC
                  </p>
                </div>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  +3.85%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Entrada:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$65,000</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Actual:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$67,500</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">P&L:</span>
                  <span className="ml-1 text-green-600 dark:text-green-400">+$375</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Stop:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$64,000</span>
                </div>
              </div>
              <div className="mt-3 flex space-x-2">
                <button className="flex-1 py-1 px-2 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded">
                  Modificar
                </button>
                <button className="flex-1 py-1 px-2 text-xs bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded">
                  Cerrar
                </button>
              </div>
            </div>

            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    ETH/USDT
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Short • 2.5 ETH
                  </p>
                </div>
                <span className="text-sm font-medium text-red-600 dark:text-red-400">
                  -1.23%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Entrada:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$3,500</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Actual:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$3,456</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">P&L:</span>
                  <span className="ml-1 text-red-600 dark:text-red-400">-$110</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Stop:</span>
                  <span className="ml-1 text-gray-900 dark:text-white">$3,600</span>
                </div>
              </div>
              <div className="mt-3 flex space-x-2">
                <button className="flex-1 py-1 px-2 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded">
                  Modificar
                </button>
                <button className="flex-1 py-1 px-2 text-xs bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded">
                  Cerrar
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Trading;