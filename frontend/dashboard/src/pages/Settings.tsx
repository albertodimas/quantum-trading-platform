import React, { useState } from 'react';
import { Save, Key, Bell, Shield, Database, Webhook, AlertCircle } from 'lucide-react';
import { Switch } from '@radix-ui/react-switch';

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'general' | 'api' | 'notifications' | 'security'>('general');
  const [settings, setSettings] = useState({
    // General
    language: 'es',
    timezone: 'America/Mexico_City',
    currency: 'USD',
    theme: 'dark',
    
    // Trading
    defaultLeverage: 1,
    maxPositionSize: 10000,
    stopLossDefault: 2,
    takeProfitDefault: 5,
    
    // Notificaciones
    emailNotifications: true,
    tradingAlerts: true,
    systemAlerts: true,
    priceAlerts: false,
    
    // API
    binanceApiKey: '',
    binanceSecret: '',
    testMode: true,
  });

  const handleSave = () => {
    // Aquí se guardarían las configuraciones
    console.log('Guardando configuraciones:', settings);
  };

  const tabs = [
    { id: 'general', label: 'General', icon: <Database className="h-4 w-4" /> },
    { id: 'api', label: 'API / Exchanges', icon: <Key className="h-4 w-4" /> },
    { id: 'notifications', label: 'Notificaciones', icon: <Bell className="h-4 w-4" /> },
    { id: 'security', label: 'Seguridad', icon: <Shield className="h-4 w-4" /> },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Configuración
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Personaliza tu sistema de trading
          </p>
        </div>
        <button
          onClick={handleSave}
          className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors flex items-center"
        >
          <Save className="h-4 w-4 mr-2" />
          Guardar Cambios
        </button>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar de navegación */}
        <div className="w-full lg:w-64">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeTab === tab.id
                    ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {tab.icon}
                <span className="ml-3">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Contenido principal */}
        <div className="flex-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            {activeTab === 'general' && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Configuración General
                </h3>

                {/* Preferencias de Usuario */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Idioma
                    </label>
                    <select
                      value={settings.language}
                      onChange={(e) => setSettings({ ...settings, language: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="es">Español</option>
                      <option value="en">English</option>
                      <option value="pt">Português</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Zona Horaria
                    </label>
                    <select
                      value={settings.timezone}
                      onChange={(e) => setSettings({ ...settings, timezone: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="America/Mexico_City">Ciudad de México (GMT-6)</option>
                      <option value="America/New_York">Nueva York (GMT-5)</option>
                      <option value="Europe/London">Londres (GMT+0)</option>
                      <option value="Asia/Tokyo">Tokio (GMT+9)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Moneda Base
                    </label>
                    <select
                      value={settings.currency}
                      onChange={(e) => setSettings({ ...settings, currency: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      <option value="USD">USD - Dólar Estadounidense</option>
                      <option value="EUR">EUR - Euro</option>
                      <option value="MXN">MXN - Peso Mexicano</option>
                      <option value="BTC">BTC - Bitcoin</option>
                    </select>
                  </div>
                </div>

                {/* Configuración de Trading */}
                <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-4">
                    Configuración de Trading
                  </h4>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Apalancamiento Predeterminado
                      </label>
                      <input
                        type="number"
                        value={settings.defaultLeverage}
                        onChange={(e) => setSettings({ ...settings, defaultLeverage: parseInt(e.target.value) })}
                        min="1"
                        max="100"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Tamaño Máximo de Posición ($)
                      </label>
                      <input
                        type="number"
                        value={settings.maxPositionSize}
                        onChange={(e) => setSettings({ ...settings, maxPositionSize: parseInt(e.target.value) })}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Stop Loss Predeterminado (%)
                        </label>
                        <input
                          type="number"
                          value={settings.stopLossDefault}
                          onChange={(e) => setSettings({ ...settings, stopLossDefault: parseFloat(e.target.value) })}
                          step="0.1"
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Take Profit Predeterminado (%)
                        </label>
                        <input
                          type="number"
                          value={settings.takeProfitDefault}
                          onChange={(e) => setSettings({ ...settings, takeProfitDefault: parseFloat(e.target.value) })}
                          step="0.1"
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'api' && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Configuración de API y Exchanges
                </h3>

                <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-4 mb-6">
                  <div className="flex">
                    <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5" />
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-400">
                        Modo de Prueba Activado
                      </h3>
                      <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-500">
                        Las operaciones se ejecutarán en el entorno de prueba. Desactiva esta opción para operar con fondos reales.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Modo de Prueba
                    </span>
                    <Switch
                      checked={settings.testMode}
                      onCheckedChange={(checked) => setSettings({ ...settings, testMode: checked })}
                      className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600"
                    >
                      <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                    </Switch>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Binance API Key
                    </label>
                    <input
                      type="password"
                      value={settings.binanceApiKey}
                      onChange={(e) => setSettings({ ...settings, binanceApiKey: e.target.value })}
                      placeholder="Ingresa tu API Key de Binance"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Binance Secret Key
                    </label>
                    <input
                      type="password"
                      value={settings.binanceSecret}
                      onChange={(e) => setSettings({ ...settings, binanceSecret: e.target.value })}
                      placeholder="Ingresa tu Secret Key de Binance"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    />
                  </div>

                  <div className="pt-4">
                    <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                      Probar Conexión
                    </button>
                  </div>
                </div>

                {/* Webhooks */}
                <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-4 flex items-center">
                    <Webhook className="h-4 w-4 mr-2" />
                    Webhooks
                  </h4>
                  
                  <div className="space-y-3">
                    <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-md">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        URL de Webhook para señales:
                      </p>
                      <code className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded mt-1 block">
                        https://api.quantum-trading.com/webhook/signals/abc123
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Configuración de Notificaciones
                </h3>

                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        Notificaciones por Email
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Recibe alertas importantes en tu correo
                      </p>
                    </div>
                    <Switch
                      checked={settings.emailNotifications}
                      onCheckedChange={(checked) => setSettings({ ...settings, emailNotifications: checked })}
                      className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600"
                    >
                      <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                    </Switch>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        Alertas de Trading
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Notificaciones cuando se ejecutan operaciones
                      </p>
                    </div>
                    <Switch
                      checked={settings.tradingAlerts}
                      onCheckedChange={(checked) => setSettings({ ...settings, tradingAlerts: checked })}
                      className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600"
                    >
                      <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                    </Switch>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        Alertas del Sistema
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Errores, mantenimiento y actualizaciones
                      </p>
                    </div>
                    <Switch
                      checked={settings.systemAlerts}
                      onCheckedChange={(checked) => setSettings({ ...settings, systemAlerts: checked })}
                      className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600"
                    >
                      <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                    </Switch>
                  </div>

                  <div className="flex items-center justify-between py-3">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        Alertas de Precio
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Notificaciones cuando el precio alcanza ciertos niveles
                      </p>
                    </div>
                    <Switch
                      checked={settings.priceAlerts}
                      onCheckedChange={(checked) => setSettings({ ...settings, priceAlerts: checked })}
                      className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600"
                    >
                      <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                    </Switch>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Configuración de Seguridad
                </h3>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Autenticación de Dos Factores (2FA)
                    </h4>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
                      <p className="text-sm text-green-800 dark:text-green-400">
                        La autenticación de dos factores está activada
                      </p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Sesiones Activas
                    </h4>
                    <div className="space-y-2">
                      <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-md flex justify-between items-center">
                        <div>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            Chrome - Windows
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            192.168.1.100 • Hace 2 minutos
                          </p>
                        </div>
                        <span className="text-xs bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 px-2 py-1 rounded">
                          Actual
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Cambiar Contraseña
                    </h4>
                    <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                      Cambiar Contraseña
                    </button>
                  </div>

                  <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Configuración de IP
                    </h4>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          Restringir acceso por IP
                        </span>
                        <Switch className="w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full relative cursor-pointer transition-colors data-[state=checked]:bg-primary-600">
                          <span className="block w-5 h-5 bg-white rounded-full shadow-sm transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0.5" />
                        </Switch>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;