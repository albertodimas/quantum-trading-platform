"""
Motor de Backtesting Principal
Orquesta la simulación histórica de estrategias de trading
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import pandas as pd
import numpy as np

from .data_handler import HistoricalDataHandler
from .portfolio import Portfolio
from .metrics import PerformanceMetrics
from .reports import ReportGenerator

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Motor principal de backtesting"""
    
    def __init__(
        self,
        initial_capital: Decimal = Decimal('100000'),
        commission: Decimal = Decimal('0.001'),
        slippage: Decimal = Decimal('0.0005'),
        timezone: str = 'UTC'
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.timezone = timezone
        
        # Componentes principales
        self.data_handler = HistoricalDataHandler()
        self.portfolio = Portfolio(initial_capital, commission)
        self.performance_metrics = PerformanceMetrics()
        self.report_generator = ReportGenerator()
        
        # Estado del backtest
        self.current_time = None
        self.strategy = None
        self.results = {}
        self.trades_log = []
        self.equity_curve = []
        
    async def run_backtest(
        self,
        strategy,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1h',
        benchmark: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar backtesting completo
        
        Args:
            strategy: Instancia de estrategia a probar
            symbols: Lista de símbolos a tradear
            start_date: Fecha inicio del backtest
            end_date: Fecha fin del backtest
            timeframe: Marco temporal ('1m', '5m', '1h', '1d')
            benchmark: Símbolo de referencia para comparar
            
        Returns:
            Resultados completos del backtest
        """
        logger.info(f"Iniciando backtest: {start_date} - {end_date}")
        logger.info(f"Símbolos: {symbols}, Timeframe: {timeframe}")
        
        try:
            # Inicializar componentes
            self.strategy = strategy
            self.portfolio.reset(self.initial_capital)
            self.trades_log.clear()
            self.equity_curve.clear()
            
            # Cargar datos históricos
            logger.info("Cargando datos históricos...")
            historical_data = await self._load_historical_data(
                symbols, start_date, end_date, timeframe
            )
            
            # Cargar datos de benchmark si se especifica
            benchmark_data = None
            if benchmark:
                benchmark_data = await self._load_historical_data(
                    [benchmark], start_date, end_date, timeframe
                )
                
            # Ejecutar simulación
            logger.info("Ejecutando simulación...")
            await self._run_simulation(historical_data, start_date, end_date)
            
            # Calcular métricas de rendimiento
            logger.info("Calculando métricas de rendimiento...")
            metrics = self._calculate_performance_metrics(benchmark_data)
            
            # Generar resultados finales
            results = {
                'summary': self._generate_summary(),
                'metrics': metrics,
                'trades': self.trades_log.copy(),
                'equity_curve': self.equity_curve.copy(),
                'portfolio_state': self.portfolio.get_state(),
                'parameters': {
                    'initial_capital': float(self.initial_capital),
                    'commission': float(self.commission),
                    'slippage': float(self.slippage),
                    'symbols': symbols,
                    'timeframe': timeframe,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
            
            self.results = results
            logger.info("Backtest completado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error en backtest: {e}")
            raise
            
    async def _load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Cargar datos históricos para los símbolos"""
        data = {}
        
        for symbol in symbols:
            try:
                df = await self.data_handler.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                
                if df.empty:
                    logger.warning(f"No hay datos para {symbol}")
                    continue
                    
                # Validar y limpiar datos
                df = self._validate_and_clean_data(df, symbol)
                data[symbol] = df
                
                logger.info(f"Cargados {len(df)} registros para {symbol}")
                
            except Exception as e:
                logger.error(f"Error cargando datos para {symbol}: {e}")
                
        return data
        
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validar y limpiar datos históricos"""
        # Verificar columnas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en {symbol}: {missing_columns}")
            
        # Eliminar valores nulos
        df = df.dropna()
        
        # Verificar que high >= low, close entre high/low, etc.
        df = df[df['high'] >= df['low']]
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        
        # Eliminar outliers extremos (variaciones > 50% en un periodo)
        price_change = df['close'].pct_change().abs()
        df = df[price_change <= 0.5]
        
        # Asegurar que el volumen sea positivo
        df = df[df['volume'] > 0]
        
        return df.sort_index()
        
    async def _run_simulation(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ):
        """Ejecutar la simulación del backtest"""
        if not historical_data:
            raise ValueError("No hay datos históricos disponibles")
            
        # Crear timeline unificado de todos los símbolos
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)
            
        timeline = sorted(all_timestamps)
        timeline = [ts for ts in timeline if start_date <= ts <= end_date]
        
        logger.info(f"Simulando {len(timeline)} períodos de tiempo")
        
        # Iterar por cada punto temporal
        for i, timestamp in enumerate(timeline):
            self.current_time = timestamp
            
            # Construir datos de mercado actuales
            market_data = self._build_market_data(historical_data, timestamp)
            
            if not market_data:
                continue
                
            # Actualizar portfolio con precios actuales
            self.portfolio.update_prices(market_data, timestamp)
            
            # Ejecutar lógica de estrategia
            signals = await self._execute_strategy(market_data, timestamp)
            
            # Procesar señales de trading
            await self._process_signals(signals, market_data, timestamp)
            
            # Registrar estado del portfolio
            self._record_portfolio_state(timestamp)
            
            # Progreso cada 1000 períodos
            if i % 1000 == 0:
                progress = (i / len(timeline)) * 100
                logger.debug(f"Progreso: {progress:.1f}%")
                
    def _build_market_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> Dict[str, Dict[str, float]]:
        """Construir datos de mercado para un timestamp específico"""
        market_data = {}
        
        for symbol, df in historical_data.items():
            if timestamp in df.index:
                row = df.loc[timestamp]
                market_data[symbol] = {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'timestamp': timestamp
                }
                
        return market_data
        
    async def _execute_strategy(
        self,
        market_data: Dict[str, Dict[str, float]],
        timestamp: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """Ejecutar la lógica de la estrategia"""
        try:
            # La estrategia debe tener un método generate_signals
            if hasattr(self.strategy, 'generate_signals'):\n                signals = await self.strategy.generate_signals(market_data, timestamp)\n                return signals if signals else []\n            else:\n                logger.warning(\"La estrategia no tiene método generate_signals\")\n                return []\n                \n        except Exception as e:\n            logger.error(f\"Error ejecutando estrategia en {timestamp}: {e}\")\n            return []\n            \n    async def _process_signals(\n        self,\n        signals: List[Dict[str, Any]],\n        market_data: Dict[str, Dict[str, float]],\n        timestamp: pd.Timestamp\n    ):\n        \"\"\"Procesar señales de trading\"\"\"        \n        for signal in signals:\n            try:\n                await self._execute_trade(signal, market_data, timestamp)\n            except Exception as e:\n                logger.error(f\"Error procesando señal {signal}: {e}\")\n                \n    async def _execute_trade(\n        self,\n        signal: Dict[str, Any],\n        market_data: Dict[str, Dict[str, float]],\n        timestamp: pd.Timestamp\n    ):\n        \"\"\"Ejecutar una operación de trading\"\"\"        \n        symbol = signal.get('symbol')\n        action = signal.get('action')  # 'buy', 'sell', 'close'\n        quantity = signal.get('quantity', 0)\n        order_type = signal.get('type', 'market')\n        price = signal.get('price')\n        \n        if symbol not in market_data:\n            logger.warning(f\"No hay datos de mercado para {symbol}\")\n            return\n            \n        current_price = market_data[symbol]['close']\n        \n        # Determinar precio de ejecución\n        if order_type == 'market':\n            execution_price = self._apply_slippage(current_price, action)\n        else:\n            execution_price = price or current_price\n            \n        # Validar que tenemos fondos/posición suficiente\n        if action == 'buy':\n            cost = quantity * execution_price * (1 + self.commission)\n            if self.portfolio.cash < cost:\n                logger.debug(f\"Fondos insuficientes para comprar {quantity} {symbol}\")\n                return\n        elif action == 'sell':\n            current_position = self.portfolio.positions.get(symbol, 0)\n            if current_position < quantity:\n                logger.debug(f\"Posición insuficiente para vender {quantity} {symbol}\")\n                return\n                \n        # Ejecutar la operación\n        trade_result = self.portfolio.execute_trade(\n            symbol, action, quantity, execution_price, timestamp\n        )\n        \n        if trade_result:\n            # Registrar el trade\n            trade_log = {\n                'timestamp': timestamp,\n                'symbol': symbol,\n                'action': action,\n                'quantity': quantity,\n                'price': execution_price,\n                'commission': trade_result.get('commission', 0),\n                'portfolio_value': self.portfolio.total_value,\n                'cash': self.portfolio.cash\n            }\n            \n            self.trades_log.append(trade_log)\n            logger.debug(f\"Trade ejecutado: {action} {quantity} {symbol} @ {execution_price}\")\n            \n    def _apply_slippage(self, price: float, action: str) -> float:\n        \"\"\"Aplicar slippage al precio de ejecución\"\"\"        \n        slippage_factor = float(self.slippage)\n        \n        if action == 'buy':\n            # Slippage negativo para compras (precio más alto)\n            return price * (1 + slippage_factor)\n        else:\n            # Slippage positivo para ventas (precio más bajo)\n            return price * (1 - slippage_factor)\n            \n    def _record_portfolio_state(self, timestamp: pd.Timestamp):\n        \"\"\"Registrar el estado actual del portfolio\"\"\"        \n        equity_point = {\n            'timestamp': timestamp,\n            'total_value': float(self.portfolio.total_value),\n            'cash': float(self.portfolio.cash),\n            'positions_value': float(self.portfolio.positions_value),\n            'unrealized_pnl': float(self.portfolio.unrealized_pnl),\n            'realized_pnl': float(self.portfolio.realized_pnl)\n        }\n        \n        self.equity_curve.append(equity_point)\n        \n    def _calculate_performance_metrics(self, benchmark_data: Optional[Dict] = None) -> Dict[str, Any]:\n        \"\"\"Calcular métricas de rendimiento\"\"\"        \n        if not self.equity_curve:\n            return {}\n            \n        # Crear DataFrame de equity curve\n        df = pd.DataFrame(self.equity_curve)\n        df.set_index('timestamp', inplace=True)\n        \n        # Calcular returns\n        df['returns'] = df['total_value'].pct_change()\n        \n        # Métricas básicas\n        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1\n        num_trades = len(self.trades_log)\n        \n        # Calcular métricas usando PerformanceMetrics\n        metrics = self.performance_metrics.calculate_metrics(\n            returns=df['returns'].dropna(),\n            equity_curve=df['total_value'],\n            trades=self.trades_log,\n            initial_capital=float(self.initial_capital)\n        )\n        \n        return metrics\n        \n    def _generate_summary(self) -> Dict[str, Any]:\n        \"\"\"Generar resumen del backtest\"\"\"        \n        if not self.equity_curve:\n            return {}\n            \n        initial_value = self.equity_curve[0]['total_value']\n        final_value = self.equity_curve[-1]['total_value']\n        total_return = (final_value / initial_value) - 1\n        \n        winning_trades = [t for t in self.trades_log if t.get('pnl', 0) > 0]\n        losing_trades = [t for t in self.trades_log if t.get('pnl', 0) < 0]\n        \n        summary = {\n            'initial_capital': initial_value,\n            'final_capital': final_value,\n            'total_return': total_return,\n            'total_return_pct': total_return * 100,\n            'total_trades': len(self.trades_log),\n            'winning_trades': len(winning_trades),\n            'losing_trades': len(losing_trades),\n            'win_rate': len(winning_trades) / len(self.trades_log) if self.trades_log else 0,\n            'duration_days': (self.equity_curve[-1]['timestamp'] - self.equity_curve[0]['timestamp']).days,\n            'avg_daily_return': None,  # Se calculará en métricas\n            'max_drawdown': None,      # Se calculará en métricas\n            'sharpe_ratio': None       # Se calculará en métricas\n        }\n        \n        return summary\n        \n    def generate_report(self, output_path: Optional[str] = None) -> str:\n        \"\"\"Generar reporte completo del backtest\"\"\"        \n        if not self.results:\n            raise ValueError(\"No hay resultados de backtest para reportar\")\n            \n        return self.report_generator.generate_report(\n            self.results, output_path\n        )\n        \n    def save_results(self, filepath: str):\n        \"\"\"Guardar resultados en archivo\"\"\"        \n        import json\n        \n        # Convertir timestamps para serialización JSON\n        results_copy = self.results.copy()\n        \n        # Convertir equity curve\n        for point in results_copy['equity_curve']:\n            point['timestamp'] = point['timestamp'].isoformat()\n            \n        # Convertir trades log\n        for trade in results_copy['trades']:\n            trade['timestamp'] = trade['timestamp'].isoformat()\n            \n        with open(filepath, 'w') as f:\n            json.dump(results_copy, f, indent=2, default=str)\n            \n        logger.info(f\"Resultados guardados en {filepath}\")\n        \n    def load_results(self, filepath: str):\n        \"\"\"Cargar resultados desde archivo\"\"\"        \n        import json\n        \n        with open(filepath, 'r') as f:\n            self.results = json.load(f)\n            \n        # Convertir timestamps de vuelta\n        for point in self.results['equity_curve']:\n            point['timestamp'] = pd.to_datetime(point['timestamp'])\n            \n        for trade in self.results['trades']:\n            trade['timestamp'] = pd.to_datetime(trade['timestamp'])\n            \n        logger.info(f\"Resultados cargados desde {filepath}\")"