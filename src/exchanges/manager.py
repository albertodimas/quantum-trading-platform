"""
Gestor de múltiples exchanges
Coordina operaciones entre diferentes exchanges
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal
import logging

from .base import ExchangeBase, ExchangeError, OrderBook, Ticker, Balance, Order, Trade
from .factory import ExchangeFactory

logger = logging.getLogger(__name__)

class ExchangeManager:
    """Gestor central para múltiples exchanges"""
    
    def __init__(self):
        self.exchanges: Dict[str, ExchangeBase] = {}
        self.connection_status: Dict[str, bool] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def add_exchange(
        self, 
        name: str, 
        exchange_type: str, 
        config: Dict[str, Any],
        testnet: bool = False
    ):
        """
        Añadir un exchange al gestor
        
        Args:
            name: Nombre identificador del exchange
            exchange_type: Tipo de exchange ('binance', 'kraken', etc.)
            config: Configuración con credenciales
            testnet: Si usar red de pruebas
        """
        try:
            connector = ExchangeFactory.create_exchange(exchange_type, config, testnet)
            self.exchanges[name] = connector
            self.connection_status[name] = False
            
            logger.info(f"Exchange '{name}' ({exchange_type}) añadido al gestor")
            
        except Exception as e:
            logger.error(f"Error añadiendo exchange '{name}': {e}")
            raise
            
    async def connect_all(self) -> Dict[str, bool]:
        """Conectar todos los exchanges"""
        connection_results = {}
        
        tasks = []
        for name, exchange in self.exchanges.items():
            task = asyncio.create_task(self._connect_exchange(name, exchange))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, _) in enumerate(self.exchanges.items()):
            if isinstance(results[i], Exception):
                connection_results[name] = False
                logger.error(f"Error conectando {name}: {results[i]}")
            else:
                connection_results[name] = results[i]
                
        self.connection_status.update(connection_results)
        return connection_results
        
    async def disconnect_all(self):
        """Desconectar todos los exchanges"""
        # Cancelar tareas de monitoreo
        for task in self._monitoring_tasks:
            task.cancel()
        self._monitoring_tasks.clear()
        
        # Desconectar exchanges
        tasks = []
        for name, exchange in self.exchanges.items():
            task = asyncio.create_task(self._disconnect_exchange(name, exchange))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Actualizar estado
        for name in self.exchanges:
            self.connection_status[name] = False
            
    async def get_best_price(
        self, 
        symbol: str, 
        side: str,
        exchanges: Optional[List[str]] = None
    ) -> Tuple[str, Decimal]:
        """
        Obtener el mejor precio entre exchanges
        
        Args:
            symbol: Símbolo a consultar
            side: 'buy' o 'sell'
            exchanges: Lista de exchanges a consultar (None = todos)
            
        Returns:
            Tupla (exchange_name, precio)
        """
        if not exchanges:
            exchanges = list(self.exchanges.keys())
            
        best_exchange = None
        best_price = None
        
        # Obtener tickers de todos los exchanges en paralelo
        tasks = []
        for exchange_name in exchanges:
            if self.connection_status.get(exchange_name):
                task = asyncio.create_task(
                    self._get_ticker_safe(exchange_name, symbol)
                )
                tasks.append((exchange_name, task))
                
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Encontrar el mejor precio
        for i, (exchange_name, _) in enumerate(tasks):
            if isinstance(results[i], Exception):
                continue
                
            ticker = results[i]
            if not ticker:
                continue
                
            price = ticker.ask if side == 'buy' else ticker.bid
            
            if best_price is None:
                best_price = price
                best_exchange = exchange_name
            elif (side == 'buy' and price < best_price) or (side == 'sell' and price > best_price):
                best_price = price
                best_exchange = exchange_name
                
        if best_exchange is None:
            raise ExchangeError(f"No se pudo obtener precio para {symbol}")
            
        return best_exchange, best_price
        
    async def get_arbitrage_opportunities(
        self, 
        symbols: List[str],
        min_profit_pct: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Buscar oportunidades de arbitraje entre exchanges
        
        Args:
            symbols: Lista de símbolos a analizar
            min_profit_pct: Porcentaje mínimo de ganancia
            
        Returns:
            Lista de oportunidades encontradas
        """
        opportunities = []
        
        for symbol in symbols:
            # Obtener precios de todos los exchanges
            tickers = {}
            tasks = []
            
            for exchange_name in self.exchanges:
                if self.connection_status.get(exchange_name):
                    task = asyncio.create_task(
                        self._get_ticker_safe(exchange_name, symbol)
                    )
                    tasks.append((exchange_name, task))
                    
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Recopilar tickers válidos
            for i, (exchange_name, _) in enumerate(tasks):
                if not isinstance(results[i], Exception) and results[i]:
                    tickers[exchange_name] = results[i]
                    
            # Buscar oportunidades de arbitraje
            if len(tickers) >= 2:
                opportunity = self._find_arbitrage_opportunity(
                    symbol, tickers, min_profit_pct
                )
                if opportunity:
                    opportunities.append(opportunity)
                    
        return opportunities
        
    async def execute_arbitrage(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        quantity: Decimal
    ) -> Dict[str, Any]:
        """
        Ejecutar operación de arbitraje
        
        Args:
            symbol: Símbolo a tradear
            buy_exchange: Exchange donde comprar
            sell_exchange: Exchange donde vender
            quantity: Cantidad a tradear
            
        Returns:
            Resultado de la operación
        """
        if buy_exchange not in self.exchanges or sell_exchange not in self.exchanges:
            raise ExchangeError("Exchanges no válidos")
            
        if not self.connection_status.get(buy_exchange) or not self.connection_status.get(sell_exchange):
            raise ExchangeError("Exchanges no conectados")
            
        try:
            # Ejecutar órdenes en paralelo
            buy_task = asyncio.create_task(
                self.exchanges[buy_exchange].create_order(
                    symbol, 'buy', 'market', quantity
                )
            )
            
            sell_task = asyncio.create_task(
                self.exchanges[sell_exchange].create_order(
                    symbol, 'sell', 'market', quantity
                )
            )
            
            buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
            
            return {
                'success': True,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'quantity': quantity,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error ejecutando arbitraje: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def get_aggregated_orderbook(
        self, 
        symbol: str,
        exchanges: Optional[List[str]] = None
    ) -> OrderBook:
        """
        Obtener libro de órdenes agregado de múltiples exchanges
        
        Args:
            symbol: Símbolo a consultar
            exchanges: Lista de exchanges (None = todos)
            
        Returns:
            Libro de órdenes agregado
        """
        if not exchanges:
            exchanges = list(self.exchanges.keys())
            
        all_bids = []
        all_asks = []
        
        # Obtener orderbooks de todos los exchanges
        tasks = []
        for exchange_name in exchanges:
            if self.connection_status.get(exchange_name):
                task = asyncio.create_task(
                    self._get_orderbook_safe(exchange_name, symbol)
                )
                tasks.append(task)
                
        orderbooks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Agregar todos los bids y asks
        for orderbook in orderbooks:
            if isinstance(orderbook, OrderBook):
                all_bids.extend(orderbook.bids)
                all_asks.extend(orderbook.asks)
                
        # Ordenar y consolidar
        all_bids.sort(key=lambda x: x[0], reverse=True)  # Mayor precio primero
        all_asks.sort(key=lambda x: x[0])  # Menor precio primero
        
        return OrderBook(
            bids=all_bids[:50],  # Top 50 niveles
            asks=all_asks[:50],
            timestamp=datetime.now()
        )
        
    async def get_balance_summary(self) -> Dict[str, Dict[str, Balance]]:
        """Obtener resumen de balances de todos los exchanges"""
        balance_summary = {}
        
        tasks = []
        for exchange_name in self.exchanges:
            if self.connection_status.get(exchange_name):
                task = asyncio.create_task(
                    self._get_balance_safe(exchange_name)
                )
                tasks.append((exchange_name, task))
                
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for i, (exchange_name, _) in enumerate(tasks):
            if not isinstance(results[i], Exception):
                balance_summary[exchange_name] = results[i]
            else:
                balance_summary[exchange_name] = {}
                logger.error(f"Error obteniendo balance de {exchange_name}: {results[i]}")
                
        return balance_summary
        
    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estado de todos los exchanges"""
        status = {}
        
        for name, exchange in self.exchanges.items():
            status[name] = {
                'connected': self.connection_status.get(name, False),
                'type': exchange.__class__.__name__,
                'testnet': exchange.testnet,
                'last_update': datetime.now().isoformat()
            }
            
        return status
        
    # Métodos privados
    
    async def _connect_exchange(self, name: str, exchange: ExchangeBase) -> bool:
        """Conectar un exchange específico"""
        try:
            await exchange.connect()
            logger.info(f"Exchange '{name}' conectado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error conectando exchange '{name}': {e}")
            return False
            
    async def _disconnect_exchange(self, name: str, exchange: ExchangeBase):
        """Desconectar un exchange específico"""
        try:
            await exchange.disconnect()
            logger.info(f"Exchange '{name}' desconectado")
        except Exception as e:
            logger.error(f"Error desconectando exchange '{name}': {e}")
            
    async def _get_ticker_safe(self, exchange_name: str, symbol: str) -> Optional[Ticker]:
        """Obtener ticker de forma segura"""
        try:
            return await self.exchanges[exchange_name].get_ticker(symbol)
        except Exception as e:
            logger.debug(f"Error obteniendo ticker de {exchange_name}: {e}")
            return None
            
    async def _get_orderbook_safe(self, exchange_name: str, symbol: str) -> Optional[OrderBook]:
        """Obtener orderbook de forma segura"""
        try:
            return await self.exchanges[exchange_name].get_order_book(symbol)
        except Exception as e:
            logger.debug(f"Error obteniendo orderbook de {exchange_name}: {e}")
            return None
            
    async def _get_balance_safe(self, exchange_name: str) -> Dict[str, Balance]:
        """Obtener balance de forma segura"""
        try:
            return await self.exchanges[exchange_name].get_balance()
        except Exception as e:
            logger.debug(f"Error obteniendo balance de {exchange_name}: {e}")
            return {}
            
    def _find_arbitrage_opportunity(
        self, 
        symbol: str, 
        tickers: Dict[str, Ticker],
        min_profit_pct: float
    ) -> Optional[Dict[str, Any]]:
        """Buscar oportunidad de arbitraje en los tickers"""
        if len(tickers) < 2:
            return None
            
        exchanges = list(tickers.keys())
        best_opportunity = None
        max_profit_pct = 0
        
        # Comparar todos los pares de exchanges
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1, exchange2 = exchanges[i], exchanges[j]
                ticker1, ticker2 = tickers[exchange1], tickers[exchange2]
                
                # Oportunidad 1: Comprar en exchange1, vender en exchange2
                profit_pct1 = ((ticker2.bid - ticker1.ask) / ticker1.ask) * 100
                
                # Oportunidad 2: Comprar en exchange2, vender en exchange1
                profit_pct2 = ((ticker1.bid - ticker2.ask) / ticker2.ask) * 100
                
                # Seleccionar la mejor oportunidad
                if profit_pct1 > max_profit_pct and profit_pct1 >= min_profit_pct:
                    max_profit_pct = profit_pct1
                    best_opportunity = {
                        'symbol': symbol,
                        'buy_exchange': exchange1,
                        'sell_exchange': exchange2,
                        'buy_price': ticker1.ask,
                        'sell_price': ticker2.bid,
                        'profit_pct': profit_pct1,
                        'timestamp': datetime.now()
                    }
                    
                if profit_pct2 > max_profit_pct and profit_pct2 >= min_profit_pct:
                    max_profit_pct = profit_pct2
                    best_opportunity = {
                        'symbol': symbol,
                        'buy_exchange': exchange2,
                        'sell_exchange': exchange1,
                        'buy_price': ticker2.ask,
                        'sell_price': ticker1.bid,
                        'profit_pct': profit_pct2,
                        'timestamp': datetime.now()
                    }
                    
        return best_opportunity