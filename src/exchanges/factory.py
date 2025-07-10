"""
Factory para crear conectores de exchanges
Centraliza la creación y configuración de conectores
"""

from typing import Dict, Any, Optional
import logging

from .base import ExchangeBase
from .binance_connector import BinanceConnector
from .kraken_connector import KrakenConnector
from .coinbase_connector import CoinbaseConnector

logger = logging.getLogger(__name__)

class ExchangeFactory:
    """Factory para crear instancias de conectores de exchanges"""
    
    _connectors = {
        'binance': BinanceConnector,
        'kraken': KrakenConnector,
        'coinbase': CoinbaseConnector,
    }
    
    @classmethod
    def create_exchange(
        self,
        exchange_name: str,
        config: Dict[str, Any],
        testnet: bool = False
    ) -> ExchangeBase:
        """
        Crear instancia de conector de exchange
        
        Args:
            exchange_name: Nombre del exchange ('binance', 'kraken', 'coinbase')
            config: Configuración con credenciales
            testnet: Si usar red de pruebas
            
        Returns:
            Instancia del conector
            
        Raises:
            ValueError: Si el exchange no está soportado
            KeyError: Si faltan credenciales requeridas
        """
        exchange_name = exchange_name.lower()
        
        if exchange_name not in self._connectors:
            available = ', '.join(self._connectors.keys())
            raise ValueError(f"Exchange '{exchange_name}' no soportado. Disponibles: {available}")
            
        connector_class = self._connectors[exchange_name]
        
        # Validar credenciales requeridas
        required_keys = self._get_required_keys(exchange_name)
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise KeyError(f"Credenciales faltantes para {exchange_name}: {missing_keys}")
            
        # Crear instancia según el exchange
        if exchange_name == 'binance':
            return connector_class(
                api_key=config['api_key'],
                api_secret=config['api_secret'],
                testnet=testnet
            )
            
        elif exchange_name == 'kraken':
            return connector_class(
                api_key=config['api_key'],
                api_secret=config['api_secret'],
                testnet=testnet
            )
            
        elif exchange_name == 'coinbase':
            return connector_class(
                api_key=config['api_key'],
                api_secret=config['api_secret'],
                passphrase=config['passphrase'],
                testnet=testnet
            )
            
    @classmethod
    def get_supported_exchanges(cls) -> list:
        """Obtener lista de exchanges soportados"""
        return list(cls._connectors.keys())
        
    @classmethod
    def _get_required_keys(cls, exchange_name: str) -> list:
        """Obtener claves de configuración requeridas para un exchange"""
        required_keys_map = {
            'binance': ['api_key', 'api_secret'],
            'kraken': ['api_key', 'api_secret'],
            'coinbase': ['api_key', 'api_secret', 'passphrase'],
        }
        return required_keys_map.get(exchange_name, ['api_key', 'api_secret'])
        
    @classmethod
    def validate_config(cls, exchange_name: str, config: Dict[str, Any]) -> bool:
        """
        Validar configuración para un exchange
        
        Args:
            exchange_name: Nombre del exchange
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            required_keys = cls._get_required_keys(exchange_name.lower())
            return all(key in config and config[key] for key in required_keys)
        except:
            return False
            
    @classmethod
    def create_multiple_exchanges(
        cls,
        exchanges_config: Dict[str, Dict[str, Any]],
        testnet: bool = False
    ) -> Dict[str, ExchangeBase]:
        """
        Crear múltiples conectores de exchanges
        
        Args:
            exchanges_config: Configuración para múltiples exchanges
                {
                    'binance': {'api_key': '...', 'api_secret': '...'},
                    'kraken': {'api_key': '...', 'api_secret': '...'},
                    ...
                }
            testnet: Si usar redes de prueba
            
        Returns:
            Diccionario con instancias de conectores
        """
        connectors = {}
        
        for exchange_name, config in exchanges_config.items():
            try:
                connector = cls.create_exchange(exchange_name, config, testnet)
                connectors[exchange_name] = connector
                logger.info(f"Conector de {exchange_name} creado exitosamente")
                
            except Exception as e:
                logger.error(f"Error creando conector de {exchange_name}: {e}")
                
        return connectors
        
    @classmethod
    async def connect_all(cls, connectors: Dict[str, ExchangeBase]) -> Dict[str, bool]:
        """
        Conectar todos los conectores
        
        Args:
            connectors: Diccionario de conectores
            
        Returns:
            Diccionario con estado de conexión para cada exchange
        """
        connection_status = {}
        
        for exchange_name, connector in connectors.items():
            try:
                await connector.connect()
                connection_status[exchange_name] = True
                logger.info(f"Conectado a {exchange_name}")
                
            except Exception as e:
                connection_status[exchange_name] = False
                logger.error(f"Error conectando a {exchange_name}: {e}")
                
        return connection_status
        
    @classmethod
    async def disconnect_all(cls, connectors: Dict[str, ExchangeBase]):
        """
        Desconectar todos los conectores
        
        Args:
            connectors: Diccionario de conectores
        """
        for exchange_name, connector in connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Desconectado de {exchange_name}")
                
            except Exception as e:
                logger.error(f"Error desconectando de {exchange_name}: {e}")
                
    @classmethod
    def get_exchange_info(cls, exchange_name: str) -> Dict[str, Any]:
        """
        Obtener información sobre un exchange
        
        Args:
            exchange_name: Nombre del exchange
            
        Returns:
            Información del exchange
        """
        exchange_info = {
            'binance': {
                'name': 'Binance',
                'website': 'https://www.binance.com',
                'api_docs': 'https://binance-docs.github.io/apidocs/',
                'fee_structure': 'Maker/Taker fees',
                'supports_testnet': True,
                'websocket_streams': ['ticker', 'orderbook', 'trades', 'user_data'],
                'rate_limits': 'Weight-based system',
                'base_currencies': ['USDT', 'BTC', 'ETH', 'BNB'],
            },
            'kraken': {
                'name': 'Kraken',
                'website': 'https://www.kraken.com',
                'api_docs': 'https://docs.kraken.com/rest/',
                'fee_structure': 'Volume-based fees',
                'supports_testnet': False,
                'websocket_streams': ['ticker', 'orderbook', 'trades', 'own_trades'],
                'rate_limits': 'Call rate limiter',
                'base_currencies': ['USD', 'EUR', 'BTC', 'ETH'],
            },
            'coinbase': {
                'name': 'Coinbase Pro',
                'website': 'https://pro.coinbase.com',
                'api_docs': 'https://docs.pro.coinbase.com/',
                'fee_structure': 'Maker/Taker fees based on volume',
                'supports_testnet': True,
                'websocket_streams': ['ticker', 'level2', 'matches', 'user'],
                'rate_limits': 'Request rate limiter',
                'base_currencies': ['USD', 'EUR', 'GBP', 'BTC'],
            }
        }
        
        return exchange_info.get(exchange_name.lower(), {})
        
    @classmethod
    def compare_exchanges(cls, metric: str = 'fees') -> Dict[str, Any]:
        """
        Comparar exchanges por diferentes métricas
        
        Args:
            metric: Métrica a comparar ('fees', 'volume', 'pairs')
            
        Returns:
            Comparación de exchanges
        """
        comparisons = {
            'fees': {
                'binance': 'Fees bajas, descuentos con BNB',
                'kraken': 'Fees competitivas basadas en volumen',
                'coinbase': 'Fees más altas pero gran liquidez'
            },
            'volume': {
                'binance': 'Volumen más alto globalmente',
                'kraken': 'Volumen sólido, especialmente en fiat',
                'coinbase': 'Volumen alto en mercados USD'
            },
            'pairs': {
                'binance': '1000+ pares de trading',
                'kraken': '200+ pares de trading',
                'coinbase': '200+ pares de trading'
            },
            'security': {
                'binance': 'SAFU fund, auditorías regulares',
                'kraken': 'Nunca hackeado, alta seguridad',
                'coinbase': 'Regulado en US, seguros FDIC'
            }
        }
        
        return comparisons.get(metric, {})