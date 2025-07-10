"""
Tests para conectores de exchanges
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime

from src.exchanges.base import ExchangeBase, ExchangeError, OrderBook, Ticker, Balance, Order
from src.exchanges.binance_connector import BinanceConnector
from src.exchanges.kraken_connector import KrakenConnector
from src.exchanges.coinbase_connector import CoinbaseConnector
from src.exchanges.factory import ExchangeFactory
from src.exchanges.manager import ExchangeManager

class TestExchangeBase:
    """Tests para la clase base de exchanges"""
    
    def test_parse_decimal(self):
        """Test conversión a Decimal"""
        exchange = BinanceConnector("test_key", "test_secret")
        
        assert exchange._parse_decimal("123.45") == Decimal("123.45")
        assert exchange._parse_decimal(123.45) == Decimal("123.45")
        assert exchange._parse_decimal(Decimal("123.45")) == Decimal("123.45")
        
    def test_format_symbol(self):
        """Test formateo de símbolos"""
        exchange = BinanceConnector("test_key", "test_secret")
        
        assert exchange._format_symbol("BTC", "USDT") == "BTCUSDT"
        
    def test_generate_signature(self):
        """Test generación de firma HMAC"""
        exchange = BinanceConnector("test_key", "test_secret")
        
        signature = exchange._generate_signature("test_data")
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
        
class TestBinanceConnector:
    """Tests para el conector de Binance"""
    
    @pytest.fixture
    def binance_connector(self):
        """Fixture para el conector de Binance"""
        return BinanceConnector("test_key", "test_secret", testnet=True)
        
    @pytest.mark.asyncio
    async def test_get_ticker(self, binance_connector):
        """Test obtención de ticker"""
        mock_response = {
            'symbol': 'BTCUSDT',
            'bidPrice': '50000.00',
            'askPrice': '50100.00',
            'lastPrice': '50050.00',
            'volume': '100.50',
            'closeTime': 1634567890000
        }
        
        with patch.object(binance_connector, '_request', return_value=mock_response):
            ticker = await binance_connector.get_ticker('BTC/USDT')
            
            assert ticker.symbol == 'BTCUSDT'
            assert ticker.bid == Decimal('50000.00')
            assert ticker.ask == Decimal('50100.00')
            assert ticker.last == Decimal('50050.00')
            assert ticker.volume == Decimal('100.50')
            assert isinstance(ticker.timestamp, datetime)
            
    @pytest.mark.asyncio
    async def test_get_order_book(self, binance_connector):
        """Test obtención de libro de órdenes"""
        mock_response = {
            'bids': [['50000.00', '1.5'], ['49900.00', '2.0']],
            'asks': [['50100.00', '1.0'], ['50200.00', '1.8']]
        }
        
        with patch.object(binance_connector, '_request', return_value=mock_response):
            orderbook = await binance_connector.get_order_book('BTC/USDT')
            
            assert len(orderbook.bids) == 2
            assert len(orderbook.asks) == 2
            assert orderbook.bids[0] == (Decimal('50000.00'), Decimal('1.5'))
            assert orderbook.asks[0] == (Decimal('50100.00'), Decimal('1.0'))
            
    @pytest.mark.asyncio
    async def test_create_order(self, binance_connector):
        """Test creación de orden"""
        mock_response = {
            'orderId': 123456,
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'LIMIT',
            'origQty': '0.1',
            'price': '50000.00',
            'status': 'NEW',
            'time': 1634567890000,
            'executedQty': '0'
        }
        
        with patch.object(binance_connector, '_request', return_value=mock_response):
            order = await binance_connector.create_order(
                'BTC/USDT', 'buy', 'limit', Decimal('0.1'), Decimal('50000.00')
            )
            
            assert order.id == '123456'
            assert order.symbol == 'BTCUSDT'
            assert order.side == 'buy'
            assert order.type == 'limit'
            assert order.quantity == Decimal('0.1')
            assert order.price == Decimal('50000.00')
            assert order.status == 'open'
            
class TestKrakenConnector:
    """Tests para el conector de Kraken"""
    
    @pytest.fixture
    def kraken_connector(self):
        """Fixture para el conector de Kraken"""
        return KrakenConnector("test_key", "test_secret")
        
    def test_convert_symbol(self, kraken_connector):
        """Test conversión de símbolos para Kraken"""
        assert kraken_connector._convert_symbol('BTC/USD') == 'XBTUSD'
        assert kraken_connector._convert_symbol('ETH/USD') == 'ETHUSD'
        
    def test_convert_from_kraken_symbol(self, kraken_connector):
        """Test conversión desde símbolos de Kraken"""
        assert kraken_connector._convert_from_kraken_symbol('XBTUSD') == 'BTC/USD'
        assert kraken_connector._convert_from_kraken_symbol('ETHUSD') == 'ETH/USD'
        
    @pytest.mark.asyncio
    async def test_get_ticker(self, kraken_connector):
        """Test obtención de ticker de Kraken"""
        mock_response = {
            'error': [],
            'result': {
                'XBTUSD': {
                    'a': ['50100.00', '1', '1.000'],
                    'b': ['50000.00', '2', '2.000'],
                    'c': ['50050.00', '0.1'],
                    'v': ['100.50', '1000.50']
                }
            }
        }
        
        with patch.object(kraken_connector, '_request', return_value=mock_response):
            ticker = await kraken_connector.get_ticker('BTC/USD')
            
            assert ticker.bid == Decimal('50000.00')
            assert ticker.ask == Decimal('50100.00')
            assert ticker.last == Decimal('50050.00')
            assert ticker.volume == Decimal('1000.50')
            
class TestCoinbaseConnector:
    """Tests para el conector de Coinbase Pro"""
    
    @pytest.fixture
    def coinbase_connector(self):
        """Fixture para el conector de Coinbase"""
        return CoinbaseConnector("test_key", "test_secret", "test_passphrase", testnet=True)
        
    def test_convert_symbol(self, coinbase_connector):
        """Test conversión de símbolos para Coinbase"""
        assert coinbase_connector._convert_symbol('BTC/USD') == 'BTC-USD'
        assert coinbase_connector._convert_symbol('ETH/USD') == 'ETH-USD'
        
    def test_convert_from_cb_symbol(self, coinbase_connector):
        """Test conversión desde símbolos de Coinbase"""
        assert coinbase_connector._convert_from_cb_symbol('BTC-USD') == 'BTC/USD'
        assert coinbase_connector._convert_from_cb_symbol('ETH-USD') == 'ETH/USD'
        
    @pytest.mark.asyncio
    async def test_get_ticker(self, coinbase_connector):
        """Test obtención de ticker de Coinbase"""
        mock_response = {
            'bid': '50000.00',
            'ask': '50100.00',
            'price': '50050.00',
            'volume': '100.50',
            'time': '2021-10-18T10:31:30.000000Z'
        }
        
        with patch.object(coinbase_connector, '_request', return_value=mock_response):
            ticker = await coinbase_connector.get_ticker('BTC/USD')
            
            assert ticker.bid == Decimal('50000.00')
            assert ticker.ask == Decimal('50100.00')
            assert ticker.last == Decimal('50050.00')
            assert ticker.volume == Decimal('100.50')
            
class TestExchangeFactory:
    """Tests para el factory de exchanges"""
    
    def test_create_binance_exchange(self):
        """Test creación de exchange Binance"""
        config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        }
        
        exchange = ExchangeFactory.create_exchange('binance', config, testnet=True)
        
        assert isinstance(exchange, BinanceConnector)
        assert exchange.api_key == 'test_key'
        assert exchange.api_secret == 'test_secret'
        assert exchange.testnet is True
        
    def test_create_kraken_exchange(self):
        """Test creación de exchange Kraken"""
        config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        }
        
        exchange = ExchangeFactory.create_exchange('kraken', config)
        
        assert isinstance(exchange, KrakenConnector)
        assert exchange.api_key == 'test_key'
        assert exchange.api_secret == 'test_secret'
        
    def test_create_coinbase_exchange(self):
        """Test creación de exchange Coinbase"""
        config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'passphrase': 'test_passphrase'
        }
        
        exchange = ExchangeFactory.create_exchange('coinbase', config, testnet=True)
        
        assert isinstance(exchange, CoinbaseConnector)
        assert exchange.api_key == 'test_key'
        assert exchange.api_secret == 'test_secret'
        assert exchange.passphrase == 'test_passphrase'
        
    def test_unsupported_exchange(self):
        """Test exchange no soportado"""
        config = {'api_key': 'test', 'api_secret': 'test'}
        
        with pytest.raises(ValueError, match="Exchange 'unsupported' no soportado"):
            ExchangeFactory.create_exchange('unsupported', config)
            
    def test_missing_credentials(self):
        """Test credenciales faltantes"""
        config = {'api_key': 'test'}  # Falta api_secret
        
        with pytest.raises(KeyError, match="Credenciales faltantes"):
            ExchangeFactory.create_exchange('binance', config)
            
    def test_get_supported_exchanges(self):
        """Test obtención de exchanges soportados"""
        exchanges = ExchangeFactory.get_supported_exchanges()
        
        assert 'binance' in exchanges
        assert 'kraken' in exchanges
        assert 'coinbase' in exchanges
        
    def test_validate_config(self):
        """Test validación de configuración"""
        valid_config = {'api_key': 'test', 'api_secret': 'test'}
        invalid_config = {'api_key': 'test'}
        
        assert ExchangeFactory.validate_config('binance', valid_config) is True
        assert ExchangeFactory.validate_config('binance', invalid_config) is False
        
class TestExchangeManager:
    """Tests para el gestor de exchanges"""
    
    @pytest.fixture
    async def exchange_manager(self):
        """Fixture para el gestor de exchanges"""
        manager = ExchangeManager()
        
        # Añadir exchanges mock
        config = {'api_key': 'test', 'api_secret': 'test'}
        await manager.add_exchange('binance_test', 'binance', config, testnet=True)
        
        return manager
        
    @pytest.mark.asyncio
    async def test_add_exchange(self, exchange_manager):
        """Test añadir exchange al gestor"""
        assert 'binance_test' in exchange_manager.exchanges
        assert isinstance(exchange_manager.exchanges['binance_test'], BinanceConnector)
        assert exchange_manager.connection_status['binance_test'] is False
        
    @pytest.mark.asyncio
    async def test_get_best_price(self, exchange_manager):
        """Test obtención del mejor precio"""
        # Mock ticker para exchange
        mock_ticker = Ticker(
            symbol='BTC/USDT',
            bid=Decimal('50000'),
            ask=Decimal('50100'),
            last=Decimal('50050'),
            volume=Decimal('100'),
            timestamp=datetime.now()
        )
        
        with patch.object(exchange_manager, '_get_ticker_safe', return_value=mock_ticker):
            exchange_manager.connection_status['binance_test'] = True
            
            best_exchange, best_price = await exchange_manager.get_best_price('BTC/USDT', 'buy')
            
            assert best_exchange == 'binance_test'
            assert best_price == Decimal('50100')
            
    @pytest.mark.asyncio
    async def test_get_arbitrage_opportunities(self, exchange_manager):
        """Test búsqueda de oportunidades de arbitraje"""
        # Configurar múltiples exchanges con diferentes precios
        config = {'api_key': 'test', 'api_secret': 'test'}
        await exchange_manager.add_exchange('kraken_test', 'kraken', config)
        
        # Mock tickers con diferencia de precio
        binance_ticker = Ticker('BTC/USDT', Decimal('50000'), Decimal('50100'), 
                               Decimal('50050'), Decimal('100'), datetime.now())
        kraken_ticker = Ticker('BTC/USD', Decimal('50200'), Decimal('50300'), 
                              Decimal('50250'), Decimal('50'), datetime.now())
        
        async def mock_get_ticker_safe(exchange_name, symbol):
            if exchange_name == 'binance_test':
                return binance_ticker
            elif exchange_name == 'kraken_test':
                return kraken_ticker
            return None
            
        with patch.object(exchange_manager, '_get_ticker_safe', side_effect=mock_get_ticker_safe):
            exchange_manager.connection_status = {'binance_test': True, 'kraken_test': True}
            
            opportunities = await exchange_manager.get_arbitrage_opportunities(['BTC/USDT'], min_profit_pct=0.1)
            
            assert len(opportunities) > 0
            opportunity = opportunities[0]
            assert opportunity['symbol'] == 'BTC/USDT'
            assert opportunity['profit_pct'] > 0.1
            
    def test_get_exchange_status(self, exchange_manager):
        """Test obtención del estado de exchanges"""
        status = exchange_manager.get_exchange_status()
        
        assert 'binance_test' in status
        assert status['binance_test']['connected'] is False
        assert status['binance_test']['type'] == 'BinanceConnector'
        assert status['binance_test']['testnet'] is True

if __name__ == '__main__':
    # Ejecutar tests
    pytest.main([__file__, '-v'])