# Módulo de Conectores de Exchanges

## 🌐 Descripción

Este módulo proporciona conectores unificados para múltiples exchanges de criptomonedas, permitiendo interactuar con diferentes plataformas de trading a través de una interfaz común.

## 🏪 Exchanges Soportados

### 1. Binance
- **API REST**: ✅ Completa
- **WebSocket**: ✅ Tiempo real
- **Testnet**: ✅ Disponible
- **Características**: Mayor volumen global, fees bajas, amplia variedad de pares
- **Documentación**: [Binance API](https://binance-docs.github.io/apidocs/)

### 2. Kraken
- **API REST**: ✅ Completa
- **WebSocket**: ✅ Tiempo real
- **Testnet**: ❌ No disponible
- **Características**: Alta seguridad, nunca hackeado, buena liquidez en fiat
- **Documentación**: [Kraken API](https://docs.kraken.com/rest/)

### 3. Coinbase Pro
- **API REST**: ✅ Completa
- **WebSocket**: ✅ Tiempo real
- **Testnet**: ✅ Sandbox disponible
- **Características**: Regulado en US, seguros FDIC, alta liquidez USD
- **Documentación**: [Coinbase Pro API](https://docs.pro.coinbase.com/)

## 🏗️ Arquitectura

```
exchanges/
├── base.py              # Clase base abstracta
├── binance_connector.py # Conector Binance
├── kraken_connector.py  # Conector Kraken
├── coinbase_connector.py# Conector Coinbase Pro
├── factory.py           # Factory para crear conectores
├── manager.py           # Gestor multi-exchange
└── __init__.py         # Exports públicos
```

### Clases Principales

- **ExchangeBase**: Interfaz base común para todos los exchanges
- **ExchangeFactory**: Factory para crear instancias de conectores
- **ExchangeManager**: Gestor para coordinar múltiples exchanges
- **Conectores específicos**: Implementaciones para cada exchange

## 🚀 Uso Básico

### 1. Crear Conector Individual

```python
from src.exchanges.factory import ExchangeFactory

# Configurar credenciales
config = {\n    'api_key': 'tu_api_key',\n    'api_secret': 'tu_api_secret'\n}

# Crear conector
exchange = ExchangeFactory.create_exchange('binance', config, testnet=True)

# Conectar
await exchange.connect()

# Usar el exchange
ticker = await exchange.get_ticker('BTC/USDT')
print(f\"Precio BTC: {ticker.last}\")

# Desconectar
await exchange.disconnect()
```

### 2. Usar Gestor Multi-Exchange

```python
from src.exchanges.manager import ExchangeManager

# Crear gestor
manager = ExchangeManager()

# Añadir exchanges
await manager.add_exchange('binance_main', 'binance', binance_config)
await manager.add_exchange('kraken_main', 'kraken', kraken_config)

# Conectar todos
await manager.connect_all()

# Obtener mejor precio
best_exchange, best_price = await manager.get_best_price('BTC/USDT', 'buy')

# Buscar arbitraje
opportunities = await manager.get_arbitrage_opportunities(['BTC/USDT'])
```

## 📊 Operaciones Disponibles

### Market Data
- `get_ticker(symbol)` - Obtener ticker
- `get_order_book(symbol, limit)` - Libro de órdenes
- `get_candles(symbol, interval, limit)` - Datos OHLCV

### Trading
- `create_order(symbol, side, type, quantity, price)` - Crear orden
- `cancel_order(order_id, symbol)` - Cancelar orden
- `get_order(order_id, symbol)` - Estado de orden
- `get_open_orders(symbol)` - Órdenes abiertas
- `get_trades(symbol, limit)` - Historial de trades

### Account
- `get_balance()` - Balances de cuenta

### WebSocket Streaming
- `subscribe_ticker(symbol, callback)` - Actualizaciones de ticker
- `subscribe_order_book(symbol, callback)` - Actualizaciones de orderbook
- `subscribe_trades(callback)` - Trades propios en tiempo real

## ⚙️ Configuración

### Archivo de Configuración

Edita `config/exchanges.json`:

```json
{
  \"exchanges\": {
    \"binance\": {
      \"enabled\": true,
      \"testnet\": true,
      \"credentials\": {
        \"api_key\": \"tu_api_key\",
        \"api_secret\": \"tu_api_secret\"
      },
      \"settings\": {
        \"rate_limit\": 1200,
        \"order_book_limit\": 20
      }
    }
  }
}
```

### Variables de Entorno

```bash
# Binance
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Kraken
KRAKEN_API_KEY=tu_api_key
KRAKEN_API_SECRET=tu_api_secret

# Coinbase Pro
COINBASE_API_KEY=tu_api_key
COINBASE_API_SECRET=tu_api_secret
COINBASE_PASSPHRASE=tu_passphrase
```

## 🔒 Seguridad

### Mejores Prácticas

1. **API Keys**:
   - Usa permisos mínimos necesarios
   - Activa restricción por IP cuando sea posible
   - Rota las keys regularmente
   - NUNCA commits keys en el código

2. **Testnet Primero**:
   - Siempre prueba en testnet antes de producción
   - Valida todas las operaciones críticas

3. **Rate Limiting**:
   - Respeta los límites de cada exchange
   - Implementa backoff exponencial
   - Monitorea el uso de límites

4. **Error Handling**:
   - Maneja errores de red y API
   - Implementa reconexión automática
   - Logea errores importantes

## 📈 Funcionalidades Avanzadas

### 1. Arbitraje Automático

```python
# Buscar oportunidades
opportunities = await manager.get_arbitrage_opportunities(
    symbols=['BTC/USDT', 'ETH/USDT'],
    min_profit_pct=0.5
)

# Ejecutar arbitraje
for opportunity in opportunities:
    result = await manager.execute_arbitrage(
        opportunity['symbol'],
        opportunity['buy_exchange'],
        opportunity['sell_exchange'],
        Decimal('0.01')
    )
```

### 2. Libro de Órdenes Agregado

```python
# Obtener libro agregado de múltiples exchanges
aggregated_book = await manager.get_aggregated_orderbook('BTC/USDT')

# Mejor liquidez disponible
best_bids = aggregated_book.bids[:10]
best_asks = aggregated_book.asks[:10]
```

### 3. Streaming de Datos

```python
async def on_ticker_update(ticker):
    print(f\"Precio actualizado: {ticker.symbol} = {ticker.last}\")

# Suscribirse a múltiples exchanges
for exchange_name, exchange in manager.exchanges.items():
    await exchange.subscribe_ticker('BTC/USDT', on_ticker_update)
```

## 🧪 Testing

### Ejecutar Tests

```bash
# Tests completos
python -m pytest tests/test_exchanges.py -v

# Test específico
python -m pytest tests/test_exchanges.py::TestBinanceConnector -v

# Con cobertura
python -m pytest tests/test_exchanges.py --cov=src.exchanges
```

### Demo Scripts

```bash
# Demo básico
python examples/exchange_demo.py

# Demo con credenciales (configurar primero)
python examples/exchange_demo.py --live
```

## 🔧 Extensibilidad

### Añadir Nuevo Exchange

1. **Crear Conector**:
```python
from .base import ExchangeBase

class NuevoExchangeConnector(ExchangeBase):
    async def connect(self):
        # Implementar conexión
        pass
        
    async def get_ticker(self, symbol):
        # Implementar obtención de ticker
        pass
    
    # ... implementar otros métodos abstractos
```

2. **Registrar en Factory**:
```python
# En factory.py
_connectors = {
    'binance': BinanceConnector,
    'kraken': KrakenConnector,
    'coinbase': CoinbaseConnector,
    'nuevo_exchange': NuevoExchangeConnector,  # Añadir aquí
}
```

3. **Añadir Tests**:
```python
class TestNuevoExchangeConnector:
    def test_get_ticker(self):
        # Implementar tests
        pass
```

## 📊 Monitoreo y Métricas

### Métricas Disponibles

- Latencia de API calls
- Rate limit usage
- Errores por exchange
- Volumen de trades
- Uptime de conexiones

### Logs

```python
import logging

# Configurar logging
logging.getLogger('src.exchanges').setLevel(logging.INFO)

# Los logs incluyen:
# - Conexiones/desconexiones
# - Errores de API
# - Rate limiting
# - Operaciones críticas
```

## 🚨 Solución de Problemas

### Errores Comunes

1. **API Key Invalid**:
   - Verifica las credenciales
   - Confirma permisos de la key
   - Revisa restricciones de IP

2. **Rate Limited**:
   - Reduce frecuencia de calls
   - Implementa delays
   - Usa WebSocket cuando sea posible

3. **Connection Timeout**:
   - Verifica conectividad de red
   - Aumenta timeout en configuración
   - Implementa retry logic

4. **Symbol Not Found**:
   - Verifica formato del símbolo por exchange
   - Usa símbolos válidos del exchange
   - Consulta documentación del exchange

### Debug Mode

```python
# Activar logs detallados
import logging
logging.getLogger('src.exchanges').setLevel(logging.DEBUG)

# WebSocket debug
exchange.ws_debug = True
```

## 🔄 Roadmap

### Próximas Funcionalidades

- [ ] Soporte para más exchanges (Bybit, OKX, etc.)
- [ ] Trading algorítmico integrado
- [ ] Métricas avanzadas de performance
- [ ] Risk management automático
- [ ] Integration con DeFi protocols
- [ ] Support para options y futures
- [ ] Advanced order types (OCO, etc.)
- [ ] Portfolio rebalancing automático

## 📚 Referencias

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Kraken API Documentation](https://docs.kraken.com/rest/)
- [Coinbase Pro API Documentation](https://docs.pro.coinbase.com/)
- [CCXT Library](https://github.com/ccxt/ccxt) - Inspiración para unificación
- [WebSocket Best Practices](https://tools.ietf.org/html/rfc6455)

## 📄 Licencia

Proprietary - Quantum Trading Platform