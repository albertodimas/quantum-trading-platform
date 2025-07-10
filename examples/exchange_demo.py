"""
Demo de uso de conectores de exchanges
Muestra cómo usar los diferentes conectores y el gestor
"""

import asyncio
import json
from decimal import Decimal
from datetime import datetime

from src.exchanges.factory import ExchangeFactory
from src.exchanges.manager import ExchangeManager

async def demo_binance_connector():
    """Demo del conector de Binance"""
    print("=== Demo Conector Binance ===")
    
    # Configuración (usar credenciales reales en producción)
    config = {
        'api_key': 'tu_api_key_binance',
        'api_secret': 'tu_api_secret_binance'
    }
    
    try:
        # Crear conector
        exchange = ExchangeFactory.create_exchange('binance', config, testnet=True)
        
        # Conectar
        await exchange.connect()
        print("✅ Conectado a Binance testnet")
        
        # Obtener ticker
        ticker = await exchange.get_ticker('BTC/USDT')
        print(f"📊 Ticker BTC/USDT: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
        
        # Obtener libro de órdenes
        orderbook = await exchange.get_order_book('BTC/USDT', limit=5)
        print(f"📖 OrderBook - Bids: {len(orderbook.bids)}, Asks: {len(orderbook.asks)}")
        print(f"   Mejor Bid: {orderbook.bids[0][0]} ({orderbook.bids[0][1]})")
        print(f"   Mejor Ask: {orderbook.asks[0][0]} ({orderbook.asks[0][1]})")
        
        # Obtener balance (requiere credenciales válidas)
        try:
            balances = await exchange.get_balance()
            print(f"💰 Balances encontrados: {len(balances)}")
            for currency, balance in list(balances.items())[:3]:
                print(f"   {currency}: {balance.total} (Free: {balance.free}, Locked: {balance.locked})")
        except Exception as e:
            print(f"⚠️  Error obteniendo balance: {e}")
            
        # Desconectar
        await exchange.disconnect()
        print("✅ Desconectado de Binance")
        
    except Exception as e:
        print(f"❌ Error en demo Binance: {e}")

async def demo_kraken_connector():
    """Demo del conector de Kraken"""
    print("\n=== Demo Conector Kraken ===")
    
    config = {
        'api_key': 'tu_api_key_kraken',
        'api_secret': 'tu_api_secret_kraken'
    }
    
    try:
        # Crear conector
        exchange = ExchangeFactory.create_exchange('kraken', config)
        
        # Conectar
        await exchange.connect()
        print("✅ Conectado a Kraken")
        
        # Obtener ticker
        ticker = await exchange.get_ticker('BTC/USD')
        print(f"📊 Ticker BTC/USD: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
        
        # Obtener libro de órdenes
        orderbook = await exchange.get_order_book('BTC/USD', limit=5)
        print(f"📖 OrderBook - Bids: {len(orderbook.bids)}, Asks: {len(orderbook.asks)}")
        
        # Desconectar
        await exchange.disconnect()
        print("✅ Desconectado de Kraken")
        
    except Exception as e:
        print(f"❌ Error en demo Kraken: {e}")

async def demo_coinbase_connector():
    """Demo del conector de Coinbase Pro"""
    print("\n=== Demo Conector Coinbase Pro ===")
    
    config = {
        'api_key': 'tu_api_key_coinbase',
        'api_secret': 'tu_api_secret_coinbase',
        'passphrase': 'tu_passphrase_coinbase'
    }
    
    try:
        # Crear conector
        exchange = ExchangeFactory.create_exchange('coinbase', config, testnet=True)
        
        # Conectar
        await exchange.connect()
        print("✅ Conectado a Coinbase Pro sandbox")
        
        # Obtener ticker
        ticker = await exchange.get_ticker('BTC/USD')
        print(f"📊 Ticker BTC/USD: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
        
        # Obtener libro de órdenes
        orderbook = await exchange.get_order_book('BTC/USD', limit=5)
        print(f"📖 OrderBook - Bids: {len(orderbook.bids)}, Asks: {len(orderbook.asks)}")
        
        # Desconectar
        await exchange.disconnect()
        print("✅ Desconectado de Coinbase Pro")
        
    except Exception as e:
        print(f"❌ Error en demo Coinbase: {e}")

async def demo_exchange_manager():
    """Demo del gestor de múltiples exchanges"""
    print("\n=== Demo Exchange Manager ===")
    
    try:
        # Crear gestor
        manager = ExchangeManager()
        
        # Configuraciones de ejemplo
        exchanges_config = {
            'binance': {
                'api_key': 'binance_key',
                'api_secret': 'binance_secret'
            },
            'kraken': {
                'api_key': 'kraken_key',
                'api_secret': 'kraken_secret'
            }
        }
        
        # Añadir exchanges
        for name, config in exchanges_config.items():
            await manager.add_exchange(f"{name}_demo", name, config, testnet=True)
            print(f"➕ Exchange {name} añadido")
            
        # Mostrar estado
        status = manager.get_exchange_status()
        print(f"📈 Estado de exchanges: {len(status)} configurados")
        
        for exchange_name, info in status.items():
            print(f"   {exchange_name}: {'🟢' if info['connected'] else '🔴'} "
                  f"({info['type']}, testnet={info['testnet']})")
            
        # Demo de obtención del mejor precio (requiere conexión real)
        print("\n🔍 Demo búsqueda mejor precio (simulado):")
        # best_exchange, best_price = await manager.get_best_price('BTC/USDT', 'buy')
        # print(f"💰 Mejor precio para comprar BTC/USDT: {best_price} en {best_exchange}")
        
        # Demo de arbitraje (simulado)
        print("🔄 Demo búsqueda arbitraje (simulado):")
        # opportunities = await manager.get_arbitrage_opportunities(['BTC/USDT'], min_profit_pct=0.5)
        # print(f"⚡ Oportunidades encontradas: {len(opportunities)}")
        
        print("✅ Demo del gestor completado")
        
    except Exception as e:
        print(f"❌ Error en demo del gestor: {e}")

async def demo_websocket_streaming():
    """Demo de streaming WebSocket"""
    print("\n=== Demo WebSocket Streaming ===")
    
    config = {
        'api_key': 'tu_api_key',
        'api_secret': 'tu_api_secret'
    }
    
    try:
        exchange = ExchangeFactory.create_exchange('binance', config, testnet=True)
        
        # Callback para ticker
        async def on_ticker_update(ticker):
            print(f"🔄 Ticker Update: {ticker.symbol} = {ticker.last} "
                  f"(Bid: {ticker.bid}, Ask: {ticker.ask})")
                  
        # Callback para orderbook
        async def on_orderbook_update(data):
            print(f"📖 OrderBook Update: {data}")
            
        # Conectar
        await exchange.connect()
        print("✅ Conectado para streaming")
        
        # Suscribirse a actualizaciones
        await exchange.subscribe_ticker('BTC/USDT', on_ticker_update)
        await exchange.subscribe_order_book('BTC/USDT', on_orderbook_update)
        
        print("🎧 Escuchando actualizaciones por 30 segundos...")
        await asyncio.sleep(30)
        
        # Desconectar
        await exchange.disconnect()
        print("✅ Streaming finalizado")
        
    except Exception as e:
        print(f"❌ Error en demo WebSocket: {e}")

async def demo_trading_operations():
    """Demo de operaciones de trading"""
    print("\n=== Demo Operaciones de Trading ===")
    print("⚠️  CUIDADO: Este demo usa órdenes reales en testnet")
    
    config = {
        'api_key': 'tu_api_key_real',
        'api_secret': 'tu_api_secret_real'
    }
    
    try:
        exchange = ExchangeFactory.create_exchange('binance', config, testnet=True)
        await exchange.connect()
        
        symbol = 'BTC/USDT'
        
        # Obtener precio actual
        ticker = await exchange.get_ticker(symbol)
        print(f"📊 Precio actual {symbol}: {ticker.last}")
        
        # Crear orden limit de compra por debajo del precio actual
        buy_price = ticker.bid * Decimal('0.95')  # 5% por debajo
        quantity = Decimal('0.001')  # Cantidad pequeña para demo
        
        print(f"📝 Creando orden de compra: {quantity} {symbol} a {buy_price}")
        
        # Crear orden (descomenta para ejecutar)
        # order = await exchange.create_order(symbol, 'buy', 'limit', quantity, buy_price)
        # print(f"✅ Orden creada: ID={order.id}, Status={order.status}")
        
        # # Verificar estado de la orden
        # order_status = await exchange.get_order(order.id, symbol)
        # print(f"🔍 Estado orden: {order_status.status}")
        
        # # Cancelar orden
        # cancelled = await exchange.cancel_order(order.id, symbol)
        # print(f"❌ Orden cancelada: {cancelled}")
        
        # Obtener órdenes abiertas
        open_orders = await exchange.get_open_orders(symbol)
        print(f"📋 Órdenes abiertas: {len(open_orders)}")
        
        # Obtener historial de trades
        trades = await exchange.get_trades(symbol, limit=5)
        print(f"📈 Trades recientes: {len(trades)}")
        
        await exchange.disconnect()
        print("✅ Demo de trading completado")
        
    except Exception as e:
        print(f"❌ Error en demo trading: {e}")

def demo_factory_info():
    """Demo de información del factory"""
    print("\n=== Demo Factory Info ===")
    
    # Exchanges soportados
    supported = ExchangeFactory.get_supported_exchanges()
    print(f"🏪 Exchanges soportados: {supported}")
    
    # Información de exchanges
    for exchange_name in supported:
        info = ExchangeFactory.get_exchange_info(exchange_name)
        print(f"\n📋 {info.get('name', exchange_name)}:")
        print(f"   Website: {info.get('website', 'N/A')}")
        print(f"   Testnet: {'✅' if info.get('supports_testnet') else '❌'}")
        print(f"   Fees: {info.get('fee_structure', 'N/A')}")
        print(f"   WebSocket: {', '.join(info.get('websocket_streams', []))}")
        
    # Comparación de exchanges
    print("\n💰 Comparación por fees:")
    fee_comparison = ExchangeFactory.compare_exchanges('fees')
    for exchange, description in fee_comparison.items():
        print(f"   {exchange}: {description}")
        
    print("\n🔒 Comparación por seguridad:")
    security_comparison = ExchangeFactory.compare_exchanges('security')
    for exchange, description in security_comparison.items():
        print(f"   {exchange}: {description}")

async def main():
    """Función principal del demo"""
    print("🚀 Demo de Conectores de Exchanges")
    print("=" * 50)
    
    # Demo de información del factory
    demo_factory_info()
    
    # Demo de conectores individuales (descomenta para probar con credenciales reales)
    # await demo_binance_connector()
    # await demo_kraken_connector()
    # await demo_coinbase_connector()
    
    # Demo del gestor de exchanges
    await demo_exchange_manager()
    
    # Demo de WebSocket (requiere credenciales válidas)
    # await demo_websocket_streaming()
    
    # Demo de trading (requiere credenciales válidas y MUCHO CUIDADO)
    # await demo_trading_operations()
    
    print("\n🎉 Demo completado")
    print("\n📌 Para usar con credenciales reales:")
    print("   1. Obtén API keys de los exchanges")
    print("   2. Configura las credenciales en config/exchanges.json")
    print("   3. Descomenta las secciones relevantes")
    print("   4. SIEMPRE usa testnet primero!")

if __name__ == "__main__":
    asyncio.run(main())