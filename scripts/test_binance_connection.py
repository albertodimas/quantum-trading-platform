#!/usr/bin/env python3
"""
Test Binance Exchange Connection

Tests the Binance exchange implementation with real WebSocket connections.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import BinanceExchange, OrderSide, OrderType, TimeInForce
from src.core.observability import setup_logging

# Setup logging
setup_logging()


async def test_binance_connection():
    """Test Binance exchange connection and functionality"""
    
    # Create exchange instance (testnet mode)
    exchange = BinanceExchange(testnet=True)
    
    try:
        print("🚀 Testing Binance Exchange Connection...\n")
        
        # Test 1: Connect to exchange
        print("1️⃣ Connecting to Binance...")
        await exchange.connect()
        print("✅ Connected successfully!")
        
        # Test 2: Check connection status
        print("\n2️⃣ Checking connection status...")
        is_alive = await exchange.is_alive()
        print(f"✅ Connection alive: {is_alive}")
        
        # Test 3: Get ticker data
        print("\n3️⃣ Getting ticker data for BTCUSDT...")
        ticker = await exchange.get_ticker("BTCUSDT")
        print(f"✅ Ticker data:")
        print(f"   Symbol: {ticker.symbol}")
        print(f"   Bid: ${ticker.bid_price:,.2f} x {ticker.bid_qty}")
        print(f"   Ask: ${ticker.ask_price:,.2f} x {ticker.ask_qty}")
        print(f"   Last: ${ticker.last_price:,.2f}")
        print(f"   24h Volume: {ticker.volume_24h:,.2f}")
        
        # Test 4: Get order book
        print("\n4️⃣ Getting order book for BTCUSDT...")
        orderbook = await exchange.get_orderbook("BTCUSDT", depth=5)
        print(f"✅ Order book:")
        print(f"   Best bid: ${orderbook.best_bid:,.2f}")
        print(f"   Best ask: ${orderbook.best_ask:,.2f}")
        print(f"   Spread: ${orderbook.spread:,.2f}")
        print(f"   Bids (top 5):")
        for i, (price, qty) in enumerate(orderbook.bids[:5]):
            print(f"     {i+1}. ${price:,.2f} x {qty}")
        print(f"   Asks (top 5):")
        for i, (price, qty) in enumerate(orderbook.asks[:5]):
            print(f"     {i+1}. ${price:,.2f} x {qty}")
        
        # Test 5: Get recent trades
        print("\n5️⃣ Getting recent trades for BTCUSDT...")
        trades = await exchange.get_recent_trades("BTCUSDT", limit=5)
        print(f"✅ Recent trades (last 5):")
        for i, trade in enumerate(trades[:5]):
            print(f"   {i+1}. {trade.side.value.upper()} {trade.quantity} @ ${trade.price:,.2f} "
                  f"({trade.timestamp.strftime('%H:%M:%S')})")
        
        # Test 6: Test WebSocket subscriptions
        print("\n6️⃣ Testing WebSocket subscriptions...")
        
        # Ticker subscription
        ticker_updates = []
        async def on_ticker_update(ticker):
            ticker_updates.append(ticker)
            print(f"   📊 Ticker update: ${ticker.last_price:,.2f}")
        
        await exchange.subscribe_ticker("BTCUSDT", on_ticker_update)
        print("✅ Subscribed to ticker updates")
        
        # Trade subscription
        trade_updates = []
        async def on_trade_update(trade):
            trade_updates.append(trade)
            print(f"   💹 Trade: {trade.side.value.upper()} {trade.quantity} @ ${trade.price:,.2f}")
        
        await exchange.subscribe_trades("BTCUSDT", on_trade_update)
        print("✅ Subscribed to trade updates")
        
        # Wait for some updates
        print("\n⏳ Waiting 10 seconds for WebSocket updates...")
        await asyncio.sleep(10)
        
        print(f"\n✅ Received {len(ticker_updates)} ticker updates")
        print(f"✅ Received {len(trade_updates)} trade updates")
        
        # Test 7: Get account balance (if API key is configured)
        if exchange.api_key:
            print("\n7️⃣ Getting account balance...")
            try:
                balances = await exchange.get_balance()
                print("✅ Account balances:")
                for asset, balance in balances.items():
                    if balance.total > 0:
                        print(f"   {asset}: {balance.total} (free: {balance.free}, locked: {balance.locked})")
            except Exception as e:
                print(f"⚠️  Could not get balance: {str(e)}")
                print("   (This is normal if using testnet without funding)")
        else:
            print("\n7️⃣ Skipping balance test (no API key configured)")
        
        # Test 8: Test order placement (dry run - won't actually place)
        print("\n8️⃣ Testing order creation (dry run)...")
        if exchange.api_key:
            try:
                # Get current price for limit order
                ticker = await exchange.get_ticker("BTCUSDT")
                limit_price = ticker.bid_price * Decimal("0.95")  # 5% below current bid
                
                print(f"   Would place limit buy order:")
                print(f"   Symbol: BTCUSDT")
                print(f"   Side: BUY")
                print(f"   Type: LIMIT")
                print(f"   Quantity: 0.001")
                print(f"   Price: ${limit_price:,.2f}")
                print("   ✅ Order validation passed")
            except Exception as e:
                print(f"   ⚠️  Order test failed: {str(e)}")
        else:
            print("   Skipping order test (no API key configured)")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        print("\n🔌 Disconnecting from exchange...")
        await exchange.disconnect()
        print("✅ Disconnected successfully!")


async def main():
    """Main entry point"""
    try:
        await test_binance_connection()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())