"""
Example backtesting script demonstrating the framework usage
"""

import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting import (
    BacktestEngine, BacktestConfig, BacktestMode,
    HistoricalDataProvider
)
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.core.architecture import DIContainer
from src.core.cache import CacheManager
from src.core.observability import setup_logging

async def run_backtest():
    """Run a simple backtest example"""
    # Setup logging
    setup_logging()
    
    # Initialize dependency injection
    container = DIContainer()
    
    # Create configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=100000,
        symbols=['BTC/USDT', 'ETH/USDT'],
        mode=BacktestMode.BAR_BY_BAR,
        timeframe='1h',
        commission=0.001,
        slippage_model='realistic',
        allow_shorting=True,
        margin_ratio=0.5
    )    
    # Create components
    cache_manager = CacheManager()
    data_provider = HistoricalDataProvider(cache_manager)
    
    # Create strategy
    strategy = MeanReversionStrategy(
        lookback_period=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        position_size=0.1
    )
    
    # Create and initialize engine
    engine = BacktestEngine()
    await engine.initialize(config, strategy, data_provider)
    
    # Run backtest
    print("Starting backtest...")
    result = await engine.run()
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Total Return: {result.performance_metrics.total_return:.2%}")
    print(f"Annual Return: {result.performance_metrics.annual_return:.2%}")
    print(f"Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.performance_metrics.max_drawdown:.2%}")
    print(f"Win Rate: {result.performance_metrics.win_rate:.2%}")
    print(f"Total Trades: {result.performance_metrics.total_trades}")
    
    # Save results
    result.equity_curve.to_csv('backtest_equity_curve.csv')
    result.trades.to_csv('backtest_trades.csv')
    
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    asyncio.run(run_backtest())