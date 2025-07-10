"""
🧪 Tests para MEV Flash Loan Engine
Tests completos para el motor de arbitraje con flash loans
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timedelta

from src.arbitrage.mev_flash_loan_engine import (
    MEVFlashLoanEngine,
    FlashLoanOpportunity,
    ArbitrageExecution,
    ArbitrageType,
    ArbitrageStatus,
    ExchangePair
)

@pytest.fixture
def mock_config():
    """Configuración mock para tests"""
    return {
        'database_url': 'postgresql://test:test@localhost:5432/test_db',
        'min_profit_threshold': '100',  # $100 USD mínimo
        'max_gas_price': 50,  # 50 Gwei
        'max_slippage': 0.005,  # 0.5%
        'opportunity_timeout': 30,  # 30 segundos
        'ethereum_rpc': 'http://localhost:8545',
        'polygon_rpc': 'http://localhost:8546',
        'exchanges': {
            'binance': {
                'enabled': True,
                'api_key': 'test_key',
                'secret': 'test_secret',
                'sandbox': True
            },
            'coinbase': {
                'enabled': True,
                'api_key': 'test_key',
                'secret': 'test_secret',
                'sandbox': True
            }
        }
    }

@pytest.fixture
def mev_engine(mock_config):
    """Instancia del motor MEV para tests"""
    return MEVFlashLoanEngine(mock_config)

@pytest.fixture
def sample_opportunity():
    """Oportunidad de arbitraje de ejemplo"""
    return FlashLoanOpportunity(
        id="test_opp_001",
        type=ArbitrageType.CROSS_EXCHANGE,
        asset="BTC/USDT",
        amount=Decimal('1.0'),
        profit_estimate=Decimal('150.0'),
        gas_cost=Decimal('20.0'),
        net_profit=Decimal('130.0'),
        confidence=0.85,
        expiry_time=datetime.now() + timedelta(seconds=30),
        execution_path=[
            {'exchange': 'binance', 'action': 'buy', 'price': 45000},
            {'exchange': 'coinbase', 'action': 'sell', 'price': 45150}
        ],
        risk_score=0.3
    )

class TestMEVFlashLoanEngine:
    """Tests para la clase principal MEVFlashLoanEngine"""
    
    def test_initialization(self, mev_engine, mock_config):
        """Test inicialización del motor"""
        assert mev_engine.config == mock_config
        assert mev_engine.min_profit_threshold == Decimal('100')
        assert mev_engine.max_gas_price == 50
        assert mev_engine.max_slippage == 0.005
        assert mev_engine.opportunity_timeout == 30
        assert len(mev_engine.active_opportunities) == 0
        assert len(mev_engine.execution_history) == 0
        assert mev_engine.total_profit == Decimal('0')
        assert mev_engine.success_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mev_engine):
        """Test inicialización exitosa"""
        with patch.object(mev_engine, '_connect_database', new_callable=AsyncMock), \
             patch.object(mev_engine, '_initialize_web3_connections', new_callable=AsyncMock), \
             patch.object(mev_engine, '_load_dex_contracts', new_callable=AsyncMock), \
             patch.object(mev_engine, '_load_flash_loan_providers', new_callable=AsyncMock), \
             patch.object(mev_engine, '_initialize_exchanges', new_callable=AsyncMock), \
             patch.object(mev_engine, '_setup_mev_monitors', new_callable=AsyncMock):
            
            await mev_engine.initialize()
            
            # Verificar que todos los métodos de inicialización fueron llamados
            mev_engine._connect_database.assert_called_once()
            mev_engine._initialize_web3_connections.assert_called_once()
            mev_engine._load_dex_contracts.assert_called_once()
            mev_engine._load_flash_loan_providers.assert_called_once()
            mev_engine._initialize_exchanges.assert_called_once()
            mev_engine._setup_mev_monitors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_opportunity(self, mev_engine, sample_opportunity):
        """Test agregar nueva oportunidad"""
        with patch.object(mev_engine, '_validate_opportunity', return_value=True) as mock_validate:
            await mev_engine._add_opportunity(sample_opportunity)
            
            assert sample_opportunity.id in mev_engine.active_opportunities
            assert mev_engine.active_opportunities[sample_opportunity.id] == sample_opportunity
            assert sample_opportunity.status == ArbitrageStatus.READY
            mock_validate.assert_called_once_with(sample_opportunity)
    
    def test_get_statistics_empty(self, mev_engine):
        """Test estadísticas con motor vacío"""
        stats = mev_engine.get_statistics()
        
        assert stats['total_profit_usd'] == 0.0
        assert stats['active_opportunities'] == 0
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_profit_per_trade'] == 0.0
        assert len(stats['opportunities_by_type']) == 0
        assert len(stats['recent_executions']) == 0
    
    def test_get_statistics_with_data(self, mev_engine, sample_opportunity):
        """Test estadísticas con datos"""
        # Agregar oportunidad activa
        mev_engine.active_opportunities[sample_opportunity.id] = sample_opportunity
        
        # Agregar historial de ejecución
        execution = ArbitrageExecution(
            opportunity_id=sample_opportunity.id,
            executed_at=datetime.now(),
            actual_profit=Decimal('125.0'),
            gas_used=21000,
            gas_price=Decimal('20'),
            transaction_hash='0x123',
            success=True
        )
        mev_engine.execution_history.append(execution)
        mev_engine.total_profit = Decimal('125.0')
        
        stats = mev_engine.get_statistics()
        
        assert stats['total_profit_usd'] == 125.0
        assert stats['active_opportunities'] == 1
        assert stats['total_executions'] == 1
        assert stats['successful_executions'] == 1
        assert stats['success_rate'] == 1.0
        assert stats['average_profit_per_trade'] == 125.0
        assert stats['opportunities_by_type']['cross_exchange'] == 1
        assert len(stats['recent_executions']) == 1

class TestOpportunityDetection:
    """Tests para detección de oportunidades"""
    
    @pytest.mark.asyncio
    async def test_detect_cross_exchange_opportunities(self, mev_engine):
        """Test detección de arbitraje cross-exchange"""
        price_data = {
            'BTC/USDT': {
                'binance': {'bid': 45000, 'ask': 45010},
                'coinbase': {'bid': 44900, 'ask': 44910}
            }
        }
        
        with patch.object(mev_engine, '_create_cross_exchange_opportunity') as mock_create:
            mock_opportunity = Mock()
            mock_opportunity.net_profit = Decimal('150')
            mock_create.return_value = mock_opportunity
            
            opportunities = await mev_engine._detect_cross_exchange_opportunities(price_data)
            
            assert len(opportunities) == 1
            assert opportunities[0] == mock_opportunity
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_dex_arbitrage(self, mev_engine):
        """Test detección de arbitraje DEX"""
        dex_prices = {
            'ETH/USDT': {
                'uniswap_v2': 3000.0,
                'sushiswap': 2985.0
            }
        }
        
        with patch.object(mev_engine, '_create_dex_arbitrage_opportunity') as mock_create:
            mock_opportunity = Mock()
            mock_opportunity.net_profit = Decimal('75')
            mock_create.return_value = mock_opportunity
            
            opportunities = await mev_engine._detect_dex_arbitrage(dex_prices)
            
            assert len(opportunities) == 1
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_sandwich_opportunities(self, mev_engine):
        """Test detección de oportunidades sandwich"""
        pending_txs = [
            {'hash': '0x123', 'value': 1000000, 'to': '0xUniswapRouter'}
        ]
        
        with patch.object(mev_engine, '_is_large_swap_transaction', return_value=True), \
             patch.object(mev_engine, '_calculate_price_impact', return_value=Decimal('0.02')), \
             patch.object(mev_engine, '_create_sandwich_opportunity') as mock_create:
            
            mock_opportunity = Mock()
            mock_opportunity.net_profit = Decimal('200')
            mock_create.return_value = mock_opportunity
            
            opportunities = await mev_engine._detect_sandwich_opportunities(pending_txs)
            
            assert len(opportunities) == 1
            mock_create.assert_called_once()

class TestArbitrageExecution:
    """Tests para ejecución de arbitraje"""
    
    @pytest.mark.asyncio
    async def test_execute_cross_exchange_arbitrage_success(self, mev_engine, sample_opportunity):
        """Test ejecución exitosa de arbitraje cross-exchange"""
        # Mock de resultados de trading
        buy_result = {'success': True, 'cost': 44950, 'order_id': 'buy123'}
        sell_result = {'success': True, 'proceeds': 45100, 'order_id': 'sell456'}
        
        with patch.object(mev_engine, '_execute_buy_order', return_value=buy_result), \
             patch.object(mev_engine, '_execute_sell_order', return_value=sell_result):
            
            execution = await mev_engine._execute_cross_exchange_arbitrage(sample_opportunity)
            
            assert execution.success == True
            assert execution.actual_profit == Decimal('150')  # 45100 - 44950
            assert execution.transaction_hash == 'buy123|sell456'
    
    @pytest.mark.asyncio
    async def test_execute_cross_exchange_arbitrage_failure(self, mev_engine, sample_opportunity):
        """Test fallo en ejecución de arbitraje cross-exchange"""
        # Mock de fallo en compra
        buy_result = {'success': False, 'error': 'Insufficient balance'}
        sell_result = {'success': True, 'proceeds': 45100, 'order_id': 'sell456'}
        
        with patch.object(mev_engine, '_execute_buy_order', return_value=buy_result), \
             patch.object(mev_engine, '_execute_sell_order', return_value=sell_result):
            
            execution = await mev_engine._execute_cross_exchange_arbitrage(sample_opportunity)
            
            assert execution.success == False
            assert execution.actual_profit == Decimal('0')
            assert execution.error_message == "Una o ambas operaciones fallaron"
    
    @pytest.mark.asyncio
    async def test_execute_flash_loan_arbitrage_success(self, mev_engine, sample_opportunity):
        """Test ejecución exitosa de flash loan arbitrage"""
        sample_opportunity.type = ArbitrageType.FLASH_LOAN
        
        with patch.object(mev_engine, '_select_best_flash_loan_provider', return_value='aave'), \
             patch.object(mev_engine, '_build_flash_loan_transaction', return_value={}), \
             patch.object(mev_engine, '_simulate_flash_loan', return_value={'success': True}), \
             patch.object(mev_engine, '_execute_flash_loan_transaction', return_value='0x456'), \
             patch.object(mev_engine, '_wait_for_confirmation') as mock_wait, \
             patch.object(mev_engine, '_calculate_actual_profit', return_value=Decimal('120')):
            
            # Mock del receipt
            mock_receipt = {
                'gasUsed': 150000,
                'effectiveGasPrice': 20000000000  # 20 Gwei
            }
            mock_wait.return_value = mock_receipt
            
            execution = await mev_engine._execute_flash_loan_arbitrage(sample_opportunity)
            
            assert execution.success == True
            assert execution.actual_profit == Decimal('120')
            assert execution.transaction_hash == '0x456'
            assert execution.gas_used == 150000

class TestFlashLoanOpportunity:
    """Tests para la clase FlashLoanOpportunity"""
    
    def test_opportunity_creation(self):
        """Test creación de oportunidad"""
        opportunity = FlashLoanOpportunity(
            id="test_001",
            type=ArbitrageType.FLASH_LOAN,
            asset="ETH/USDT",
            amount=Decimal('10.0'),
            profit_estimate=Decimal('200.0'),
            gas_cost=Decimal('50.0'),
            net_profit=Decimal('150.0'),
            confidence=0.9,
            expiry_time=datetime.now() + timedelta(seconds=60),
            execution_path=[{'step': 1, 'action': 'swap'}],
            risk_score=0.2
        )
        
        assert opportunity.id == "test_001"
        assert opportunity.type == ArbitrageType.FLASH_LOAN
        assert opportunity.asset == "ETH/USDT"
        assert opportunity.amount == Decimal('10.0')
        assert opportunity.net_profit == Decimal('150.0')
        assert opportunity.confidence == 0.9
        assert opportunity.risk_score == 0.2
        assert opportunity.status == ArbitrageStatus.DETECTED

class TestArbitrageExecution:
    """Tests para la clase ArbitrageExecution"""
    
    def test_execution_creation_success(self):
        """Test creación de ejecución exitosa"""
        execution = ArbitrageExecution(
            opportunity_id="test_001",
            executed_at=datetime.now(),
            actual_profit=Decimal('175.5'),
            gas_used=180000,
            gas_price=Decimal('25'),
            transaction_hash='0xabc123',
            success=True
        )
        
        assert execution.opportunity_id == "test_001"
        assert execution.actual_profit == Decimal('175.5')
        assert execution.gas_used == 180000
        assert execution.gas_price == Decimal('25')
        assert execution.transaction_hash == '0xabc123'
        assert execution.success == True
        assert execution.error_message is None
    
    def test_execution_creation_failure(self):
        """Test creación de ejecución fallida"""
        execution = ArbitrageExecution(
            opportunity_id="test_002",
            executed_at=datetime.now(),
            actual_profit=Decimal('0'),
            gas_used=0,
            gas_price=Decimal('0'),
            transaction_hash='',
            success=False,
            error_message="Transaction reverted"
        )
        
        assert execution.success == False
        assert execution.actual_profit == Decimal('0')
        assert execution.error_message == "Transaction reverted"

class TestExchangePair:
    """Tests para la clase ExchangePair"""
    
    def test_exchange_pair_creation(self):
        """Test creación de par de exchanges"""
        pair = ExchangePair(
            exchange_a="binance",
            exchange_b="coinbase",
            symbol="BTC/USDT",
            price_a=Decimal('45000'),
            price_b=Decimal('44950'),
            spread=Decimal('50'),
            volume_a=Decimal('100'),
            volume_b=Decimal('85'),
            min_trade_amount=Decimal('0.001'),
            max_trade_amount=Decimal('10')
        )
        
        assert pair.exchange_a == "binance"
        assert pair.exchange_b == "coinbase"
        assert pair.symbol == "BTC/USDT"
        assert pair.spread == Decimal('50')
        assert pair.volume_a == Decimal('100')

@pytest.mark.asyncio
async def test_opportunity_lifecycle(mev_engine, sample_opportunity):
    """Test ciclo de vida completo de una oportunidad"""
    # 1. Agregar oportunidad
    with patch.object(mev_engine, '_validate_opportunity', return_value=True):
        await mev_engine._add_opportunity(sample_opportunity)
    
    assert sample_opportunity.status == ArbitrageStatus.READY
    assert sample_opportunity.id in mev_engine.active_opportunities
    
    # 2. Ejecutar oportunidad
    with patch.object(mev_engine, '_execute_cross_exchange_arbitrage') as mock_execute:
        mock_execution = ArbitrageExecution(
            opportunity_id=sample_opportunity.id,
            executed_at=datetime.now(),
            actual_profit=Decimal('125'),
            gas_used=0,
            gas_price=Decimal('0'),
            transaction_hash='test_tx',
            success=True
        )
        mock_execute.return_value = mock_execution
        
        await mev_engine._execute_opportunity(sample_opportunity)
    
    # 3. Verificar resultado
    assert sample_opportunity.status == ArbitrageStatus.COMPLETED
    assert len(mev_engine.execution_history) == 1
    assert mev_engine.total_profit == Decimal('125')

@pytest.mark.asyncio
async def test_opportunity_expiration(mev_engine):
    """Test expiración de oportunidades"""
    # Crear oportunidad ya expirada
    expired_opportunity = FlashLoanOpportunity(
        id="expired_001",
        type=ArbitrageType.CROSS_EXCHANGE,
        asset="ETH/USDT",
        amount=Decimal('1.0'),
        profit_estimate=Decimal('100.0'),
        gas_cost=Decimal('20.0'),
        net_profit=Decimal('80.0'),
        confidence=0.8,
        expiry_time=datetime.now() - timedelta(seconds=10),  # Ya expirada
        execution_path=[],
        risk_score=0.3
    )
    
    mev_engine.active_opportunities[expired_opportunity.id] = expired_opportunity
    
    # Ejecutar limpieza
    await mev_engine._cleanup_expired_opportunities()
    
    # Verificar que la oportunidad fue removida
    assert expired_opportunity.id not in mev_engine.active_opportunities
    assert expired_opportunity.status == ArbitrageStatus.EXPIRED

# Configuración para pytest
if __name__ == "__main__":
    pytest.main([__file__])