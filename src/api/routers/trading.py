"""
Trading API endpoints.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user, get_trading_engine
from src.core.logging import get_logger
from src.trading.engine import TradingEngine
from src.trading.models import OrderSide, OrderType, Signal

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models

class SignalRequest(BaseModel):
    """Trading signal request."""
    
    symbol: str = Field(..., example="BTC/USDT")
    side: OrderSide
    confidence: float = Field(..., ge=0.0, le=1.0)
    entry_price: float = Field(..., gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    quantity: Optional[float] = Field(None, gt=0)
    strategy: str = Field(..., example="momentum_ai")
    metadata: Dict = Field(default_factory=dict)


class SignalResponse(BaseModel):
    """Trading signal response."""
    
    order_id: Optional[str]
    status: str
    message: str


class OrderRequest(BaseModel):
    """Direct order request."""
    
    symbol: str = Field(..., example="BTC/USDT")
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    time_in_force: str = Field(default="GTC")


class PortfolioStatus(BaseModel):
    """Portfolio status response."""
    
    timestamp: str
    total_value: float
    total_pnl: float
    position_count: int
    positions: List[Dict]
    metrics: Dict


# Endpoints

@router.post("/signal", response_model=SignalResponse)
async def process_signal(
    signal_request: SignalRequest,
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> SignalResponse:
    """
    Process a trading signal.
    
    This endpoint receives trading signals from strategies or AI agents
    and processes them through risk management before creating orders.
    """
    logger.info(
        "Processing trading signal",
        user=current_user.id,
        symbol=signal_request.symbol,
        side=signal_request.side,
    )
    
    try:
        # Convert to Signal model
        signal = Signal(
            symbol=signal_request.symbol,
            side=signal_request.side,
            confidence=signal_request.confidence,
            entry_price=signal_request.entry_price,
            stop_loss=signal_request.stop_loss,
            take_profit=signal_request.take_profit,
            quantity=signal_request.quantity,
            strategy=signal_request.strategy,
            metadata={
                **signal_request.metadata,
                "user_id": current_user.id,
            }
        )
        
        # Process signal
        order_id = await trading_engine.process_signal(signal)
        
        if order_id:
            return SignalResponse(
                order_id=order_id,
                status="accepted",
                message="Signal processed and order created",
            )
        else:
            return SignalResponse(
                order_id=None,
                status="rejected",
                message="Signal rejected by risk management",
            )
        
    except Exception as e:
        logger.error(f"Failed to process signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process signal",
        )


@router.post("/order")
async def create_order(
    order_request: OrderRequest,
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> Dict:
    """
    Create a direct order.
    
    This endpoint allows manual order creation, bypassing signal processing.
    Still goes through risk management checks.
    """
    logger.info(
        "Creating direct order",
        user=current_user.id,
        symbol=order_request.symbol,
        side=order_request.side,
        type=order_request.order_type,
    )
    
    try:
        # Create order through order manager
        order = await trading_engine.order_manager.create_order(
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            time_in_force=order_request.time_in_force,
            metadata={"user_id": current_user.id},
        )
        
        # Submit order
        await trading_engine._submit_order(order)
        
        return {
            "order_id": order.id,
            "status": order.status,
            "created_at": order.created_at,
        }
        
    except Exception as e:
        logger.error(f"Failed to create order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/order/{order_id}")
async def cancel_order(
    order_id: str,
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> Dict:
    """Cancel an active order."""
    logger.info(
        "Cancelling order",
        user=current_user.id,
        order_id=order_id,
    )
    
    success = await trading_engine.cancel_order(order_id)
    
    if success:
        return {"status": "cancelled", "order_id": order_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to cancel order",
        )


@router.get("/portfolio", response_model=PortfolioStatus)
async def get_portfolio_status(
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> PortfolioStatus:
    """Get current portfolio status including positions and P&L."""
    try:
        status = await trading_engine.get_portfolio_status()
        return PortfolioStatus(**status)
        
    except Exception as e:
        logger.error(f"Failed to get portfolio status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio status",
        )


@router.post("/close-all")
async def close_all_positions(
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> Dict:
    """
    Close all open positions.
    
    Emergency endpoint to close all positions at market price.
    """
    logger.warning(
        "Closing all positions",
        user=current_user.id,
    )
    
    try:
        await trading_engine._close_all_positions()
        return {
            "status": "success",
            "message": "All positions closed",
        }
        
    except Exception as e:
        logger.error(f"Failed to close positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close positions",
        )


@router.get("/performance")
async def get_performance_metrics(
    days: int = 30,
    trading_engine: TradingEngine = Depends(get_trading_engine),
    current_user = Depends(get_current_user),
) -> Dict:
    """Get trading performance metrics."""
    # This would calculate various performance metrics
    # like Sharpe ratio, win rate, average P&L, etc.
    
    return {
        "period_days": days,
        "total_trades": 150,
        "win_rate": 0.65,
        "average_pnl": 2.5,
        "sharpe_ratio": 1.8,
        "max_drawdown": -5.2,
        "total_pnl": 375.0,
    }