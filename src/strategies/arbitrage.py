"""
Arbitrage trading strategy implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from src.strategies.base import BaseStrategy
from src.core.logging import get_logger
from src.trading.models import Signal, OrderSide

logger = get_logger(__name__)


class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage trading strategy.
    
    Identifies and exploits price inefficiencies across markets,
    exchanges, or related assets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize arbitrage strategy."""
        super().__init__("ArbitrageStrategy", config)
        
        # Track arbitrage opportunities
        self.opportunities = defaultdict(list)
        self.executed_arbs = []
    
    def _initialize_parameters(self):
        """Initialize arbitrage-specific parameters."""
        # Arbitrage types
        self.enable_triangular = self.config.get("enable_triangular", True)
        self.enable_statistical = self.config.get("enable_statistical", True)
        self.enable_cross_exchange = self.config.get("enable_cross_exchange", True)
        
        # Thresholds
        self.min_profit_pct = self.config.get("min_profit_pct", 0.002)  # 0.2% minimum
        self.max_exposure_pct = self.config.get("max_exposure_pct", 0.3)  # 30% of capital
        
        # Statistical arbitrage
        self.lookback_period = self.config.get("lookback_period", 100)
        self.zscore_entry = self.config.get("zscore_entry", 2.0)
        self.zscore_exit = self.config.get("zscore_exit", 0.5)
        self.cointegration_pvalue = self.config.get("cointegration_pvalue", 0.05)
        
        # Triangular arbitrage
        self.triangular_pairs = self.config.get("triangular_pairs", [
            ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
            ["BTC/USDT", "BNB/BTC", "BNB/USDT"],
        ])
        
        # Cross-exchange arbitrage
        self.min_exchange_spread = self.config.get("min_exchange_spread", 0.003)  # 0.3%
        self.transfer_time = self.config.get("transfer_time", 300)  # 5 minutes
        self.network_fees = self.config.get("network_fees", {
            "BTC": 0.0005,
            "ETH": 0.005,
            "USDT": 1.0,
        })
        
        # Risk parameters
        self.max_slippage = self.config.get("max_slippage", 0.001)  # 0.1%
        self.execution_timeout = self.config.get("execution_timeout", 5)  # seconds
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for arbitrage opportunities.
        
        Args:
            market_data: Multi-asset/exchange market data
            
        Returns:
            Arbitrage analysis results
        """
        try:
            # Extract market data
            assets_data = market_data.get("assets", {})
            exchanges_data = market_data.get("exchanges", {})
            
            if not assets_data and not exchanges_data:
                return {"error": "No market data provided"}
            
            analysis_results = {}
            
            # Triangular arbitrage analysis
            if self.enable_triangular and assets_data:
                triangular_opps = await self._analyze_triangular_arbitrage(assets_data)
                analysis_results["triangular"] = triangular_opps
            
            # Statistical arbitrage analysis
            if self.enable_statistical and assets_data:
                stat_arb_opps = await self._analyze_statistical_arbitrage(assets_data)
                analysis_results["statistical"] = stat_arb_opps
            
            # Cross-exchange arbitrage analysis
            if self.enable_cross_exchange and exchanges_data:
                cross_exchange_opps = await self._analyze_cross_exchange_arbitrage(exchanges_data)
                analysis_results["cross_exchange"] = cross_exchange_opps
            
            # Filter and rank opportunities
            all_opportunities = self._aggregate_opportunities(analysis_results)
            ranked_opportunities = self._rank_opportunities(all_opportunities)
            
            # Calculate overall metrics
            total_opportunities = len(all_opportunities)
            viable_opportunities = len([o for o in all_opportunities if o.get("viable", False)])
            
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_results": analysis_results,
                "opportunities": ranked_opportunities[:10],  # Top 10
                "total_opportunities": total_opportunities,
                "viable_opportunities": viable_opportunities,
                "best_opportunity": ranked_opportunities[0] if ranked_opportunities else None,
                "signal_strength": self._calculate_signal_strength(ranked_opportunities),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Arbitrage analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals from arbitrage analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        opportunities = analysis.get("opportunities", [])
        
        for opp in opportunities[:3]:  # Top 3 opportunities
            if not opp.get("viable", False):
                continue
            
            confidence = opp.get("confidence", 0)
            if confidence < self.min_confidence:
                continue
            
            # Generate signals based on opportunity type
            if opp["type"] == "triangular":
                signals.extend(self._generate_triangular_signals(opp))
            elif opp["type"] == "statistical":
                signals.extend(self._generate_statistical_signals(opp))
            elif opp["type"] == "cross_exchange":
                signals.extend(self._generate_cross_exchange_signals(opp))
        
        return signals
    
    async def _analyze_triangular_arbitrage(self, assets_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze triangular arbitrage opportunities."""
        opportunities = []
        
        for triangle in self.triangular_pairs:
            if len(triangle) != 3:
                continue
            
            # Get prices for the triangle
            prices = {}
            for pair in triangle:
                if pair in assets_data:
                    prices[pair] = assets_data[pair].get("price", 0)
            
            if len(prices) != 3:
                continue
            
            # Calculate arbitrage
            arb_result = self._calculate_triangular_arbitrage(triangle, prices)
            
            if arb_result["profit_pct"] > self.min_profit_pct:
                opportunities.append({
                    "triangle": triangle,
                    "prices": prices,
                    "profit_pct": arb_result["profit_pct"],
                    "direction": arb_result["direction"],
                    "amounts": arb_result["amounts"],
                    "viable": True,
                })
        
        return {
            "count": len(opportunities),
            "opportunities": opportunities,
            "best": max(opportunities, key=lambda x: x["profit_pct"]) if opportunities else None,
        }
    
    def _calculate_triangular_arbitrage(self, triangle: List[str], 
                                      prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate triangular arbitrage profit."""
        # Example: BTC/USDT, ETH/BTC, ETH/USDT
        # Path 1: USDT -> BTC -> ETH -> USDT
        # Path 2: USDT -> ETH -> BTC -> USDT
        
        pair1, pair2, pair3 = triangle
        price1 = prices[pair1]  # BTC/USDT
        price2 = prices[pair2]  # ETH/BTC
        price3 = prices[pair3]  # ETH/USDT
        
        # Forward path: USDT -> BTC -> ETH -> USDT
        # 1 USDT -> 1/price1 BTC -> (1/price1)*price2 ETH -> (1/price1)*price2*price3 USDT
        forward_result = (1 / price1) * price2 * price3
        
        # Reverse path: USDT -> ETH -> BTC -> USDT
        # 1 USDT -> 1/price3 ETH -> (1/price3)/price2 BTC -> (1/price3)/price2*price1 USDT
        reverse_result = (1 / price3) / price2 * price1
        
        # Calculate profit
        forward_profit = forward_result - 1
        reverse_profit = reverse_result - 1
        
        if forward_profit > reverse_profit:
            return {
                "profit_pct": forward_profit,
                "direction": "forward",
                "amounts": {
                    "step1": 1000,  # Start with 1000 USDT
                    "step2": 1000 / price1,  # BTC amount
                    "step3": (1000 / price1) * price2,  # ETH amount
                    "final": (1000 / price1) * price2 * price3,  # Final USDT
                }
            }
        else:
            return {
                "profit_pct": reverse_profit,
                "direction": "reverse",
                "amounts": {
                    "step1": 1000,  # Start with 1000 USDT
                    "step2": 1000 / price3,  # ETH amount
                    "step3": (1000 / price3) / price2,  # BTC amount
                    "final": (1000 / price3) / price2 * price1,  # Final USDT
                }
            }
    
    async def _analyze_statistical_arbitrage(self, assets_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical arbitrage opportunities."""
        opportunities = []
        
        # Find cointegrated pairs
        pairs = self._find_cointegrated_pairs(assets_data)
        
        for pair_info in pairs:
            asset1, asset2 = pair_info["pair"]
            
            # Get price data
            if asset1 not in assets_data or asset2 not in assets_data:
                continue
            
            # Calculate spread and z-score
            spread_analysis = self._analyze_spread(
                assets_data[asset1], 
                assets_data[asset2],
                pair_info["hedge_ratio"]
            )
            
            # Check for entry signals
            if abs(spread_analysis["z_score"]) > self.zscore_entry:
                opportunities.append({
                    "pair": [asset1, asset2],
                    "hedge_ratio": pair_info["hedge_ratio"],
                    "z_score": spread_analysis["z_score"],
                    "spread": spread_analysis["spread"],
                    "mean": spread_analysis["mean"],
                    "std": spread_analysis["std"],
                    "signal": "long_spread" if spread_analysis["z_score"] < 0 else "short_spread",
                    "expected_profit": abs(spread_analysis["z_score"]) * spread_analysis["std"],
                    "viable": True,
                })
        
        return {
            "count": len(opportunities),
            "opportunities": opportunities,
            "best": max(opportunities, key=lambda x: x["expected_profit"]) if opportunities else None,
        }
    
    def _find_cointegrated_pairs(self, assets_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find cointegrated pairs for statistical arbitrage."""
        cointegrated_pairs = []
        
        # Get assets with sufficient history
        valid_assets = []
        for asset, data in assets_data.items():
            if "ohlcv" in data and len(data["ohlcv"]) >= self.lookback_period:
                valid_assets.append(asset)
        
        # Test pairs for cointegration
        for i in range(len(valid_assets)):
            for j in range(i + 1, len(valid_assets)):
                asset1, asset2 = valid_assets[i], valid_assets[j]
                
                # Get price series
                prices1 = pd.DataFrame(assets_data[asset1]["ohlcv"])["close"]
                prices2 = pd.DataFrame(assets_data[asset2]["ohlcv"])["close"]
                
                # Align series
                min_len = min(len(prices1), len(prices2))
                prices1 = prices1.iloc[-min_len:]
                prices2 = prices2.iloc[-min_len:]
                
                # Test cointegration (simplified)
                coint_result = self._test_cointegration(prices1, prices2)
                
                if coint_result["p_value"] < self.cointegration_pvalue:
                    cointegrated_pairs.append({
                        "pair": [asset1, asset2],
                        "p_value": coint_result["p_value"],
                        "hedge_ratio": coint_result["hedge_ratio"],
                    })
        
        return cointegrated_pairs
    
    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Test cointegration between two price series (simplified)."""
        # Use OLS regression to find hedge ratio
        x = series1.values
        y = series2.values
        
        # Add constant
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        # OLS regression
        try:
            coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            hedge_ratio = coeffs[1]
            
            # Calculate residuals
            residuals = y - (coeffs[0] + hedge_ratio * x)
            
            # Test stationarity of residuals (simplified ADF test)
            # Using autocorrelation as proxy
            autocorr = pd.Series(residuals).autocorr()
            
            # Simplified p-value calculation
            p_value = abs(autocorr)  # Lower is better
            
            return {
                "p_value": p_value,
                "hedge_ratio": hedge_ratio,
                "stationary": p_value < 0.1,
            }
        except:
            return {
                "p_value": 1.0,
                "hedge_ratio": 1.0,
                "stationary": False,
            }
    
    def _analyze_spread(self, asset1_data: Dict[str, Any], 
                       asset2_data: Dict[str, Any],
                       hedge_ratio: float) -> Dict[str, Any]:
        """Analyze spread between two assets."""
        # Get current prices
        price1 = asset1_data.get("price", 0)
        price2 = asset2_data.get("price", 0)
        
        # Calculate current spread
        current_spread = price2 - hedge_ratio * price1
        
        # Get historical spreads
        if "ohlcv" in asset1_data and "ohlcv" in asset2_data:
            prices1 = pd.DataFrame(asset1_data["ohlcv"])["close"]
            prices2 = pd.DataFrame(asset2_data["ohlcv"])["close"]
            
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1.iloc[-min_len:]
            prices2 = prices2.iloc[-min_len:]
            
            spreads = prices2 - hedge_ratio * prices1
            
            mean_spread = spreads.mean()
            std_spread = spreads.std()
            
            # Calculate z-score
            z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
        else:
            mean_spread = current_spread
            std_spread = 0
            z_score = 0
        
        return {
            "spread": current_spread,
            "mean": mean_spread,
            "std": std_spread,
            "z_score": z_score,
        }
    
    async def _analyze_cross_exchange_arbitrage(self, exchanges_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-exchange arbitrage opportunities."""
        opportunities = []
        
        # Group by asset
        asset_prices = defaultdict(dict)
        for exchange, data in exchanges_data.items():
            for asset, price_data in data.items():
                asset_prices[asset][exchange] = price_data
        
        # Find arbitrage opportunities
        for asset, exchange_prices in asset_prices.items():
            if len(exchange_prices) < 2:
                continue
            
            # Find best bid and ask across exchanges
            best_bid = {"exchange": None, "price": 0}
            best_ask = {"exchange": None, "price": float('inf')}
            
            for exchange, price_data in exchange_prices.items():
                bid = price_data.get("bid", 0)
                ask = price_data.get("ask", float('inf'))
                
                if bid > best_bid["price"]:
                    best_bid = {"exchange": exchange, "price": bid}
                
                if ask < best_ask["price"]:
                    best_ask = {"exchange": exchange, "price": ask}
            
            # Calculate spread
            if best_bid["price"] > 0 and best_ask["price"] < float('inf'):
                spread_pct = (best_bid["price"] - best_ask["price"]) / best_ask["price"]
                
                # Account for fees
                base_asset = asset.split("/")[0]
                network_fee = self.network_fees.get(base_asset, 0)
                
                # Estimate profit after fees
                profit_pct = spread_pct - network_fee / best_ask["price"]
                
                if profit_pct > self.min_exchange_spread:
                    opportunities.append({
                        "asset": asset,
                        "buy_exchange": best_ask["exchange"],
                        "sell_exchange": best_bid["exchange"],
                        "buy_price": best_ask["price"],
                        "sell_price": best_bid["price"],
                        "spread_pct": spread_pct,
                        "profit_pct": profit_pct,
                        "network_fee": network_fee,
                        "viable": True,
                    })
        
        return {
            "count": len(opportunities),
            "opportunities": opportunities,
            "best": max(opportunities, key=lambda x: x["profit_pct"]) if opportunities else None,
        }
    
    def _aggregate_opportunities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate all arbitrage opportunities."""
        all_opportunities = []
        
        # Triangular opportunities
        if "triangular" in analysis_results:
            for opp in analysis_results["triangular"]["opportunities"]:
                all_opportunities.append({
                    "type": "triangular",
                    "profit_pct": opp["profit_pct"],
                    "confidence": self._calculate_triangular_confidence(opp),
                    "data": opp,
                    "viable": opp.get("viable", False),
                })
        
        # Statistical arbitrage opportunities
        if "statistical" in analysis_results:
            for opp in analysis_results["statistical"]["opportunities"]:
                all_opportunities.append({
                    "type": "statistical",
                    "profit_pct": opp["expected_profit"] / 1000,  # Normalize
                    "confidence": self._calculate_statistical_confidence(opp),
                    "data": opp,
                    "viable": opp.get("viable", False),
                })
        
        # Cross-exchange opportunities
        if "cross_exchange" in analysis_results:
            for opp in analysis_results["cross_exchange"]["opportunities"]:
                all_opportunities.append({
                    "type": "cross_exchange",
                    "profit_pct": opp["profit_pct"],
                    "confidence": self._calculate_cross_exchange_confidence(opp),
                    "data": opp,
                    "viable": opp.get("viable", False),
                })
        
        return all_opportunities
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank arbitrage opportunities by attractiveness."""
        # Score each opportunity
        for opp in opportunities:
            # Base score from profit
            profit_score = min(opp["profit_pct"] / 0.01, 1.0)  # Normalize to 1% max
            
            # Confidence score
            confidence_score = opp["confidence"]
            
            # Type preference (triangular is fastest)
            type_scores = {
                "triangular": 1.0,
                "cross_exchange": 0.7,
                "statistical": 0.5,
            }
            type_score = type_scores.get(opp["type"], 0.5)
            
            # Combined score
            opp["score"] = (profit_score * 0.4 + confidence_score * 0.4 + type_score * 0.2)
        
        # Sort by score
        return sorted(opportunities, key=lambda x: x["score"], reverse=True)
    
    def _calculate_signal_strength(self, opportunities: List[Dict[str, Any]]) -> float:
        """Calculate overall signal strength."""
        if not opportunities:
            return 0
        
        # Get top opportunity score
        top_score = opportunities[0]["score"] if opportunities else 0
        
        # Count viable opportunities
        viable_count = len([o for o in opportunities if o.get("viable", False)])
        
        # Signal strength based on best opportunity and count
        strength = top_score * 0.7 + min(viable_count / 5, 1.0) * 0.3
        
        return strength
    
    def _calculate_triangular_confidence(self, opp: Dict[str, Any]) -> float:
        """Calculate confidence for triangular arbitrage."""
        base_confidence = 0.8  # High confidence due to instant execution
        
        # Adjust for profit magnitude
        if opp["profit_pct"] > 0.01:  # > 1%
            base_confidence *= 1.2
        elif opp["profit_pct"] < 0.003:  # < 0.3%
            base_confidence *= 0.8
        
        return min(base_confidence, 0.95)
    
    def _calculate_statistical_confidence(self, opp: Dict[str, Any]) -> float:
        """Calculate confidence for statistical arbitrage."""
        base_confidence = 0.6  # Lower due to execution risk
        
        # Adjust for z-score
        z_score = abs(opp["z_score"])
        if z_score > 3:
            base_confidence *= 1.3
        elif z_score < 2:
            base_confidence *= 0.8
        
        return min(base_confidence, 0.85)
    
    def _calculate_cross_exchange_confidence(self, opp: Dict[str, Any]) -> float:
        """Calculate confidence for cross-exchange arbitrage."""
        base_confidence = 0.5  # Lowest due to transfer risk
        
        # Adjust for spread size
        if opp["spread_pct"] > 0.02:  # > 2%
            base_confidence *= 1.4
        elif opp["spread_pct"] < 0.005:  # < 0.5%
            base_confidence *= 0.7
        
        return min(base_confidence, 0.8)
    
    def _generate_triangular_signals(self, opportunity: Dict[str, Any]) -> List[Signal]:
        """Generate signals for triangular arbitrage."""
        signals = []
        data = opportunity["data"]
        triangle = data["triangle"]
        direction = data["direction"]
        
        if direction == "forward":
            # Step 1: Buy BTC with USDT
            signals.append(Signal(
                symbol=triangle[0],
                side=OrderSide.BUY,
                confidence=opportunity["confidence"],
                strategy=self.name,
                metadata={
                    "arbitrage_type": "triangular",
                    "step": 1,
                    "triangle": triangle,
                    "expected_profit": data["profit_pct"],
                }
            ))
            
            # Step 2: Buy ETH with BTC
            signals.append(Signal(
                symbol=triangle[1],
                side=OrderSide.BUY,
                confidence=opportunity["confidence"],
                strategy=self.name,
                metadata={
                    "arbitrage_type": "triangular",
                    "step": 2,
                    "triangle": triangle,
                }
            ))
            
            # Step 3: Sell ETH for USDT
            signals.append(Signal(
                symbol=triangle[2],
                side=OrderSide.SELL,
                confidence=opportunity["confidence"],
                strategy=self.name,
                metadata={
                    "arbitrage_type": "triangular",
                    "step": 3,
                    "triangle": triangle,
                }
            ))
        
        return signals
    
    def _generate_statistical_signals(self, opportunity: Dict[str, Any]) -> List[Signal]:
        """Generate signals for statistical arbitrage."""
        signals = []
        data = opportunity["data"]
        
        if data["signal"] == "long_spread":
            # Buy undervalued asset
            signals.append(Signal(
                symbol=data["pair"][1],
                side=OrderSide.BUY,
                confidence=opportunity["confidence"],
                strategy=self.name,
                metadata={
                    "arbitrage_type": "statistical",
                    "pair": data["pair"],
                    "z_score": data["z_score"],
                    "hedge_ratio": data["hedge_ratio"],
                }
            ))
            
            # Sell overvalued asset (hedge)
            signals.append(Signal(
                symbol=data["pair"][0],
                side=OrderSide.SELL,
                confidence=opportunity["confidence"],
                strategy=self.name,
                metadata={
                    "arbitrage_type": "statistical",
                    "pair": data["pair"],
                    "hedge": True,
                }
            ))
        
        return signals
    
    def _generate_cross_exchange_signals(self, opportunity: Dict[str, Any]) -> List[Signal]:
        """Generate signals for cross-exchange arbitrage."""
        signals = []
        data = opportunity["data"]
        
        # Buy on cheaper exchange
        signals.append(Signal(
            symbol=data["asset"],
            side=OrderSide.BUY,
            confidence=opportunity["confidence"],
            entry_price=data["buy_price"],
            strategy=self.name,
            metadata={
                "arbitrage_type": "cross_exchange",
                "exchange": data["buy_exchange"],
                "expected_profit": data["profit_pct"],
            }
        ))
        
        # Sell on expensive exchange (after transfer)
        signals.append(Signal(
            symbol=data["asset"],
            side=OrderSide.SELL,
            confidence=opportunity["confidence"],
            entry_price=data["sell_price"],
            strategy=self.name,
            metadata={
                "arbitrage_type": "cross_exchange",
                "exchange": data["sell_exchange"],
                "delay": self.transfer_time,
            }
        ))
        
        return signals