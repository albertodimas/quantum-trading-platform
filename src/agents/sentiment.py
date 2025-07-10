"""
Sentiment Analysis Agent for news and social media.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import re
from collections import defaultdict

from src.agents.base import BaseAgent
from src.core.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalysisAgent(BaseAgent):
    """
    Agent specialized in analyzing market sentiment from news and social media.
    
    Uses NLP to analyze text data and gauge market sentiment for
    trading decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sentiment analysis agent."""
        super().__init__("SentimentAnalysisAgent", config)
        
        # Sentiment keywords and weights
        self.sentiment_keywords = {
            "bullish": {
                "keywords": ["buy", "bullish", "moon", "pump", "breaking out", "support", 
                           "accumulation", "golden cross", "breakout", "surge", "rally",
                           "all-time high", "ath", "hodl", "diamond hands", "to the moon"],
                "weight": 1.0
            },
            "bearish": {
                "keywords": ["sell", "bearish", "dump", "crash", "resistance", "distribution",
                           "death cross", "breakdown", "plunge", "collapse", "bear market",
                           "panic", "capitulation", "weak hands", "rekt"],
                "weight": -1.0
            },
            "neutral": {
                "keywords": ["consolidation", "sideways", "ranging", "undecided", "waiting",
                           "observing", "monitoring", "stable", "flat"],
                "weight": 0.0
            }
        }
        
        # Source credibility scores
        self.source_credibility = {
            "reuters": 0.95,
            "bloomberg": 0.95,
            "coindesk": 0.85,
            "cointelegraph": 0.80,
            "twitter_verified": 0.70,
            "twitter": 0.50,
            "reddit": 0.60,
            "telegram": 0.40,
        }
        
        # Sentiment decay factor (older news have less impact)
        self.decay_factor = config.get("decay_factor", 0.95)
        
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from news and social media data.
        
        Args:
            data: News items and social media posts
            
        Returns:
            Sentiment analysis results
        """
        if not await self.validate_data(data):
            return {"error": "Invalid data provided"}
        
        try:
            # Extract news and social data
            news_items = data.get("news", [])
            social_posts = data.get("social", [])
            symbol = data.get("symbol", "")
            
            # Analyze news sentiment
            news_sentiment = await self._analyze_news_sentiment(news_items, symbol)
            
            # Analyze social sentiment
            social_sentiment = await self._analyze_social_sentiment(social_posts, symbol)
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(news_items, social_posts)
            
            # Detect sentiment trends
            trends = self._detect_sentiment_trends(news_items + social_posts)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(
                news_sentiment, social_sentiment, volume_metrics
            )
            
            # Generate analysis
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "overall_sentiment": overall_sentiment,
                "volume_metrics": volume_metrics,
                "trends": trends,
                "signal_strength": abs(overall_sentiment["score"]),
                "confidence": overall_sentiment["confidence"],
            }
            
            await self._store_analysis(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    async def get_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from sentiment analysis."""
        signals = []
        
        if "error" in analysis:
            return signals
        
        overall = analysis.get("overall_sentiment", {})
        score = overall.get("score", 0)
        confidence = overall.get("confidence", 0)
        
        # Only generate signals above confidence threshold
        if confidence < self._confidence_threshold:
            return signals
        
        # Generate signals based on sentiment
        if score > 0.3:  # Bullish sentiment
            signals.append({
                "type": "sentiment",
                "action": "buy",
                "symbol": analysis["symbol"],
                "confidence": confidence,
                "sentiment_score": score,
                "reasons": self._get_sentiment_reasons(analysis, "bullish"),
                "sources": self._get_top_sources(analysis),
                "time_horizon": "short",  # Sentiment is usually short-term
            })
        elif score < -0.3:  # Bearish sentiment
            signals.append({
                "type": "sentiment",
                "action": "sell",
                "symbol": analysis["symbol"],
                "confidence": confidence,
                "sentiment_score": score,
                "reasons": self._get_sentiment_reasons(analysis, "bearish"),
                "sources": self._get_top_sources(analysis),
                "time_horizon": "short",
            })
        
        return signals
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for sentiment analysis."""
        return ["symbol"]
    
    async def _analyze_news_sentiment(self, news_items: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not news_items:
            return {"score": 0, "count": 0, "articles": []}
        
        sentiments = []
        analyzed_articles = []
        
        for item in news_items:
            # Extract relevant fields
            title = item.get("title", "")
            content = item.get("content", "")
            source = item.get("source", "unknown")
            published = item.get("published_at", datetime.utcnow().isoformat())
            
            # Skip if not relevant to symbol
            if symbol.lower() not in (title + content).lower():
                continue
            
            # Analyze sentiment
            sentiment = self._analyze_text_sentiment(title + " " + content)
            
            # Apply source credibility
            credibility = self.source_credibility.get(source.lower(), 0.5)
            weighted_sentiment = sentiment * credibility
            
            # Apply time decay
            age_hours = (datetime.utcnow() - datetime.fromisoformat(published)).total_seconds() / 3600
            decay = self.decay_factor ** (age_hours / 24)  # Daily decay
            final_sentiment = weighted_sentiment * decay
            
            sentiments.append(final_sentiment)
            analyzed_articles.append({
                "title": title[:100],
                "source": source,
                "sentiment": sentiment,
                "credibility": credibility,
                "age_hours": age_hours,
                "final_score": final_sentiment,
            })
        
        # Calculate aggregate sentiment
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                "score": avg_sentiment,
                "count": len(sentiments),
                "articles": sorted(analyzed_articles, 
                                 key=lambda x: abs(x["final_score"]), 
                                 reverse=True)[:5],  # Top 5
            }
        
        return {"score": 0, "count": 0, "articles": []}
    
    async def _analyze_social_sentiment(self, social_posts: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from social media posts."""
        if not social_posts:
            return {"score": 0, "count": 0, "posts": []}
        
        sentiments = []
        sentiment_distribution = defaultdict(int)
        influential_posts = []
        
        for post in social_posts:
            # Extract fields
            text = post.get("text", "")
            platform = post.get("platform", "unknown")
            author = post.get("author", {})
            metrics = post.get("metrics", {})
            created = post.get("created_at", datetime.utcnow().isoformat())
            
            # Skip if not relevant
            if symbol.lower() not in text.lower():
                continue
            
            # Analyze sentiment
            sentiment = self._analyze_text_sentiment(text)
            
            # Calculate influence score
            influence = self._calculate_influence_score(author, metrics, platform)
            
            # Apply time decay
            age_hours = (datetime.utcnow() - datetime.fromisoformat(created)).total_seconds() / 3600
            decay = self.decay_factor ** (age_hours / 12)  # Faster decay for social
            
            final_sentiment = sentiment * influence * decay
            sentiments.append(final_sentiment)
            
            # Track distribution
            if sentiment > 0.3:
                sentiment_distribution["bullish"] += 1
            elif sentiment < -0.3:
                sentiment_distribution["bearish"] += 1
            else:
                sentiment_distribution["neutral"] += 1
            
            # Store influential posts
            if influence > 0.7:
                influential_posts.append({
                    "text": text[:200],
                    "platform": platform,
                    "author": author.get("username", "unknown"),
                    "sentiment": sentiment,
                    "influence": influence,
                    "engagement": metrics.get("engagement_rate", 0),
                })
        
        # Calculate aggregate
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                "score": avg_sentiment,
                "count": len(sentiments),
                "distribution": dict(sentiment_distribution),
                "influential_posts": sorted(influential_posts, 
                                          key=lambda x: x["influence"], 
                                          reverse=True)[:5],
            }
        
        return {"score": 0, "count": 0, "distribution": {}, "influential_posts": []}
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a text using keyword matching."""
        text_lower = text.lower()
        
        # Remove noise
        text_lower = re.sub(r'[^\w\s]', ' ', text_lower)
        
        # Count sentiment keywords
        bullish_count = 0
        bearish_count = 0
        
        for keyword in self.sentiment_keywords["bullish"]["keywords"]:
            bullish_count += text_lower.count(keyword)
        
        for keyword in self.sentiment_keywords["bearish"]["keywords"]:
            bearish_count += text_lower.count(keyword)
        
        # Calculate sentiment score
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0
        
        sentiment = (bullish_count - bearish_count) / total_keywords
        
        # Apply sigmoid to bound between -1 and 1
        import math
        return 2 / (1 + math.exp(-sentiment * 2)) - 1
    
    def _calculate_influence_score(self, author: Dict, metrics: Dict, platform: str) -> float:
        """Calculate influence score of a social media author."""
        influence = 0.5  # Base score
        
        # Platform base scores
        platform_scores = {
            "twitter": 0.6,
            "reddit": 0.5,
            "telegram": 0.4,
            "discord": 0.4,
        }
        influence = platform_scores.get(platform.lower(), 0.5)
        
        # Author metrics
        followers = author.get("followers", 0)
        if followers > 100000:
            influence += 0.3
        elif followers > 10000:
            influence += 0.2
        elif followers > 1000:
            influence += 0.1
        
        # Engagement metrics
        engagement_rate = metrics.get("engagement_rate", 0)
        if engagement_rate > 0.1:  # 10% engagement
            influence += 0.2
        elif engagement_rate > 0.05:
            influence += 0.1
        
        # Verification status
        if author.get("verified", False):
            influence += 0.2
        
        return min(influence, 1.0)
    
    def _calculate_volume_metrics(self, news_items: List[Dict], social_posts: List[Dict]) -> Dict[str, Any]:
        """Calculate volume and velocity metrics."""
        all_items = news_items + social_posts
        
        if not all_items:
            return {"total_mentions": 0, "velocity": 0, "acceleration": 0}
        
        # Time buckets (hourly)
        time_buckets = defaultdict(int)
        now = datetime.utcnow()
        
        for item in all_items:
            published = item.get("published_at") or item.get("created_at")
            if published:
                item_time = datetime.fromisoformat(published)
                hours_ago = int((now - item_time).total_seconds() / 3600)
                if hours_ago < 24:  # Last 24 hours
                    time_buckets[hours_ago] += 1
        
        # Calculate metrics
        total_mentions = sum(time_buckets.values())
        
        # Velocity (mentions per hour in last 6 hours)
        recent_mentions = sum(time_buckets[i] for i in range(6))
        velocity = recent_mentions / 6 if recent_mentions > 0 else 0
        
        # Acceleration (change in velocity)
        first_half = sum(time_buckets[i] for i in range(3, 6))
        second_half = sum(time_buckets[i] for i in range(0, 3))
        acceleration = (second_half - first_half) / 3 if first_half > 0 else 0
        
        return {
            "total_mentions": total_mentions,
            "velocity": velocity,
            "acceleration": acceleration,
            "trending": acceleration > 0.5,
        }
    
    def _detect_sentiment_trends(self, items: List[Dict]) -> Dict[str, Any]:
        """Detect trends in sentiment over time."""
        if not items:
            return {"trend": "neutral", "strength": 0}
        
        # Sort by time
        sorted_items = sorted(items, 
                            key=lambda x: x.get("published_at") or x.get("created_at", ""),
                            reverse=True)
        
        # Analyze sentiment progression
        recent_sentiments = []
        older_sentiments = []
        
        for item in sorted_items[:20]:  # Last 20 items
            text = item.get("title", "") + item.get("content", "") + item.get("text", "")
            sentiment = self._analyze_text_sentiment(text)
            
            if len(recent_sentiments) < 10:
                recent_sentiments.append(sentiment)
            else:
                older_sentiments.append(sentiment)
        
        if recent_sentiments and older_sentiments:
            recent_avg = sum(recent_sentiments) / len(recent_sentiments)
            older_avg = sum(older_sentiments) / len(older_sentiments)
            
            trend_diff = recent_avg - older_avg
            
            if trend_diff > 0.2:
                return {"trend": "improving", "strength": min(trend_diff, 1.0)}
            elif trend_diff < -0.2:
                return {"trend": "deteriorating", "strength": min(abs(trend_diff), 1.0)}
        
        return {"trend": "neutral", "strength": 0}
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, social_sentiment: Dict, 
                                   volume_metrics: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment score."""
        # Weight factors
        news_weight = 0.6
        social_weight = 0.4
        
        # Adjust weights based on volume
        if volume_metrics["total_mentions"] > 100:
            social_weight = 0.5
            news_weight = 0.5
        
        # Calculate weighted score
        news_score = news_sentiment.get("score", 0)
        social_score = social_sentiment.get("score", 0)
        
        weighted_score = (news_score * news_weight + social_score * social_weight)
        
        # Boost score if trending
        if volume_metrics.get("trending", False):
            weighted_score *= 1.2
        
        # Calculate confidence
        total_items = news_sentiment.get("count", 0) + social_sentiment.get("count", 0)
        confidence = min(total_items / 20, 1.0)  # Max confidence at 20+ items
        
        # Adjust confidence based on agreement
        if news_score * social_score > 0:  # Same direction
            confidence *= 1.1
        else:  # Conflicting signals
            confidence *= 0.8
        
        return {
            "score": max(-1, min(1, weighted_score)),
            "confidence": min(confidence, 1.0),
            "direction": "bullish" if weighted_score > 0 else "bearish" if weighted_score < 0 else "neutral",
        }
    
    def _get_sentiment_reasons(self, analysis: Dict[str, Any], direction: str) -> List[str]:
        """Get reasons for sentiment signal."""
        reasons = []
        
        news = analysis.get("news_sentiment", {})
        social = analysis.get("social_sentiment", {})
        volume = analysis.get("volume_metrics", {})
        trends = analysis.get("trends", {})
        
        if direction == "bullish":
            if news.get("score", 0) > 0.3:
                reasons.append(f"Positive news sentiment ({news['count']} articles)")
            if social.get("score", 0) > 0.3:
                reasons.append(f"Bullish social sentiment ({social['count']} posts)")
            if volume.get("trending", False):
                reasons.append("Trending topic with increasing mentions")
            if trends.get("trend") == "improving":
                reasons.append("Improving sentiment trend")
        else:
            if news.get("score", 0) < -0.3:
                reasons.append(f"Negative news sentiment ({news['count']} articles)")
            if social.get("score", 0) < -0.3:
                reasons.append(f"Bearish social sentiment ({social['count']} posts)")
            if trends.get("trend") == "deteriorating":
                reasons.append("Deteriorating sentiment trend")
        
        return reasons
    
    def _get_top_sources(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top influential sources from analysis."""
        sources = []
        
        # Add top news articles
        news_articles = analysis.get("news_sentiment", {}).get("articles", [])
        for article in news_articles[:3]:
            sources.append({
                "type": "news",
                "title": article.get("title"),
                "source": article.get("source"),
                "sentiment": article.get("sentiment"),
            })
        
        # Add top social posts
        social_posts = analysis.get("social_sentiment", {}).get("influential_posts", [])
        for post in social_posts[:2]:
            sources.append({
                "type": "social",
                "platform": post.get("platform"),
                "author": post.get("author"),
                "sentiment": post.get("sentiment"),
            })
        
        return sources