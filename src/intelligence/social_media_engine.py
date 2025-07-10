"""
🌐 Social Media Intelligence Engine
Análisis avanzado de sentimientos y tendencias en redes sociales para trading
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

import aiohttp
import asyncpg
from textblob import TextBlob
import tweepy
import praw
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class InfluenceLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CELEBRITY = 4
    WHALE = 5

@dataclass
class SocialMediaPost:
    """Estructura de un post de redes sociales"""
    id: str
    platform: str
    author: str
    content: str
    timestamp: datetime
    likes: int
    shares: int
    comments: int
    author_followers: int
    hashtags: List[str]
    mentions: List[str]
    url: Optional[str] = None
    image_urls: List[str] = None
    video_urls: List[str] = None

@dataclass
class SentimentAnalysis:
    """Resultado del análisis de sentimiento"""
    score: float  # -1 a 1
    confidence: float  # 0 a 1
    label: SentimentScore
    emotions: Dict[str, float]  # fear, greed, excitement, etc.
    keywords: List[str]
    impact_score: float  # Combinación de sentimiento e influencia

@dataclass
class InfluencerProfile:
    """Perfil de influencer"""
    username: str
    platform: str
    followers: int
    influence_level: InfluenceLevel
    credibility_score: float  # 0 a 1
    specialization: List[str]  # crypto, stocks, forex, etc.
    avg_engagement: float
    prediction_accuracy: float
    last_updated: datetime

@dataclass
class TrendingTopic:
    """Tema trending"""
    keyword: str
    platforms: List[str]
    mentions_count: int
    sentiment_avg: float
    sentiment_std: float
    growth_rate: float  # Velocidad de crecimiento
    related_symbols: List[str]
    influence_score: float
    first_detected: datetime

class SocialMediaIntelligenceEngine:
    """Motor de inteligencia de redes sociales para trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # APIs Configuration
        self.twitter_api = None
        self.reddit_api = None
        
        # AI Models
        self.sentiment_model = None
        self.emotion_model = None
        self.influence_classifier = None
        
        # Database
        self.db_pool = None
        
        # Cache
        self.posts_cache = {}
        self.influencers_cache = {}
        self.trends_cache = {}
        
        # Analysis parameters
        self.min_followers_threshold = config.get('min_followers', 1000)
        self.sentiment_window = config.get('sentiment_window_hours', 24)
        self.trend_detection_threshold = config.get('trend_threshold', 0.1)
        
    async def initialize(self):
        """Inicializar el motor de inteligencia"""
        logger.info("Inicializando Social Media Intelligence Engine...")
        
        # Configurar APIs
        await self._setup_apis()
        
        # Cargar modelos de IA
        await self._load_ai_models()
        
        # Conectar a base de datos
        await self._setup_database()
        
        # Cargar influencers conocidos
        await self._load_influencers()
        
        logger.info("Social Media Intelligence Engine inicializado")
    
    async def _setup_apis(self):
        """Configurar APIs de redes sociales"""
        # Twitter API v2
        if self.config.get('twitter_bearer_token'):
            self.twitter_api = tweepy.Client(
                bearer_token=self.config['twitter_bearer_token'],
                consumer_key=self.config.get('twitter_consumer_key'),
                consumer_secret=self.config.get('twitter_consumer_secret'),
                access_token=self.config.get('twitter_access_token'),
                access_token_secret=self.config.get('twitter_access_token_secret'),
                wait_on_rate_limit=True
            )
        
        # Reddit API
        if self.config.get('reddit_client_id'):
            self.reddit_api = praw.Reddit(
                client_id=self.config['reddit_client_id'],
                client_secret=self.config['reddit_client_secret'],
                user_agent=self.config.get('reddit_user_agent', 'QuantumTrading/1.0')
            )
    
    async def _load_ai_models(self):
        """Cargar modelos de IA para análisis de sentimientos"""
        try:
            # Modelo de sentimientos financieros
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Modelo de emociones
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Clasificador de influencia (entrenado con datos históricos)
            self.influence_classifier = self._create_influence_classifier()
            
            logger.info("Modelos de IA cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelos de IA: {e}")
            # Fallback a modelos básicos
            self.sentiment_model = TextBlob
    
    def _create_influence_classifier(self):
        """Crear clasificador de influencia"""
        # Este sería entrenado con datos históricos reales
        # Por ahora usamos un modelo simple
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Datos de ejemplo para entrenamiento
        # En producción, esto vendría de datos históricos
        X_example = np.random.rand(1000, 5)  # followers, engagement, mentions, etc.
        y_example = np.random.randint(1, 6, 1000)  # influence levels
        
        classifier.fit(X_example, y_example)
        return classifier
    
    async def _setup_database(self):
        """Configurar conexión a base de datos"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.config['database_url'],
                min_size=5,
                max_size=20
            )
            
            # Crear tablas si no existen
            await self._create_tables()
            
        except Exception as e:
            logger.error(f"Error configurando base de datos: {e}")
    
    async def _create_tables(self):
        """Crear tablas necesarias"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS social_posts (
                    id VARCHAR PRIMARY KEY,
                    platform VARCHAR NOT NULL,
                    author VARCHAR NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    likes INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    author_followers INTEGER DEFAULT 0,
                    hashtags JSONB,
                    mentions JSONB,
                    sentiment_score FLOAT,
                    impact_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS influencers (
                    username VARCHAR PRIMARY KEY,
                    platform VARCHAR NOT NULL,
                    followers INTEGER NOT NULL,
                    influence_level INTEGER NOT NULL,
                    credibility_score FLOAT NOT NULL,
                    specialization JSONB,
                    avg_engagement FLOAT DEFAULT 0,
                    prediction_accuracy FLOAT DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS trending_topics (
                    keyword VARCHAR PRIMARY KEY,
                    platforms JSONB NOT NULL,
                    mentions_count INTEGER NOT NULL,
                    sentiment_avg FLOAT NOT NULL,
                    growth_rate FLOAT NOT NULL,
                    related_symbols JSONB,
                    influence_score FLOAT NOT NULL,
                    first_detected TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
    
    async def _load_influencers(self):
        """Cargar influencers conocidos de la base de datos"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM influencers")
                
                for row in rows:
                    influencer = InfluencerProfile(
                        username=row['username'],
                        platform=row['platform'],
                        followers=row['followers'],
                        influence_level=InfluenceLevel(row['influence_level']),
                        credibility_score=row['credibility_score'],
                        specialization=row['specialization'],
                        avg_engagement=row['avg_engagement'],
                        prediction_accuracy=row['prediction_accuracy'],
                        last_updated=row['last_updated']
                    )
                    
                    key = f"{row['platform']}:{row['username']}"
                    self.influencers_cache[key] = influencer
                    
            logger.info(f"Cargados {len(self.influencers_cache)} influencers")
            
        except Exception as e:
            logger.error(f"Error cargando influencers: {e}")
    
    async def collect_social_data(self, symbols: List[str], hours_back: int = 24) -> List[SocialMediaPost]:
        """Recolectar datos de redes sociales para símbolos específicos"""
        all_posts = []
        
        # Generar keywords de búsqueda
        search_terms = self._generate_search_terms(symbols)
        
        # Recolectar de Twitter
        if self.twitter_api:
            twitter_posts = await self._collect_twitter_data(search_terms, hours_back)
            all_posts.extend(twitter_posts)
        
        # Recolectar de Reddit
        if self.reddit_api:
            reddit_posts = await self._collect_reddit_data(search_terms, hours_back)
            all_posts.extend(reddit_posts)
        
        # Recolectar de otras fuentes
        telegram_posts = await self._collect_telegram_data(search_terms, hours_back)
        all_posts.extend(telegram_posts)
        
        # Filtrar y limpiar datos
        filtered_posts = self._filter_and_clean_posts(all_posts)
        
        # Guardar en base de datos
        await self._save_posts_to_db(filtered_posts)
        
        logger.info(f"Recolectados {len(filtered_posts)} posts de redes sociales")
        return filtered_posts
    
    def _generate_search_terms(self, symbols: List[str]) -> List[str]:
        """Generar términos de búsqueda para los símbolos"""
        search_terms = []
        
        for symbol in symbols:
            # Symbol base (ej: BTC, ETH)
            base = symbol.split('/')[0] if '/' in symbol else symbol
            search_terms.append(base)
            
            # Hashtags comunes
            search_terms.extend([
                f"#{base}",
                f"${base}",
                f"{base}USD",
                f"{base}USDT"
            ])
            
            # Términos relacionados
            if base == 'BTC':
                search_terms.extend(['bitcoin', 'Bitcoin', '#bitcoin', '$BTC'])
            elif base == 'ETH':
                search_terms.extend(['ethereum', 'Ethereum', '#ethereum', '$ETH'])
            elif base == 'ADA':
                search_terms.extend(['cardano', 'Cardano', '#cardano', '$ADA'])
        
        return list(set(search_terms))
    
    async def _collect_twitter_data(self, search_terms: List[str], hours_back: int) -> List[SocialMediaPost]:
        """Recolectar datos de Twitter"""
        posts = []
        
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            for term in search_terms[:10]:  # Limitar para evitar rate limits
                tweets = tweepy.Paginator(
                    self.twitter_api.search_recent_tweets,
                    query=term,
                    tweet_fields=['public_metrics', 'created_at', 'author_id', 'entities'],
                    user_fields=['public_metrics'],
                    expansions=['author_id'],
                    start_time=start_time,
                    max_results=100
                ).flatten(limit=500)
                
                for tweet in tweets:
                    try:
                        # Obtener información del autor
                        author_info = None
                        if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                            for user in tweet.includes['users']:
                                if user.id == tweet.author_id:
                                    author_info = user
                                    break
                        
                        # Extraer hashtags y mentions
                        hashtags = []
                        mentions = []
                        if tweet.entities:
                            if 'hashtags' in tweet.entities:
                                hashtags = [h['tag'] for h in tweet.entities['hashtags']]
                            if 'mentions' in tweet.entities:
                                mentions = [m['username'] for m in tweet.entities['mentions']]
                        
                        post = SocialMediaPost(
                            id=f"twitter_{tweet.id}",
                            platform="twitter",
                            author=author_info.username if author_info else f"user_{tweet.author_id}",
                            content=tweet.text,
                            timestamp=tweet.created_at,
                            likes=tweet.public_metrics.get('like_count', 0),
                            shares=tweet.public_metrics.get('retweet_count', 0),
                            comments=tweet.public_metrics.get('reply_count', 0),
                            author_followers=author_info.public_metrics.get('followers_count', 0) if author_info else 0,
                            hashtags=hashtags,
                            mentions=mentions,
                            url=f"https://twitter.com/user/status/{tweet.id}"
                        )
                        
                        posts.append(post)
                        
                    except Exception as e:
                        logger.warning(f"Error procesando tweet {tweet.id}: {e}")
                        continue
                
                # Pequeña pausa para evitar rate limits
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error recolectando datos de Twitter: {e}")
        
        return posts
    
    async def _collect_reddit_data(self, search_terms: List[str], hours_back: int) -> List[SocialMediaPost]:
        """Recolectar datos de Reddit"""
        posts = []
        
        try:
            crypto_subreddits = [
                'cryptocurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets',
                'altcoin', 'CryptoCurrency', 'btc', 'ethtrader'
            ]
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            for subreddit_name in crypto_subreddits:
                try:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    # Posts recientes
                    for submission in subreddit.new(limit=50):
                        if datetime.fromtimestamp(submission.created_utc) < cutoff_time:
                            continue
                        
                        # Verificar si contiene nuestros términos
                        text_content = f"{submission.title} {submission.selftext}".lower()
                        if not any(term.lower() in text_content for term in search_terms):
                            continue
                        
                        post = SocialMediaPost(
                            id=f"reddit_{submission.id}",
                            platform="reddit",
                            author=str(submission.author) if submission.author else "deleted",
                            content=f"{submission.title}\n\n{submission.selftext}",
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            likes=submission.score,
                            shares=0,  # Reddit no tiene shares directos
                            comments=submission.num_comments,
                            author_followers=0,  # No disponible fácilmente en Reddit
                            hashtags=[],
                            mentions=[],
                            url=f"https://reddit.com{submission.permalink}"
                        )
                        
                        posts.append(post)
                    
                    # Comments en posts populares
                    for submission in subreddit.hot(limit=10):
                        if datetime.fromtimestamp(submission.created_utc) < cutoff_time:
                            continue
                        
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list()[:20]:
                            if datetime.fromtimestamp(comment.created_utc) < cutoff_time:
                                continue
                            
                            if not any(term.lower() in comment.body.lower() for term in search_terms):
                                continue
                            
                            post = SocialMediaPost(
                                id=f"reddit_comment_{comment.id}",
                                platform="reddit",
                                author=str(comment.author) if comment.author else "deleted",
                                content=comment.body,
                                timestamp=datetime.fromtimestamp(comment.created_utc),
                                likes=comment.score,
                                shares=0,
                                comments=0,
                                author_followers=0,
                                hashtags=[],
                                mentions=[],
                                url=f"https://reddit.com{comment.permalink}"
                            )
                            
                            posts.append(post)
                
                except Exception as e:
                    logger.warning(f"Error recolectando de subreddit {subreddit_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error recolectando datos de Reddit: {e}")
        
        return posts
    
    async def _collect_telegram_data(self, search_terms: List[str], hours_back: int) -> List[SocialMediaPost]:
        """Recolectar datos de Telegram (usando web scraping público)"""
        posts = []
        
        # Telegram público channels conocidos
        public_channels = [
            'bitcoin', 'ethereum', 'cryptonews', 'binance',
            'coindesk', 'cointelegraph'
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for channel in public_channels:
                    try:
                        url = f"https://t.me/s/{channel}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                html = await response.text()
                                # Aquí procesaríamos el HTML para extraer posts
                                # Por simplicidad, simulamos algunos posts
                                pass
                    except Exception as e:
                        logger.warning(f"Error accediendo a canal {channel}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error recolectando datos de Telegram: {e}")
        
        return posts
    
    def _filter_and_clean_posts(self, posts: List[SocialMediaPost]) -> List[SocialMediaPost]:
        """Filtrar y limpiar posts recolectados"""
        filtered = []
        
        for post in posts:
            # Filtrar spam y bots
            if self._is_spam_or_bot(post):
                continue
            
            # Filtrar por mínimo de followers
            if post.author_followers < self.min_followers_threshold:
                continue
            
            # Limpiar contenido
            post.content = self._clean_text(post.content)
            
            if len(post.content.strip()) < 10:  # Muy corto
                continue
            
            filtered.append(post)
        
        return filtered
    
    def _is_spam_or_bot(self, post: SocialMediaPost) -> bool:
        """Detectar spam y bots"""
        content = post.content.lower()
        
        # Patrones de spam comunes
        spam_patterns = [
            r'buy.*now.*limited.*time',
            r'guaranteed.*profit',
            r'click.*link.*below',
            r'dm.*me.*for.*signals',
            r'\d+%.*profit.*daily'
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, content):
                return True
        
        # Verificar ratio de hashtags
        hashtag_ratio = len(post.hashtags) / max(len(post.content.split()), 1)
        if hashtag_ratio > 0.3:  # Demasiados hashtags
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto de posts"""
        # Remover URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        
        # Remover menciones excesivas
        text = re.sub(r'@\w+', '', text)
        
        # Limpiar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _save_posts_to_db(self, posts: List[SocialMediaPost]):
        """Guardar posts en base de datos"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                for post in posts:
                    await conn.execute("""
                        INSERT INTO social_posts 
                        (id, platform, author, content, timestamp, likes, shares, comments, 
                         author_followers, hashtags, mentions)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (id) DO NOTHING
                    """, 
                    post.id, post.platform, post.author, post.content,
                    post.timestamp, post.likes, post.shares, post.comments,
                    post.author_followers, json.dumps(post.hashtags), 
                    json.dumps(post.mentions))
        
        except Exception as e:
            logger.error(f"Error guardando posts en DB: {e}")
    
    async def analyze_sentiment(self, posts: List[SocialMediaPost]) -> Dict[str, Any]:
        """Analizar sentimiento de los posts recolectados"""
        if not posts:
            return {'average_sentiment': 0, 'confidence': 0, 'analysis': []}
        
        analysis_results = []
        
        for post in posts:
            try:
                sentiment = await self._analyze_post_sentiment(post)
                analysis_results.append({
                    'post_id': post.id,
                    'platform': post.platform,
                    'author': post.author,
                    'sentiment': sentiment,
                    'timestamp': post.timestamp,
                    'influence_weight': self._calculate_influence_weight(post)
                })
                
                # Actualizar en base de datos
                if self.db_pool:
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE social_posts 
                            SET sentiment_score = $1, impact_score = $2
                            WHERE id = $3
                        """, sentiment.score, sentiment.impact_score, post.id)
                
            except Exception as e:
                logger.warning(f"Error analizando sentimiento de post {post.id}: {e}")
                continue
        
        # Calcular métricas agregadas
        if analysis_results:
            weighted_sentiments = []
            total_weight = 0
            
            for result in analysis_results:
                weight = result['influence_weight']
                sentiment_score = result['sentiment'].score
                weighted_sentiments.append(sentiment_score * weight)
                total_weight += weight
            
            average_sentiment = sum(weighted_sentiments) / total_weight if total_weight > 0 else 0
            confidence = self._calculate_confidence(analysis_results)
            
            return {
                'average_sentiment': average_sentiment,
                'confidence': confidence,
                'total_posts': len(analysis_results),
                'positive_posts': len([r for r in analysis_results if r['sentiment'].score > 0.1]),
                'negative_posts': len([r for r in analysis_results if r['sentiment'].score < -0.1]),
                'neutral_posts': len([r for r in analysis_results if abs(r['sentiment'].score) <= 0.1]),
                'analysis': analysis_results
            }
        
        return {'average_sentiment': 0, 'confidence': 0, 'analysis': []}
    
    async def _analyze_post_sentiment(self, post: SocialMediaPost) -> SentimentAnalysis:
        """Analizar sentimiento de un post individual"""
        try:
            if self.sentiment_model and hasattr(self.sentiment_model, '__call__'):
                # Usar modelo FinBERT
                result = self.sentiment_model(post.content)
                
                # FinBERT devuelve positive, negative, neutral
                sentiment_scores = {r['label']: r['score'] for r in result[0]}
                
                # Convertir a escala -1 a 1
                positive = sentiment_scores.get('positive', 0)
                negative = sentiment_scores.get('negative', 0)
                neutral = sentiment_scores.get('neutral', 0)
                
                score = positive - negative
                confidence = max(positive, negative, neutral)
                
                # Determinar label
                if score > 0.5:
                    label = SentimentScore.VERY_POSITIVE
                elif score > 0.1:
                    label = SentimentScore.POSITIVE
                elif score < -0.5:
                    label = SentimentScore.VERY_NEGATIVE
                elif score < -0.1:
                    label = SentimentScore.NEGATIVE
                else:
                    label = SentimentScore.NEUTRAL
                
                # Analizar emociones si tenemos el modelo
                emotions = {}
                if self.emotion_model:
                    emotion_result = self.emotion_model(post.content)
                    emotions = {r['label']: r['score'] for r in emotion_result[0]}
                
                # Extraer keywords
                keywords = self._extract_keywords(post.content)
                
                # Calcular impact score
                influence_weight = self._calculate_influence_weight(post)
                impact_score = abs(score) * confidence * influence_weight
                
                return SentimentAnalysis(
                    score=score,
                    confidence=confidence,
                    label=label,
                    emotions=emotions,
                    keywords=keywords,
                    impact_score=impact_score
                )
            
            else:
                # Fallback a TextBlob
                blob = TextBlob(post.content)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.5:
                    label = SentimentScore.VERY_POSITIVE
                elif polarity > 0.1:
                    label = SentimentScore.POSITIVE
                elif polarity < -0.5:
                    label = SentimentScore.VERY_NEGATIVE
                elif polarity < -0.1:
                    label = SentimentScore.NEGATIVE
                else:
                    label = SentimentScore.NEUTRAL
                
                return SentimentAnalysis(
                    score=polarity,
                    confidence=blob.sentiment.subjectivity,
                    label=label,
                    emotions={},
                    keywords=self._extract_keywords(post.content),
                    impact_score=abs(polarity) * self._calculate_influence_weight(post)
                )
        
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento: {e}")
            return SentimentAnalysis(
                score=0,
                confidence=0,
                label=SentimentScore.NEUTRAL,
                emotions={},
                keywords=[],
                impact_score=0
            )
    
    def _calculate_influence_weight(self, post: SocialMediaPost) -> float:
        """Calcular peso de influencia de un post"""
        base_weight = 1.0
        
        # Factor de followers
        if post.author_followers > 1000000:  # > 1M followers
            follower_factor = 5.0
        elif post.author_followers > 100000:  # > 100K followers
            follower_factor = 3.0
        elif post.author_followers > 10000:   # > 10K followers
            follower_factor = 2.0
        else:
            follower_factor = 1.0
        
        # Factor de engagement
        total_engagement = post.likes + post.shares + post.comments
        engagement_rate = total_engagement / max(post.author_followers, 1)
        engagement_factor = min(engagement_rate * 100, 3.0)  # Cap at 3x
        
        # Factor de plataforma
        platform_factor = {
            'twitter': 1.0,
            'reddit': 0.8,
            'telegram': 0.6
        }.get(post.platform, 0.5)
        
        # Verificar si es un influencer conocido
        influencer_key = f"{post.platform}:{post.author}"
        if influencer_key in self.influencers_cache:
            influencer = self.influencers_cache[influencer_key]
            influencer_factor = {
                InfluenceLevel.WHALE: 10.0,
                InfluenceLevel.CELEBRITY: 8.0,
                InfluenceLevel.HIGH: 5.0,
                InfluenceLevel.MEDIUM: 3.0,
                InfluenceLevel.LOW: 1.0
            }.get(influencer.influence_level, 1.0)
        else:
            influencer_factor = 1.0
        
        return base_weight * follower_factor * engagement_factor * platform_factor * influencer_factor
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraer keywords relevantes del texto"""
        # Lista de keywords financieras importantes
        financial_keywords = [
            'bull', 'bear', 'bullish', 'bearish', 'pump', 'dump',
            'moon', 'crash', 'dip', 'rally', 'breakout', 'support',
            'resistance', 'hodl', 'buy', 'sell', 'profit', 'loss',
            'fomo', 'fud', 'ath', 'bottom', 'top', 'reversal'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_confidence(self, analysis_results: List[Dict]) -> float:
        """Calcular confianza general del análisis"""
        if not analysis_results:
            return 0.0
        
        confidences = [result['sentiment'].confidence for result in analysis_results]
        weights = [result['influence_weight'] for result in analysis_results]
        
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    async def detect_trending_topics(self, posts: List[SocialMediaPost]) -> List[TrendingTopic]:
        """Detectar temas trending"""
        if not posts:
            return []
        
        # Contar menciones de keywords
        keyword_counts = {}
        keyword_sentiments = {}
        keyword_platforms = {}
        keyword_first_seen = {}
        
        for post in posts:
            # Extraer keywords del contenido
            words = re.findall(r'\b\w+\b', post.content.lower())
            hashtags = [h.lower() for h in post.hashtags]
            all_keywords = words + hashtags
            
            for keyword in all_keywords:
                if len(keyword) < 3:  # Filtrar palabras muy cortas
                    continue
                
                if keyword not in keyword_counts:
                    keyword_counts[keyword] = 0
                    keyword_sentiments[keyword] = []
                    keyword_platforms[keyword] = set()
                    keyword_first_seen[keyword] = post.timestamp
                
                keyword_counts[keyword] += 1
                keyword_platforms[keyword].add(post.platform)
                
                # Si tenemos análisis de sentimiento del post
                # (esto requeriría que ya hayamos analizado los posts)
                keyword_sentiments[keyword].append(0)  # Placeholder
                
                # Actualizar primera vez visto
                if post.timestamp < keyword_first_seen[keyword]:
                    keyword_first_seen[keyword] = post.timestamp
        
        # Identificar trending topics
        trending_topics = []
        min_mentions = max(len(posts) * 0.05, 5)  # Al menos 5% de posts o 5 menciones
        
        for keyword, count in keyword_counts.items():
            if count >= min_mentions:
                sentiments = keyword_sentiments[keyword]
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
                
                # Calcular growth rate (simplificado)
                hours_since_first = (datetime.utcnow() - keyword_first_seen[keyword]).total_seconds() / 3600
                growth_rate = count / max(hours_since_first, 1)
                
                # Determinar símbolos relacionados
                related_symbols = self._find_related_symbols(keyword)
                
                # Calcular influence score
                influence_score = count * (1 + abs(avg_sentiment)) * len(keyword_platforms[keyword])
                
                trending_topic = TrendingTopic(
                    keyword=keyword,
                    platforms=list(keyword_platforms[keyword]),
                    mentions_count=count,
                    sentiment_avg=avg_sentiment,
                    sentiment_std=sentiment_std,
                    growth_rate=growth_rate,
                    related_symbols=related_symbols,
                    influence_score=influence_score,
                    first_detected=keyword_first_seen[keyword]
                )
                
                trending_topics.append(trending_topic)
        
        # Ordenar por influence score
        trending_topics.sort(key=lambda x: x.influence_score, reverse=True)
        
        return trending_topics[:20]  # Top 20 trending topics
    
    def _find_related_symbols(self, keyword: str) -> List[str]:
        """Encontrar símbolos relacionados con un keyword"""
        symbol_mappings = {
            'bitcoin': ['BTC/USDT', 'BTC/USD'],
            'btc': ['BTC/USDT', 'BTC/USD'],
            'ethereum': ['ETH/USDT', 'ETH/USD'],
            'eth': ['ETH/USDT', 'ETH/USD'],
            'cardano': ['ADA/USDT', 'ADA/USD'],
            'ada': ['ADA/USDT', 'ADA/USD'],
            'binance': ['BNB/USDT', 'BNB/USD'],
            'bnb': ['BNB/USDT', 'BNB/USD'],
            'solana': ['SOL/USDT', 'SOL/USD'],
            'sol': ['SOL/USDT', 'SOL/USD'],
            'dogecoin': ['DOGE/USDT', 'DOGE/USD'],
            'doge': ['DOGE/USDT', 'DOGE/USD']
        }
        
        keyword_lower = keyword.lower()
        return symbol_mappings.get(keyword_lower, [])
    
    async def generate_trading_signals(self, sentiment_analysis: Dict, trending_topics: List[TrendingTopic]) -> Dict[str, Any]:
        """Generar señales de trading basadas en análisis de redes sociales"""
        signals = {}
        
        # Señal basada en sentimiento general
        avg_sentiment = sentiment_analysis.get('average_sentiment', 0)
        confidence = sentiment_analysis.get('confidence', 0)
        
        if avg_sentiment > 0.3 and confidence > 0.7:
            signals['general_sentiment'] = {
                'action': 'BUY',
                'strength': min(avg_sentiment * confidence, 1.0),
                'reason': f'Sentimiento muy positivo ({avg_sentiment:.2f}) con alta confianza ({confidence:.2f})'
            }
        elif avg_sentiment < -0.3 and confidence > 0.7:
            signals['general_sentiment'] = {
                'action': 'SELL',
                'strength': min(abs(avg_sentiment) * confidence, 1.0),
                'reason': f'Sentimiento muy negativo ({avg_sentiment:.2f}) con alta confianza ({confidence:.2f})'
            }
        else:
            signals['general_sentiment'] = {
                'action': 'HOLD',
                'strength': 0.5,
                'reason': f'Sentimiento neutral o baja confianza'
            }
        
        # Señales basadas en trending topics
        for topic in trending_topics[:5]:  # Top 5 trending topics
            if topic.related_symbols:
                for symbol in topic.related_symbols:
                    if symbol not in signals:
                        signals[symbol] = []
                    
                    # Determinar acción basada en sentimiento y growth rate
                    if topic.sentiment_avg > 0.2 and topic.growth_rate > 5:
                        action = 'BUY'
                        strength = min((topic.sentiment_avg + topic.growth_rate / 10) / 2, 1.0)
                        reason = f'Trending topic "{topic.keyword}" con sentimiento positivo y alto crecimiento'
                    elif topic.sentiment_avg < -0.2 and topic.growth_rate > 5:
                        action = 'SELL'
                        strength = min((abs(topic.sentiment_avg) + topic.growth_rate / 10) / 2, 1.0)
                        reason = f'Trending topic "{topic.keyword}" con sentimiento negativo y alto crecimiento'
                    else:
                        action = 'MONITOR'
                        strength = 0.3
                        reason = f'Trending topic "{topic.keyword}" requiere monitoreo'
                    
                    signals[symbol].append({
                        'action': action,
                        'strength': strength,
                        'reason': reason,
                        'topic': topic.keyword,
                        'mentions': topic.mentions_count,
                        'platforms': topic.platforms
                    })
        
        return signals
    
    async def get_influencer_activity(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Analizar actividad de influencers conocidos"""
        if not self.influencers_cache:
            return {}
        
        activity_summary = {}
        
        for influencer_key, influencer in self.influencers_cache.items():
            if influencer.specialization and any(spec in ['crypto', 'bitcoin', 'trading'] 
                                               for spec in influencer.specialization):
                
                # Buscar posts recientes del influencer
                platform, username = influencer_key.split(':', 1)
                
                # Aquí buscaríamos posts específicos del influencer
                # Por simplicidad, simulamos algunos datos
                activity_summary[username] = {
                    'platform': platform,
                    'influence_level': influencer.influence_level.name,
                    'credibility_score': influencer.credibility_score,
                    'recent_posts': 0,  # Se llenaría con datos reales
                    'avg_sentiment': 0,  # Se calcularía con posts reales
                    'topics_mentioned': []  # Se extraería de posts reales
                }
        
        return activity_summary
    
    async def generate_report(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Generar reporte completo de inteligencia de redes sociales"""
        logger.info(f"Generando reporte de inteligencia para {symbols}")
        
        # Recolectar datos
        posts = await self.collect_social_data(symbols, hours_back)
        
        # Analizar sentimiento
        sentiment_analysis = await self.analyze_sentiment(posts)
        
        # Detectar trending topics
        trending_topics = await self.detect_trending_topics(posts)
        
        # Generar señales de trading
        trading_signals = await self.generate_trading_signals(sentiment_analysis, trending_topics)
        
        # Analizar actividad de influencers
        influencer_activity = await self.get_influencer_activity(symbols, hours_back)
        
        # Compilar reporte
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbols_analyzed': symbols,
            'time_range_hours': hours_back,
            'data_collection': {
                'total_posts': len(posts),
                'platforms': list(set(post.platform for post in posts)),
                'unique_authors': len(set(post.author for post in posts))
            },
            'sentiment_analysis': sentiment_analysis,
            'trending_topics': [
                {
                    'keyword': topic.keyword,
                    'mentions': topic.mentions_count,
                    'sentiment': topic.sentiment_avg,
                    'growth_rate': topic.growth_rate,
                    'platforms': topic.platforms,
                    'related_symbols': topic.related_symbols,
                    'influence_score': topic.influence_score
                }
                for topic in trending_topics
            ],
            'trading_signals': trading_signals,
            'influencer_activity': influencer_activity,
            'summary': {
                'overall_sentiment': sentiment_analysis.get('average_sentiment', 0),
                'confidence_level': sentiment_analysis.get('confidence', 0),
                'key_trends': [topic.keyword for topic in trending_topics[:5]],
                'recommended_action': self._get_overall_recommendation(sentiment_analysis, trading_signals)
            }
        }
        
        logger.info("Reporte de inteligencia generado exitosamente")
        return report
    
    def _get_overall_recommendation(self, sentiment_analysis: Dict, trading_signals: Dict) -> str:
        """Obtener recomendación general basada en el análisis"""
        avg_sentiment = sentiment_analysis.get('average_sentiment', 0)
        confidence = sentiment_analysis.get('confidence', 0)
        
        if avg_sentiment > 0.3 and confidence > 0.7:
            return "BULLISH - Sentimiento muy positivo detectado"
        elif avg_sentiment < -0.3 and confidence > 0.7:
            return "BEARISH - Sentimiento muy negativo detectado"
        elif confidence < 0.4:
            return "UNCERTAIN - Datos insuficientes o contradictorios"
        else:
            return "NEUTRAL - Mantener posiciones actuales"
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Social Media Intelligence Engine limpiado")


# Función de utilidad para crear instancia configurada
def create_social_media_engine(config: Dict[str, Any]) -> SocialMediaIntelligenceEngine:
    """Crear instancia configurada del motor de inteligencia"""
    return SocialMediaIntelligenceEngine(config)


if __name__ == "__main__":
    # Ejemplo de uso
    import asyncio
    
    async def main():
        config = {
            'database_url': 'postgresql://postgres:password@localhost:5432/quantum_trading',
            'twitter_bearer_token': 'your_token_here',
            'reddit_client_id': 'your_client_id_here',
            'reddit_client_secret': 'your_secret_here',
            'min_followers': 1000,
            'sentiment_window_hours': 24,
            'trend_threshold': 0.1
        }
        
        engine = create_social_media_engine(config)
        await engine.initialize()
        
        # Generar reporte para Bitcoin y Ethereum
        report = await engine.generate_report(['BTC/USDT', 'ETH/USDT'], hours_back=24)
        
        print(json.dumps(report, indent=2, default=str))
        
        await engine.cleanup()
    
    asyncio.run(main())