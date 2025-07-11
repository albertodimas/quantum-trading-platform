"""
⚡ Rate Limiter Pattern
Patrón para controlar la velocidad de operaciones y prevenir sobrecarga del sistema
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime, timedelta
import collections
import statistics
from .dependency_injection import injectable

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Estrategias de rate limiting"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Configuración del Rate Limiter"""
    max_requests: int = 100           # Máximo número de requests
    time_window: float = 60.0         # Ventana de tiempo en segundos
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    burst_size: Optional[int] = None  # Tamaño del burst (para token bucket)
    refill_rate: Optional[float] = None  # Velocidad de refill (tokens por segundo)
    backoff_factor: float = 1.5       # Factor de backoff exponencial
    max_backoff: float = 300.0        # Máximo tiempo de backoff
    enable_adaptive: bool = False     # Habilitar rate limiting adaptativo

@dataclass
class RateLimit:
    """Límite de velocidad"""
    key: str
    remaining: int
    limit: int
    reset_time: float
    retry_after: Optional[float] = None

class RateLimitResult:
    """Resultado de verificación de rate limit"""
    
    def __init__(self, allowed: bool, rate_limit: RateLimit, 
                 retry_after: Optional[float] = None):
        self.allowed = allowed
        self.rate_limit = rate_limit
        self.retry_after = retry_after
        self.timestamp = time.time()

class RateLimitExceededException(Exception):
    """Excepción lanzada cuando se excede el rate limit"""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, 
                 rate_limit: Optional[RateLimit] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limit = rate_limit

@injectable
class RateLimiter:
    """Implementation del patrón Rate Limiter"""
    
    def __init__(self, name: str, config: Optional[RateLimitConfig] = None):
        self.name = name
        self.config = config or RateLimitConfig()
        
        # Estado interno por estrategia
        self._tokens: Dict[str, float] = {}
        self._last_refill: Dict[str, float] = {}
        self._requests: Dict[str, collections.deque] = {}
        self._buckets: Dict[str, collections.deque] = {}
        
        # Métricas
        self._total_requests = 0
        self._allowed_requests = 0
        self._denied_requests = 0
        self._avg_wait_times: List[float] = []
        
        # Threading
        self._lock = threading.RLock()
        
        # Adaptive rate limiting
        self._adaptive_limits: Dict[str, float] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        
        # Callbacks
        self._on_rate_limit: Optional[Callable] = None
        self._on_reset: Optional[Callable] = None
    
    async def check_rate_limit(self, key: str, 
                              requests: int = 1,
                              custom_config: Optional[RateLimitConfig] = None) -> RateLimitResult:
        """Verificar si la request está dentro del rate limit"""
        
        config = custom_config or self.config
        current_time = time.time()
        
        with self._lock:
            self._total_requests += 1
            
            if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = await self._check_token_bucket(key, requests, config, current_time)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = await self._check_sliding_window(key, requests, config, current_time)
            elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = await self._check_fixed_window(key, requests, config, current_time)
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                result = await self._check_leaky_bucket(key, requests, config, current_time)
            else:
                raise ValueError(f"Estrategia no soportada: {config.strategy}")
            
            # Actualizar métricas
            if result.allowed:
                self._allowed_requests += 1
            else:
                self._denied_requests += 1
                
                # Callback de rate limit
                if self._on_rate_limit:
                    await self._safe_callback(self._on_rate_limit, key, result)
            
            # Rate limiting adaptativo
            if config.enable_adaptive:
                await self._update_adaptive_limits(key, result, current_time)
            
            return result
    
    async def _check_token_bucket(self, key: str, requests: int, 
                                 config: RateLimitConfig, current_time: float) -> RateLimitResult:
        """Verificar con estrategia Token Bucket"""
        
        max_tokens = config.burst_size or config.max_requests
        refill_rate = config.refill_rate or (config.max_requests / config.time_window)
        
        # Inicializar bucket si no existe
        if key not in self._tokens:
            self._tokens[key] = max_tokens
            self._last_refill[key] = current_time
        
        # Calcular tokens a agregar
        time_passed = current_time - self._last_refill[key]
        tokens_to_add = time_passed * refill_rate
        
        # Refill tokens
        self._tokens[key] = min(max_tokens, self._tokens[key] + tokens_to_add)
        self._last_refill[key] = current_time
        
        # Verificar si hay suficientes tokens
        if self._tokens[key] >= requests:
            self._tokens[key] -= requests
            remaining = int(self._tokens[key])
            reset_time = current_time + (max_tokens - self._tokens[key]) / refill_rate
            
            rate_limit = RateLimit(
                key=key,
                remaining=remaining,
                limit=max_tokens,
                reset_time=reset_time
            )
            
            return RateLimitResult(allowed=True, rate_limit=rate_limit)
        else:
            # No hay suficientes tokens
            retry_after = (requests - self._tokens[key]) / refill_rate
            
            rate_limit = RateLimit(
                key=key,
                remaining=0,
                limit=max_tokens,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
            return RateLimitResult(
                allowed=False, 
                rate_limit=rate_limit, 
                retry_after=retry_after
            )
    
    async def _check_sliding_window(self, key: str, requests: int,
                                   config: RateLimitConfig, current_time: float) -> RateLimitResult:
        """Verificar con estrategia Sliding Window"""
        
        # Inicializar ventana si no existe
        if key not in self._requests:
            self._requests[key] = collections.deque()
        
        request_window = self._requests[key]
        window_start = current_time - config.time_window
        
        # Remover requests fuera de la ventana
        while request_window and request_window[0] <= window_start:
            request_window.popleft()
        
        # Verificar límite
        if len(request_window) + requests <= config.max_requests:
            # Agregar nuevas requests
            for _ in range(requests):
                request_window.append(current_time)
            
            remaining = config.max_requests - len(request_window)
            reset_time = request_window[0] + config.time_window if request_window else current_time
            
            rate_limit = RateLimit(
                key=key,
                remaining=remaining,
                limit=config.max_requests,
                reset_time=reset_time
            )
            
            return RateLimitResult(allowed=True, rate_limit=rate_limit)
        else:
            # Excede el límite
            if request_window:
                retry_after = request_window[0] + config.time_window - current_time
            else:
                retry_after = 0
            
            rate_limit = RateLimit(
                key=key,
                remaining=0,
                limit=config.max_requests,
                reset_time=current_time + config.time_window,
                retry_after=max(0, retry_after)
            )
            
            return RateLimitResult(
                allowed=False,
                rate_limit=rate_limit,
                retry_after=max(0, retry_after)
            )
    
    async def _check_fixed_window(self, key: str, requests: int,
                                 config: RateLimitConfig, current_time: float) -> RateLimitResult:
        """Verificar con estrategia Fixed Window"""
        
        # Calcular ventana actual
        window_id = int(current_time // config.time_window)
        window_key = f"{key}:{window_id}"
        
        # Inicializar contador si no existe
        if window_key not in self._requests:
            self._requests[window_key] = collections.deque()
        
        current_count = len(self._requests[window_key])
        
        # Verificar límite
        if current_count + requests <= config.max_requests:
            # Agregar requests
            for _ in range(requests):
                self._requests[window_key].append(current_time)
            
            remaining = config.max_requests - current_count - requests
            window_end = (window_id + 1) * config.time_window
            
            rate_limit = RateLimit(
                key=key,
                remaining=remaining,
                limit=config.max_requests,
                reset_time=window_end
            )
            
            return RateLimitResult(allowed=True, rate_limit=rate_limit)
        else:
            # Excede el límite
            window_end = (window_id + 1) * config.time_window
            retry_after = window_end - current_time
            
            rate_limit = RateLimit(
                key=key,
                remaining=0,
                limit=config.max_requests,
                reset_time=window_end,
                retry_after=retry_after
            )
            
            return RateLimitResult(
                allowed=False,
                rate_limit=rate_limit,
                retry_after=retry_after
            )
    
    async def _check_leaky_bucket(self, key: str, requests: int,
                                 config: RateLimitConfig, current_time: float) -> RateLimitResult:
        """Verificar con estrategia Leaky Bucket"""
        
        leak_rate = config.max_requests / config.time_window
        bucket_size = config.burst_size or config.max_requests
        
        # Inicializar bucket si no existe
        if key not in self._buckets:
            self._buckets[key] = collections.deque()
            self._last_refill[key] = current_time
        
        bucket = self._buckets[key]
        
        # Simular leak (procesar requests)
        time_passed = current_time - self._last_refill[key]
        requests_to_process = int(time_passed * leak_rate)
        
        for _ in range(min(requests_to_process, len(bucket))):
            bucket.popleft()
        
        self._last_refill[key] = current_time
        
        # Verificar si hay espacio en el bucket
        if len(bucket) + requests <= bucket_size:
            # Agregar requests al bucket
            for _ in range(requests):
                bucket.append(current_time)
            
            remaining = bucket_size - len(bucket)
            
            # Calcular tiempo de reset (cuando el bucket estará vacío)
            if bucket:
                reset_time = current_time + len(bucket) / leak_rate
            else:
                reset_time = current_time
            
            rate_limit = RateLimit(
                key=key,
                remaining=remaining,
                limit=bucket_size,
                reset_time=reset_time
            )
            
            return RateLimitResult(allowed=True, rate_limit=rate_limit)
        else:
            # Bucket lleno
            retry_after = (len(bucket) + requests - bucket_size) / leak_rate
            
            rate_limit = RateLimit(
                key=key,
                remaining=0,
                limit=bucket_size,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
            return RateLimitResult(
                allowed=False,
                rate_limit=rate_limit,
                retry_after=retry_after
            )
    
    async def _update_adaptive_limits(self, key: str, result: RateLimitResult, current_time: float):
        """Actualizar límites adaptativos basados en performance"""
        
        if key not in self._performance_metrics:
            self._performance_metrics[key] = []
        
        # Registrar métrica de performance
        if result.allowed:
            self._performance_metrics[key].append(1.0)  # Éxito
        else:
            self._performance_metrics[key].append(0.0)  # Fallo
        
        # Mantener solo métricas recientes
        if len(self._performance_metrics[key]) > 100:
            self._performance_metrics[key] = self._performance_metrics[key][-100:]
        
        # Calcular tasa de éxito
        if len(self._performance_metrics[key]) >= 10:
            success_rate = statistics.mean(self._performance_metrics[key])
            
            # Ajustar límites adaptativos
            if success_rate > 0.95:  # Alta tasa de éxito, puede aumentar límite
                if key not in self._adaptive_limits:
                    self._adaptive_limits[key] = self.config.max_requests
                self._adaptive_limits[key] *= 1.1  # Aumentar 10%
            elif success_rate < 0.8:  # Baja tasa de éxito, reducir límite
                if key not in self._adaptive_limits:
                    self._adaptive_limits[key] = self.config.max_requests
                self._adaptive_limits[key] *= 0.9  # Reducir 10%
            
            # Limitar entre 10% y 200% del límite original
            min_limit = self.config.max_requests * 0.1
            max_limit = self.config.max_requests * 2.0
            self._adaptive_limits[key] = max(min_limit, min(max_limit, self._adaptive_limits[key]))
    
    async def wait_for_reset(self, key: str, timeout: Optional[float] = None) -> bool:
        """Esperar hasta que el rate limit se resetee"""
        
        current_time = time.time()
        
        with self._lock:
            if key in self._last_refill:
                # Calcular tiempo de espera basado en estrategia
                if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    if key in self._tokens:
                        refill_rate = self.config.refill_rate or (self.config.max_requests / self.config.time_window)
                        wait_time = (1 - self._tokens[key]) / refill_rate
                    else:
                        wait_time = 0
                else:
                    wait_time = self.config.time_window
                
                # Aplicar timeout si se especifica
                if timeout and wait_time > timeout:
                    return False
                
                logger.debug(f"Esperando {wait_time:.2f}s para reset de rate limit: {key}")
                await asyncio.sleep(wait_time)
                return True
        
        return True
    
    async def reset(self, key: Optional[str] = None):
        """Resetear rate limiter para una key específica o todas"""
        
        with self._lock:
            if key:
                # Reset específico
                self._tokens.pop(key, None)
                self._last_refill.pop(key, None)
                self._requests.pop(key, None)
                self._buckets.pop(key, None)
                self._adaptive_limits.pop(key, None)
                self._performance_metrics.pop(key, None)
                
                logger.debug(f"Rate limiter reseteado para key: {key}")
                
                if self._on_reset:
                    await self._safe_callback(self._on_reset, key)
            else:
                # Reset global
                self._tokens.clear()
                self._last_refill.clear()
                self._requests.clear()
                self._buckets.clear()
                self._adaptive_limits.clear()
                self._performance_metrics.clear()
                
                logger.debug(f"Rate limiter '{self.name}' reseteado completamente")
                
                if self._on_reset:
                    await self._safe_callback(self._on_reset, None)
    
    def get_status(self, key: str) -> Optional[RateLimit]:
        """Obtener estado actual del rate limit para una key"""
        
        current_time = time.time()
        
        with self._lock:
            if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                if key in self._tokens:
                    max_tokens = self.config.burst_size or self.config.max_requests
                    remaining = int(self._tokens[key])
                    reset_time = self._last_refill[key] + self.config.time_window
                    
                    return RateLimit(
                        key=key,
                        remaining=remaining,
                        limit=max_tokens,
                        reset_time=reset_time
                    )
            
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                if key in self._requests:
                    request_window = self._requests[key]
                    remaining = self.config.max_requests - len(request_window)
                    reset_time = request_window[0] + self.config.time_window if request_window else current_time
                    
                    return RateLimit(
                        key=key,
                        remaining=remaining,
                        limit=self.config.max_requests,
                        reset_time=reset_time
                    )
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del rate limiter"""
        
        with self._lock:
            success_rate = (
                self._allowed_requests / max(self._total_requests, 1)
            ) * 100
            
            avg_wait_time = (
                statistics.mean(self._avg_wait_times) 
                if self._avg_wait_times else 0
            )
            
            return {
                'name': self.name,
                'strategy': self.config.strategy.value,
                'total_requests': self._total_requests,
                'allowed_requests': self._allowed_requests,
                'denied_requests': self._denied_requests,
                'success_rate': success_rate,
                'avg_wait_time': avg_wait_time,
                'active_keys': len(set(
                    list(self._tokens.keys()) + 
                    list(self._requests.keys()) + 
                    list(self._buckets.keys())
                )),
                'adaptive_limits': dict(self._adaptive_limits),
                'config': {
                    'max_requests': self.config.max_requests,
                    'time_window': self.config.time_window,
                    'strategy': self.config.strategy.value,
                    'burst_size': self.config.burst_size,
                    'refill_rate': self.config.refill_rate,
                    'enable_adaptive': self.config.enable_adaptive
                }
            }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Ejecutar callback de forma segura"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en callback de rate limiter: {e}")
    
    def set_on_rate_limit(self, callback: Callable):
        """Establecer callback para cuando se excede el rate limit"""
        self._on_rate_limit = callback
    
    def set_on_reset(self, callback: Callable):
        """Establecer callback para cuando se resetea el rate limiter"""
        self._on_reset = callback

# Decorador para aplicar rate limiting
def rate_limit(name: str, config: Optional[RateLimitConfig] = None, 
               key_func: Optional[Callable] = None):
    """Decorador para aplicar rate limiting a funciones"""
    
    limiter = RateLimiter(name, config)
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generar key para rate limiting
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}"
            
            # Verificar rate limit
            result = await limiter.check_rate_limit(key)
            
            if not result.allowed:
                raise RateLimitExceededException(
                    f"Rate limit excedido para {key}",
                    retry_after=result.retry_after,
                    rate_limit=result.rate_limit
                )
            
            # Ejecutar función
            return await func(*args, **kwargs)
        
        # Agregar referencia al rate limiter
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator

# Registry para gestionar múltiples rate limiters
@injectable
class RateLimiterRegistry:
    """Registro centralizado de rate limiters"""
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
        """Registrar nuevo rate limiter"""
        with self._lock:
            if name in self._limiters:
                return self._limiters[name]
            
            limiter = RateLimiter(name, config)
            self._limiters[name] = limiter
            
            logger.debug(f"Rate limiter registrado: {name}")
            return limiter
    
    def get(self, name: str) -> Optional[RateLimiter]:
        """Obtener rate limiter por nombre"""
        return self._limiters.get(name)
    
    def get_or_create(self, name: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
        """Obtener o crear rate limiter"""
        limiter = self.get(name)
        if limiter is None:
            limiter = self.register(name, config)
        return limiter
    
    def remove(self, name: str) -> bool:
        """Remover rate limiter"""
        with self._lock:
            if name in self._limiters:
                del self._limiters[name]
                logger.debug(f"Rate limiter removido: {name}")
                return True
            return False
    
    async def check_rate_limit(self, limiter_name: str, key: str, 
                              requests: int = 1) -> RateLimitResult:
        """Verificar rate limit usando limiter específico"""
        limiter = self.get(limiter_name)
        if not limiter:
            raise ValueError(f"Rate limiter no encontrado: {limiter_name}")
        
        return await limiter.check_rate_limit(key, requests)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener métricas de todos los rate limiters"""
        metrics = {}
        for name, limiter in self._limiters.items():
            metrics[name] = limiter.get_metrics()
        return metrics
    
    def get_names(self) -> List[str]:
        """Obtener nombres de todos los rate limiters"""
        return list(self._limiters.keys())
    
    async def reset_all(self):
        """Resetear todos los rate limiters"""
        with self._lock:
            for limiter in self._limiters.values():
                await limiter.reset()
            logger.info("Todos los rate limiters reseteados")

# Context manager para rate limiting temporal
class TemporaryRateLimit:
    """Context manager para rate limiting temporal"""
    
    def __init__(self, name: str, config: Optional[RateLimitConfig] = None):
        self.limiter = RateLimiter(name, config)
        self.key = f"temp_{name}"
    
    async def __aenter__(self):
        return self.limiter
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.limiter.reset(self.key)

# Configuraciones predefinidas para casos comunes
class RateLimitPresets:
    """Configuraciones predefinidas de rate limiters"""
    
    @staticmethod
    def api_calls() -> RateLimitConfig:
        """Para llamadas a APIs externas"""
        return RateLimitConfig(
            max_requests=100,
            time_window=60.0,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_size=20,
            refill_rate=1.67,  # 100 requests / 60 seconds
            enable_adaptive=True
        )
    
    @staticmethod
    def database_queries() -> RateLimitConfig:
        """Para queries de base de datos"""
        return RateLimitConfig(
            max_requests=500,
            time_window=60.0,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            enable_adaptive=False
        )
    
    @staticmethod
    def user_actions() -> RateLimitConfig:
        """Para acciones de usuario"""
        return RateLimitConfig(
            max_requests=50,
            time_window=60.0,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            enable_adaptive=False
        )
    
    @staticmethod
    def exchange_api() -> RateLimitConfig:
        """Para APIs de exchanges (muy restrictivo)"""
        return RateLimitConfig(
            max_requests=10,
            time_window=60.0,
            strategy=RateLimitStrategy.LEAKY_BUCKET,
            burst_size=5,
            backoff_factor=2.0,
            max_backoff=600.0,
            enable_adaptive=True
        )
    
    @staticmethod
    def high_frequency() -> RateLimitConfig:
        """Para operaciones de alta frecuencia"""
        return RateLimitConfig(
            max_requests=1000,
            time_window=1.0,  # 1 segundo
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_size=100,
            refill_rate=1000.0,
            enable_adaptive=True
        )

# Utilidades de integración
async def with_rate_limit(limiter: RateLimiter, key: str, func: Callable, 
                         *args, max_retries: int = 3, **kwargs):
    """Ejecutar función con rate limiting y reintentos automáticos"""
    
    for attempt in range(max_retries + 1):
        try:
            result = await limiter.check_rate_limit(key)
            
            if result.allowed:
                return await func(*args, **kwargs)
            else:
                if attempt < max_retries and result.retry_after:
                    logger.debug(f"Rate limit excedido, esperando {result.retry_after:.2f}s (intento {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(result.retry_after)
                else:
                    raise RateLimitExceededException(
                        f"Rate limit excedido después de {max_retries} intentos",
                        retry_after=result.retry_after,
                        rate_limit=result.rate_limit
                    )
        
        except RateLimitExceededException:
            if attempt == max_retries:
                raise
            await asyncio.sleep(limiter.config.backoff_factor ** attempt)
    
    raise RateLimitExceededException("Máximo número de reintentos alcanzado")