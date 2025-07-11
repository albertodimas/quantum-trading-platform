"""
⚡ Circuit Breaker Pattern
Patrón para prevenir cascadas de fallos y mejorar la resiliencia del sistema
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
import statistics
from .dependency_injection import injectable

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Estados del Circuit Breaker"""
    CLOSED = "closed"        # Funcionamiento normal
    OPEN = "open"           # Circuito abierto, rechaza llamadas
    HALF_OPEN = "half_open" # Prueba si el servicio se recuperó

@dataclass
class CircuitBreakerConfig:
    """Configuración del Circuit Breaker"""
    failure_threshold: int = 5          # Fallos antes de abrir
    recovery_timeout: float = 60.0      # Tiempo antes de intentar recuperación
    success_threshold: int = 3          # Éxitos para cerrar en half-open
    timeout: float = 30.0               # Timeout de operaciones
    expected_exception: type = Exception # Tipo de excepción que cuenta como fallo
    monitor_window: float = 300.0       # Ventana de monitoreo (5 min)
    slow_call_threshold: float = 10.0   # Umbral de llamada lenta
    slow_call_rate_threshold: float = 0.5 # % de llamadas lentas para abrir

@dataclass
class CallResult:
    """Resultado de una llamada"""
    success: bool
    duration: float
    timestamp: float
    exception: Optional[Exception] = None
    is_slow: bool = False

@injectable
class CircuitBreaker:
    """Implementation del patrón Circuit Breaker"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Estado del circuito
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._state_changed_time = time.time()
        
        # Historial de llamadas para métricas
        self._call_history: List[CallResult] = []
        self._lock = threading.RLock()
        
        # Métricas
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_timeouts = 0
        self._total_slow_calls = 0
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None
        self._on_success: Optional[Callable] = None
    
    @property
    def state(self) -> CircuitState:
        """Estado actual del circuito"""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Verificar si el circuito está cerrado"""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Verificar si el circuito está abierto"""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Verificar si el circuito está semi-abierto"""
        return self.state == CircuitState.HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecutar función protegida por circuit breaker"""
        
        # Verificar si podemos ejecutar la llamada
        if not await self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' está abierto"
            )
        
        start_time = time.time()
        result = None
        exception = None
        
        try:
            # Ejecutar con timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                # Ejecutar función síncrona en executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=self.config.timeout
                )
            
            # Registrar éxito
            duration = time.time() - start_time
            await self._on_call_success(duration)
            
            return result
            
        except asyncio.TimeoutError as e:
            exception = e
            duration = time.time() - start_time
            await self._on_call_timeout(duration)
            raise
            
        except Exception as e:
            exception = e
            duration = time.time() - start_time
            
            # Solo contar como fallo si es el tipo esperado
            if isinstance(e, self.config.expected_exception):
                await self._on_call_failure(duration, e)
            
            raise
        
        finally:
            # Registrar en historial
            if exception is not None or result is not None:
                duration = time.time() - start_time
                call_result = CallResult(
                    success=exception is None,
                    duration=duration,
                    timestamp=start_time,
                    exception=exception,
                    is_slow=duration > self.config.slow_call_threshold
                )
                self._record_call(call_result)
    
    async def _can_execute(self) -> bool:
        """Verificar si se puede ejecutar una llamada"""
        with self._lock:
            current_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Verificar si es tiempo de intentar recuperación
                if (current_time - self._last_failure_time) >= self.config.recovery_timeout:
                    self._transition_to_half_open()
                    return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                # En half-open, permitir llamadas limitadas
                return True
            
            return False
    
    async def _on_call_success(self, duration: float):
        """Manejar llamada exitosa"""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            
            if duration > self.config.slow_call_threshold:
                self._total_slow_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                
                # Si alcanzamos el umbral de éxitos, cerrar el circuito
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self._state == CircuitState.CLOSED:
                # Verificar si hay demasiadas llamadas lentas
                if await self._should_open_due_to_slow_calls():
                    self._transition_to_open("Demasiadas llamadas lentas")
        
        # Callback de éxito
        if self._on_success:
            await self._safe_callback(self._on_success, duration)
    
    async def _on_call_failure(self, duration: float, exception: Exception):
        """Manejar llamada fallida"""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if duration > self.config.slow_call_threshold:
                self._total_slow_calls += 1
            
            # Verificar si debemos abrir el circuito
            if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self.config.failure_threshold):
                self._transition_to_open(f"Umbral de fallos alcanzado: {self._failure_count}")
            
            elif self._state == CircuitState.HALF_OPEN:
                # En half-open, cualquier fallo abre el circuito
                self._transition_to_open("Fallo en estado half-open")
        
        # Callback de fallo
        if self._on_failure:
            await self._safe_callback(self._on_failure, duration, exception)
    
    async def _on_call_timeout(self, duration: float):
        """Manejar timeout de llamada"""
        with self._lock:
            self._total_calls += 1
            self._total_timeouts += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # Timeout cuenta como llamada lenta
            self._total_slow_calls += 1
            
            # Verificar si debemos abrir el circuito
            if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self.config.failure_threshold):
                self._transition_to_open(f"Umbral de timeouts alcanzado: {self._failure_count}")
            
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to_open("Timeout en estado half-open")
    
    def _transition_to_closed(self):
        """Transición a estado CLOSED"""
        logger.info(f"Circuit breaker '{self.name}' cerrando")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._state_changed_time = time.time()
        
        asyncio.create_task(self._notify_state_change(CircuitState.CLOSED))
    
    def _transition_to_open(self, reason: str = ""):
        """Transición a estado OPEN"""
        logger.warning(f"Circuit breaker '{self.name}' abriendo: {reason}")
        self._state = CircuitState.OPEN
        self._success_count = 0
        self._state_changed_time = time.time()
        
        asyncio.create_task(self._notify_state_change(CircuitState.OPEN, reason))
    
    def _transition_to_half_open(self):
        """Transición a estado HALF_OPEN"""
        logger.info(f"Circuit breaker '{self.name}' en half-open - probando recuperación")
        self._state = CircuitState.HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._state_changed_time = time.time()
        
        asyncio.create_task(self._notify_state_change(CircuitState.HALF_OPEN))
    
    async def _should_open_due_to_slow_calls(self) -> bool:
        """Verificar si debemos abrir por llamadas lentas"""
        if not self._call_history:
            return False
        
        # Analizar ventana de tiempo reciente
        current_time = time.time()
        window_start = current_time - self.config.monitor_window
        
        recent_calls = [
            call for call in self._call_history 
            if call.timestamp >= window_start
        ]
        
        if len(recent_calls) < 10:  # Mínimo de llamadas para evaluar
            return False
        
        slow_calls = [call for call in recent_calls if call.is_slow]
        slow_call_rate = len(slow_calls) / len(recent_calls)
        
        return slow_call_rate >= self.config.slow_call_rate_threshold
    
    def _record_call(self, call_result: CallResult):
        """Registrar resultado de llamada en historial"""
        with self._lock:
            self._call_history.append(call_result)
            
            # Mantener solo las llamadas recientes
            current_time = time.time()
            cutoff_time = current_time - self.config.monitor_window
            
            self._call_history = [
                call for call in self._call_history 
                if call.timestamp >= cutoff_time
            ]
    
    async def _notify_state_change(self, new_state: CircuitState, reason: str = ""):
        """Notificar cambio de estado"""
        if self._on_state_change:
            await self._safe_callback(self._on_state_change, new_state, reason)
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Ejecutar callback de forma segura"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en callback de circuit breaker: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del circuit breaker"""
        with self._lock:
            current_time = time.time()
            
            # Calcular métricas de la ventana reciente
            window_start = current_time - self.config.monitor_window
            recent_calls = [
                call for call in self._call_history 
                if call.timestamp >= window_start
            ]
            
            success_rate = 0.0
            avg_duration = 0.0
            slow_call_rate = 0.0
            
            if recent_calls:
                successful_calls = [call for call in recent_calls if call.success]
                slow_calls = [call for call in recent_calls if call.is_slow]
                durations = [call.duration for call in recent_calls]
                
                success_rate = len(successful_calls) / len(recent_calls) * 100
                avg_duration = statistics.mean(durations)
                slow_call_rate = len(slow_calls) / len(recent_calls) * 100
            
            return {
                'name': self.name,
                'state': self._state.value,
                'state_duration': current_time - self._state_changed_time,
                'total_calls': self._total_calls,
                'total_successes': self._total_successes,
                'total_failures': self._total_failures,
                'total_timeouts': self._total_timeouts,
                'total_slow_calls': self._total_slow_calls,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'recent_calls_count': len(recent_calls),
                'recent_success_rate': success_rate,
                'recent_avg_duration': avg_duration,
                'recent_slow_call_rate': slow_call_rate,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'slow_call_threshold': self.config.slow_call_threshold,
                    'slow_call_rate_threshold': self.config.slow_call_rate_threshold
                }
            }
    
    def reset(self):
        """Resetear circuit breaker a estado inicial"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0
            self._state_changed_time = time.time()
            self._call_history.clear()
            
            logger.info(f"Circuit breaker '{self.name}' reseteado")
    
    def force_open(self, reason: str = "Forzado manualmente"):
        """Forzar apertura del circuito"""
        with self._lock:
            self._transition_to_open(reason)
    
    def force_close(self):
        """Forzar cierre del circuito"""
        with self._lock:
            self._transition_to_closed()
    
    def set_on_state_change(self, callback: Callable):
        """Establecer callback para cambios de estado"""
        self._on_state_change = callback
    
    def set_on_failure(self, callback: Callable):
        """Establecer callback para fallos"""
        self._on_failure = callback
    
    def set_on_success(self, callback: Callable):
        """Establecer callback para éxitos"""
        self._on_success = callback

class CircuitBreakerOpenException(Exception):
    """Excepción lanzada cuando el circuit breaker está abierto"""
    pass

# Decorador para aplicar circuit breaker
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorador para aplicar circuit breaker a funciones"""
    breaker = CircuitBreaker(name, config)
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        # Agregar referencia al circuit breaker
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator

# Registry para gestionar múltiples circuit breakers
@injectable
class CircuitBreakerRegistry:
    """Registro centralizado de circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Registrar nuevo circuit breaker"""
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            
            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker
            
            logger.debug(f"Circuit breaker registrado: {name}")
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Obtener circuit breaker por nombre"""
        return self._breakers.get(name)
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Obtener o crear circuit breaker"""
        breaker = self.get(name)
        if breaker is None:
            breaker = self.register(name, config)
        return breaker
    
    def remove(self, name: str) -> bool:
        """Remover circuit breaker"""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.debug(f"Circuit breaker removido: {name}")
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener métricas de todos los circuit breakers"""
        metrics = {}
        for name, breaker in self._breakers.items():
            metrics[name] = breaker.get_metrics()
        return metrics
    
    def get_names(self) -> List[str]:
        """Obtener nombres de todos los circuit breakers"""
        return list(self._breakers.keys())
    
    def reset_all(self):
        """Resetear todos los circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Todos los circuit breakers reseteados")

# Context manager para circuit breaker temporal
class TemporaryCircuitBreaker:
    """Context manager para circuit breaker temporal"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.breaker = CircuitBreaker(name, config)
    
    async def __aenter__(self):
        return self.breaker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup si es necesario
        pass

# Configuraciones predefinidas para casos comunes
class CircuitBreakerPresets:
    """Configuraciones predefinidas de circuit breakers"""
    
    @staticmethod
    def fast_service() -> CircuitBreakerConfig:
        """Para servicios rápidos (APIs internas)"""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=5.0,
            slow_call_threshold=2.0,
            slow_call_rate_threshold=0.3
        )
    
    @staticmethod
    def slow_service() -> CircuitBreakerConfig:
        """Para servicios lentos (APIs externas)"""
        return CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,
            success_threshold=3,
            timeout=30.0,
            slow_call_threshold=15.0,
            slow_call_rate_threshold=0.5
        )
    
    @staticmethod
    def database() -> CircuitBreakerConfig:
        """Para conexiones de base de datos"""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=2,
            timeout=10.0,
            slow_call_threshold=5.0,
            slow_call_rate_threshold=0.4
        )
    
    @staticmethod
    def exchange_api() -> CircuitBreakerConfig:
        """Para APIs de exchanges"""
        return CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=300.0,  # 5 minutos
            success_threshold=3,
            timeout=60.0,
            slow_call_threshold=10.0,
            slow_call_rate_threshold=0.6
        )