"""
📡 Event Bus System
Sistema de eventos para comunicación desacoplada entre componentes
"""

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Type, TypeVar, Any, Callable, Optional, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import weakref
from .dependency_injection import injectable

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EventPriority(Enum):
    """Prioridades de eventos"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class DomainEvent:
    """Evento de dominio base"""
    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    version: str = "1.0"
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEvent':
        """Crear desde diccionario"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'priority' in data and isinstance(data['priority'], int):
            data['priority'] = EventPriority(data['priority'])
        return cls(**data)

class EventHandler(ABC):
    """Handler base para eventos"""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> bool:
        """Manejar evento. Retorna True si fue procesado correctamente"""
        pass
    
    @property
    @abstractmethod
    def supported_events(self) -> Set[str]:
        """Tipos de eventos soportados por este handler"""
        pass
    
    @property
    def handler_name(self) -> str:
        """Nombre del handler"""
        return self.__class__.__name__

class EventHandlerResult:
    """Resultado del procesamiento de un evento"""
    
    def __init__(self, success: bool, handler_name: str, 
                 error: Optional[Exception] = None, 
                 processing_time: Optional[float] = None):
        self.success = success
        self.handler_name = handler_name
        self.error = error
        self.processing_time = processing_time
        self.timestamp = datetime.utcnow()

@dataclass
class EventSubscription:
    """Suscripción a eventos"""
    handler: EventHandler
    event_types: Set[str]
    priority: EventPriority = EventPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    active: bool = True

@injectable
class EventBus:
    """Bus de eventos principal para comunicación desacoplada"""
    
    def __init__(self, max_workers: int = 10, enable_metrics: bool = True):
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._subscribers: List[EventSubscription] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: Set[asyncio.Task] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._lock = asyncio.Lock()
        self._weak_refs: List[weakref.ref] = []
        
        # Métricas
        self._enable_metrics = enable_metrics
        self._events_published = 0
        self._events_processed = 0
        self._events_failed = 0
        self._processing_times: List[float] = []
        self._handlers_performance: Dict[str, List[float]] = {}
        
        # Configuración
        self._max_queue_size = 10000
        self._batch_size = 100
        self._processing_timeout = 30.0
    
    async def start(self):
        """Iniciar el procesamiento de eventos"""
        if self._running:
            return
        
        self._running = True
        
        # Crear tareas de procesamiento
        for i in range(3):  # 3 workers por defecto
            task = asyncio.create_task(self._process_events())
            self._processing_tasks.add(task)
        
        logger.info("EventBus iniciado")
    
    async def stop(self):
        """Detener el procesamiento de eventos"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancelar tareas de procesamiento
        for task in self._processing_tasks:
            task.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        self._processing_tasks.clear()
        
        # Cerrar executor
        self._executor.shutdown(wait=True)
        
        logger.info("EventBus detenido")
    
    async def subscribe(self, handler: EventHandler, 
                       event_types: Union[str, List[str]] = None,
                       priority: EventPriority = EventPriority.NORMAL,
                       max_retries: int = 3,
                       retry_delay: float = 1.0,
                       timeout: Optional[float] = None) -> str:
        """Suscribir handler a eventos"""
        
        if isinstance(event_types, str):
            event_types = [event_types]
        elif event_types is None:
            event_types = list(handler.supported_events)
        
        subscription = EventSubscription(
            handler=handler,
            event_types=set(event_types),
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout
        )
        
        async with self._lock:
            self._subscribers.append(subscription)
            
            # Indexar por tipo de evento
            for event_type in event_types:
                if event_type not in self._subscriptions:
                    self._subscriptions[event_type] = []
                self._subscriptions[event_type].append(subscription)
        
        subscription_id = f"{handler.handler_name}_{uuid.uuid4().hex[:8]}"
        logger.debug(f"Handler suscrito: {handler.handler_name} para eventos {event_types}")
        
        return subscription_id
    
    async def unsubscribe(self, handler: EventHandler):
        """Desuscribir handler"""
        async with self._lock:
            # Remover de subscriptores principales
            self._subscribers = [s for s in self._subscribers if s.handler != handler]
            
            # Remover de índice por tipo
            for event_type, subs in self._subscriptions.items():
                self._subscriptions[event_type] = [s for s in subs if s.handler != handler]
        
        logger.debug(f"Handler desuscrito: {handler.handler_name}")
    
    async def publish(self, event: DomainEvent, wait_for_completion: bool = False) -> bool:
        """Publicar evento"""
        if not self._running:
            logger.warning("EventBus no está ejecutándose")
            return False
        
        if self._event_queue.qsize() >= self._max_queue_size:
            logger.error("Cola de eventos llena")
            return False
        
        # Validar evento
        if not self._validate_event(event):
            logger.error(f"Evento inválido: {event.event_type}")
            return False
        
        # Agregar a la cola
        await self._event_queue.put(event)
        
        if self._enable_metrics:
            self._events_published += 1
        
        logger.debug(f"Evento publicado: {event.event_type} ({event.event_id})")
        
        if wait_for_completion:
            # Esperar a que se procese (implementación simplificada)
            await asyncio.sleep(0.1)
        
        return True
    
    async def publish_batch(self, events: List[DomainEvent]) -> Dict[str, bool]:
        """Publicar múltiples eventos"""
        results = {}
        
        for event in events:
            success = await self.publish(event)
            results[event.event_id] = success
        
        return results
    
    async def _process_events(self):
        """Worker para procesar eventos de la cola"""
        while self._running:
            try:
                # Obtener evento con timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en procesamiento de eventos: {e}")
    
    async def _handle_event(self, event: DomainEvent):
        """Manejar un evento específico"""
        start_time = asyncio.get_event_loop().time()
        
        # Obtener handlers para este tipo de evento
        handlers = self._get_handlers_for_event(event.event_type)
        
        if not handlers:
            logger.debug(f"No hay handlers para evento: {event.event_type}")
            return
        
        # Ordenar por prioridad
        handlers.sort(key=lambda h: h.priority.value, reverse=True)
        
        # Procesar con cada handler
        results = []
        for subscription in handlers:
            if not subscription.active:
                continue
            
            result = await self._execute_handler(subscription, event)
            results.append(result)
        
        # Métricas
        if self._enable_metrics:
            processing_time = asyncio.get_event_loop().time() - start_time
            self._processing_times.append(processing_time)
            
            if any(r.success for r in results):
                self._events_processed += 1
            else:
                self._events_failed += 1
        
        logger.debug(f"Evento procesado: {event.event_type} por {len(results)} handlers")
    
    async def _execute_handler(self, subscription: EventSubscription, 
                              event: DomainEvent) -> EventHandlerResult:
        """Ejecutar handler con reintentos y timeout"""
        handler = subscription.handler
        max_retries = subscription.max_retries
        retry_delay = subscription.retry_delay
        timeout = subscription.timeout or self._processing_timeout
        
        for attempt in range(max_retries + 1):
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Ejecutar con timeout
                success = await asyncio.wait_for(
                    handler.handle(event),
                    timeout=timeout
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Actualizar métricas de handler
                if self._enable_metrics:
                    handler_name = handler.handler_name
                    if handler_name not in self._handlers_performance:
                        self._handlers_performance[handler_name] = []
                    self._handlers_performance[handler_name].append(processing_time)
                
                if success:
                    return EventHandlerResult(
                        success=True,
                        handler_name=handler.handler_name,
                        processing_time=processing_time
                    )
                else:
                    logger.warning(f"Handler {handler.handler_name} retornó False para evento {event.event_type}")
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout en handler {handler.handler_name} para evento {event.event_type}")
            except Exception as e:
                logger.error(f"Error en handler {handler.handler_name}: {e}")
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    return EventHandlerResult(
                        success=False,
                        handler_name=handler.handler_name,
                        error=e
                    )
        
        return EventHandlerResult(
            success=False,
            handler_name=handler.handler_name
        )
    
    def _get_handlers_for_event(self, event_type: str) -> List[EventSubscription]:
        """Obtener handlers para un tipo de evento"""
        return self._subscriptions.get(event_type, [])
    
    def _validate_event(self, event: DomainEvent) -> bool:
        """Validar evento"""
        if not event.event_id:
            return False
        if not event.event_type:
            return False
        if not event.timestamp:
            return False
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del EventBus"""
        if not self._enable_metrics:
            return {"metrics_disabled": True}
        
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times else 0
        )
        
        handler_metrics = {}
        for handler_name, times in self._handlers_performance.items():
            handler_metrics[handler_name] = {
                'avg_time': sum(times) / len(times) if times else 0,
                'total_executions': len(times),
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0
            }
        
        return {
            'events_published': self._events_published,
            'events_processed': self._events_processed,
            'events_failed': self._events_failed,
            'success_rate': (
                self._events_processed / max(self._events_published, 1)
            ) * 100,
            'queue_size': self._event_queue.qsize(),
            'active_subscriptions': len(self._subscribers),
            'avg_processing_time': avg_processing_time,
            'handlers_performance': handler_metrics,
            'total_processing_times': len(self._processing_times)
        }
    
    def clear_metrics(self):
        """Limpiar métricas"""
        self._events_published = 0
        self._events_processed = 0
        self._events_failed = 0
        self._processing_times.clear()
        self._handlers_performance.clear()

# Eventos específicos del dominio de trading
@dataclass
class TradingEvent(DomainEvent):
    """Evento base para trading"""
    symbol: str
    exchange: str
    
    def __post_init__(self):
        super().__post_init__()
        self.source = "trading_engine"

@dataclass
class OrderExecutedEvent(TradingEvent):
    """Evento de orden ejecutada"""
    order_id: str
    side: str  # buy/sell
    quantity: float
    price: float
    executed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "order_executed"

@dataclass  
class PriceChangeEvent(TradingEvent):
    """Evento de cambio de precio"""
    old_price: float
    new_price: float
    change_percent: float
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "price_change"

@dataclass
class PortfolioUpdateEvent(DomainEvent):
    """Evento de actualización de portfolio"""
    user_id: str
    total_value: float
    change_percent: float
    updated_positions: Dict[str, Any]
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "portfolio_update"
        self.source = "portfolio_manager"

# Handlers específicos de ejemplo
class OrderExecutionHandler(EventHandler):
    """Handler para eventos de ejecución de órdenes"""
    
    @property
    def supported_events(self) -> Set[str]:
        return {"order_executed"}
    
    async def handle(self, event: DomainEvent) -> bool:
        if isinstance(event, OrderExecutedEvent):
            logger.info(f"Orden ejecutada: {event.order_id} - {event.side} {event.quantity} {event.symbol} @ {event.price}")
            
            # Aquí iría la lógica de procesamiento
            # - Actualizar portfolio
            # - Notificar usuario
            # - Calcular PnL
            # - etc.
            
            return True
        return False

class PriceAlertHandler(EventHandler):
    """Handler para alertas de precio"""
    
    @property
    def supported_events(self) -> Set[str]:
        return {"price_change"}
    
    async def handle(self, event: DomainEvent) -> bool:
        if isinstance(event, PriceChangeEvent):
            if abs(event.change_percent) > 5.0:  # Alerta para cambios > 5%
                logger.warning(f"Alerta de precio: {event.symbol} cambió {event.change_percent:.2f}%")
                # Enviar notificación
            
            return True
        return False

class MetricsHandler(EventHandler):
    """Handler para métricas y analytics"""
    
    @property
    def supported_events(self) -> Set[str]:
        return {"order_executed", "price_change", "portfolio_update"}
    
    async def handle(self, event: DomainEvent) -> bool:
        # Registrar métricas
        logger.debug(f"Registrando métricas para evento: {event.event_type}")
        
        # Aquí iría integración con sistema de métricas
        # - Prometheus
        # - InfluxDB
        # - CloudWatch
        # etc.
        
        return True

# Decoradores para simplificar el uso
def event_handler(*event_types):
    """Decorador para marcar métodos como handlers de eventos"""
    def decorator(func):
        func._event_types = event_types
        func._is_event_handler = True
        return func
    return decorator

def publish_event(event_bus: EventBus):
    """Decorador para publicar eventos automáticamente"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Si el resultado es un evento, publicarlo
            if isinstance(result, DomainEvent):
                await event_bus.publish(result)
            
            return result
        return wrapper
    return decorator

# Factory para crear event bus configurado
class EventBusFactory:
    """Factory para crear instancias configuradas de EventBus"""
    
    @staticmethod
    def create_default() -> EventBus:
        """Crear EventBus con configuración por defecto"""
        return EventBus(max_workers=5, enable_metrics=True)
    
    @staticmethod
    def create_high_performance() -> EventBus:
        """Crear EventBus optimizado para alta performance"""
        bus = EventBus(max_workers=20, enable_metrics=False)
        bus._max_queue_size = 50000
        bus._batch_size = 500
        return bus
    
    @staticmethod
    def create_testing() -> EventBus:
        """Crear EventBus para testing"""
        return EventBus(max_workers=1, enable_metrics=True)