"""
🔧 Dependency Injection Container
Sistema avanzado de inyección de dependencias para desacoplamiento
"""

import inspect
import threading
from typing import Any, Dict, Type, TypeVar, Callable, Optional, Union, get_type_hints
from functools import wraps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Scope(Enum):
    """Alcances de vida de las dependencias"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    REQUEST = "request"

@dataclass
class ServiceDescriptor:
    """Descriptor de servicio para el contenedor DI"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: Scope = Scope.TRANSIENT
    dependencies: Optional[Dict[str, Type]] = None

class DIContainer:
    """Contenedor de inyección de dependencias enterprise"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._scope_stack: List[str] = []
    
    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """Registrar servicio como singleton"""
        return self.register(service_type, implementation_type, Scope.SINGLETON)
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """Registrar servicio como transient"""
        return self.register(service_type, implementation_type, Scope.TRANSIENT)
    
    def register_scoped(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """Registrar servicio como scoped"""
        return self.register(service_type, implementation_type, Scope.SCOPED)
    
    def register(self, service_type: Type[T], implementation_type: Type[T] = None, 
                scope: Scope = Scope.TRANSIENT) -> 'DIContainer':
        """Registrar servicio en el contenedor"""
        with self._lock:
            impl_type = implementation_type or service_type
            
            # Analizar dependencias
            dependencies = self._analyze_dependencies(impl_type)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=impl_type,
                scope=scope,
                dependencies=dependencies
            )
            
            self._services[service_type] = descriptor
            
            logger.debug(f"Registrado servicio {service_type.__name__} con scope {scope.value}")
            
        return self
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T], 
                        scope: Scope = Scope.TRANSIENT) -> 'DIContainer':
        """Registrar factory para crear instancias"""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                scope=scope
            )
            
            self._services[service_type] = descriptor
            
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'DIContainer':
        """Registrar instancia específica"""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                instance=instance,
                scope=Scope.SINGLETON
            )
            
            self._services[service_type] = descriptor
            self._instances[service_type] = instance
            
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolver servicio y sus dependencias"""
        with self._lock:
            return self._resolve_internal(service_type)
    
    def _resolve_internal(self, service_type: Type[T]) -> T:
        """Resolución interna de servicios"""
        if service_type not in self._services:
            raise ValueError(f"Servicio {service_type.__name__} no registrado")
        
        descriptor = self._services[service_type]
        
        # Singleton - una sola instancia global
        if descriptor.scope == Scope.SINGLETON:
            if service_type in self._instances:
                return self._instances[service_type]
            
            instance = self._create_instance(descriptor)
            self._instances[service_type] = instance
            return instance
        
        # Scoped - una instancia por scope
        elif descriptor.scope == Scope.SCOPED:
            current_scope = self._get_current_scope()
            if current_scope not in self._scoped_instances:
                self._scoped_instances[current_scope] = {}
            
            scoped_instances = self._scoped_instances[current_scope]
            if service_type in scoped_instances:
                return scoped_instances[service_type]
            
            instance = self._create_instance(descriptor)
            scoped_instances[service_type] = instance
            return instance
        
        # Transient - nueva instancia cada vez
        else:
            return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Crear instancia usando descriptor"""
        # Instancia pre-creada
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Factory method
        if descriptor.factory is not None:
            return descriptor.factory()
        
        # Constructor injection
        if descriptor.implementation_type is not None:
            return self._create_with_injection(descriptor.implementation_type, descriptor.dependencies)
        
        raise ValueError(f"No se puede crear instancia para {descriptor.service_type.__name__}")
    
    def _create_with_injection(self, implementation_type: Type, dependencies: Dict[str, Type]) -> Any:
        """Crear instancia con inyección de dependencias"""
        if not dependencies:
            return implementation_type()
        
        # Resolver todas las dependencias
        resolved_dependencies = {}
        for param_name, param_type in dependencies.items():
            resolved_dependencies[param_name] = self._resolve_internal(param_type)
        
        return implementation_type(**resolved_dependencies)
    
    def _analyze_dependencies(self, implementation_type: Type) -> Dict[str, Type]:
        """Analizar dependencias del constructor"""
        try:
            signature = inspect.signature(implementation_type.__init__)
            type_hints = get_type_hints(implementation_type.__init__)
            
            dependencies = {}
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param_name in type_hints:
                    param_type = type_hints[param_name]
                    dependencies[param_name] = param_type
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Error analizando dependencias de {implementation_type.__name__}: {e}")
            return {}
    
    def begin_scope(self, scope_id: str = None) -> 'ScopeContext':
        """Iniciar nuevo scope"""
        if scope_id is None:
            scope_id = f"scope_{threading.current_thread().ident}_{len(self._scope_stack)}"
        
        return ScopeContext(self, scope_id)
    
    def _get_current_scope(self) -> str:
        """Obtener scope actual"""
        if not self._scope_stack:
            return "default"
        return self._scope_stack[-1]
    
    def _enter_scope(self, scope_id: str):
        """Entrar en scope"""
        self._scope_stack.append(scope_id)
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
    
    def _exit_scope(self, scope_id: str):
        """Salir de scope y limpiar instancias"""
        if self._scope_stack and self._scope_stack[-1] == scope_id:
            self._scope_stack.pop()
        
        # Limpiar instancias del scope
        if scope_id in self._scoped_instances:
            # Llamar dispose en instancias que lo soporten
            for instance in self._scoped_instances[scope_id].values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing instance: {e}")
            
            del self._scoped_instances[scope_id]
    
    def clear(self):
        """Limpiar todas las instancias"""
        with self._lock:
            # Dispose de singletons
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing singleton: {e}")
            
            self._instances.clear()
            self._scoped_instances.clear()
            self._scope_stack.clear()
    
    def is_registered(self, service_type: Type) -> bool:
        """Verificar si un servicio está registrado"""
        return service_type in self._services
    
    def get_registrations(self) -> Dict[Type, ServiceDescriptor]:
        """Obtener todas las registraciones"""
        return self._services.copy()

class ScopeContext:
    """Contexto de scope para uso con 'with' statement"""
    
    def __init__(self, container: DIContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id
    
    def __enter__(self):
        self.container._enter_scope(self.scope_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._exit_scope(self.scope_id)

# Contenedor global
_global_container = DIContainer()

def injectable(cls: Type[T]) -> Type[T]:
    """Decorator para marcar clases como injectable"""
    cls._injectable = True
    return cls

def inject(service_type: Type[T]) -> T:
    """Función helper para inyección manual"""
    return _global_container.resolve(service_type)

def configure_container() -> DIContainer:
    """Configurar el contenedor global"""
    return _global_container

class DIContainerBuilder:
    """Builder para configurar el contenedor DI"""
    
    def __init__(self):
        self.container = DIContainer()
    
    def add_singleton(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainerBuilder':
        """Agregar singleton"""
        self.container.register_singleton(service_type, implementation_type)
        return self
    
    def add_transient(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainerBuilder':
        """Agregar transient"""
        self.container.register_transient(service_type, implementation_type)
        return self
    
    def add_scoped(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainerBuilder':
        """Agregar scoped"""
        self.container.register_scoped(service_type, implementation_type)
        return self
    
    def add_factory(self, service_type: Type[T], factory: Callable[[], T], 
                   scope: Scope = Scope.TRANSIENT) -> 'DIContainerBuilder':
        """Agregar factory"""
        self.container.register_factory(service_type, factory, scope)
        return self
    
    def add_instance(self, service_type: Type[T], instance: T) -> 'DIContainerBuilder':
        """Agregar instancia"""
        self.container.register_instance(service_type, instance)
        return self
    
    def build(self) -> DIContainer:
        """Construir contenedor configurado"""
        return self.container

# Decorador para inyección automática en métodos
def auto_inject(func):
    """Decorator para inyección automática de parámetros"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Analizar signature del método
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Resolver parámetros no proporcionados
        bound_args = signature.bind_partial(*args, **kwargs)
        
        for param_name, param in signature.parameters.items():
            if param_name not in bound_args.arguments:
                if param_name in type_hints:
                    param_type = type_hints[param_name]
                    if _global_container.is_registered(param_type):
                        bound_args.arguments[param_name] = _global_container.resolve(param_type)
        
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper

# Service Locator pattern (anti-pattern, usar solo cuando DI no sea posible)
class ServiceLocator:
    """Service Locator para casos donde DI no es viable"""
    
    _container: Optional[DIContainer] = None
    
    @classmethod
    def set_container(cls, container: DIContainer):
        """Configurar contenedor"""
        cls._container = container
    
    @classmethod
    def get_service(cls, service_type: Type[T]) -> T:
        """Obtener servicio"""
        if cls._container is None:
            raise RuntimeError("Container no configurado")
        return cls._container.resolve(service_type)

# Configuración por defecto
ServiceLocator.set_container(_global_container)