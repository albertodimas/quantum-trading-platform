"""
🏭 Factory Registry Pattern
Sistema centralizado para la creación y gestión de objetos complejos
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar, Any, Callable, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import inspect
import threading
from .dependency_injection import injectable, DIContainer

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CreationStrategy(Enum):
    """Estrategias de creación de objetos"""
    SINGLETON = "singleton"
    PROTOTYPE = "prototype"
    POOLED = "pooled"
    LAZY = "lazy"

@dataclass
class FactoryDescriptor:
    """Descriptor de factory"""
    factory_type: Type
    strategy: CreationStrategy
    parameters: Dict[str, Any]
    dependencies: List[str]
    created_instances: int = 0
    max_instances: Optional[int] = None
    pool_size: Optional[int] = None
    initialization_func: Optional[Callable] = None
    cleanup_func: Optional[Callable] = None

class Factory(ABC):
    """Factory base abstracta"""
    
    @abstractmethod
    async def create(self, **kwargs) -> Any:
        """Crear objeto"""
        pass
    
    @abstractmethod
    def can_create(self, type_name: str) -> bool:
        """Verificar si puede crear tipo específico"""
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Tipos soportados por esta factory"""
        pass

@injectable
class FactoryRegistry:
    """Registro centralizado de factories"""
    
    def __init__(self, di_container: Optional[DIContainer] = None):
        self._factories: Dict[str, Factory] = {}
        self._descriptors: Dict[str, FactoryDescriptor] = {}
        self._instances: Dict[str, Any] = {}  # Para singletons
        self._pools: Dict[str, List[Any]] = {}  # Para pooled objects
        self._di_container = di_container
        self._lock = threading.RLock()
        
        # Configuración
        self._enable_metrics = True
        self._default_pool_size = 5
        self._lazy_loading = True
    
    def register_factory(self, 
                        type_name: str, 
                        factory: Factory,
                        strategy: CreationStrategy = CreationStrategy.PROTOTYPE,
                        max_instances: Optional[int] = None,
                        pool_size: Optional[int] = None,
                        initialization_func: Optional[Callable] = None,
                        cleanup_func: Optional[Callable] = None,
                        **parameters) -> None:
        """Registrar factory para un tipo"""
        
        with self._lock:
            if not factory.can_create(type_name):
                raise ValueError(f"Factory {factory.__class__.__name__} no puede crear tipo {type_name}")
            
            descriptor = FactoryDescriptor(
                factory_type=type(factory),
                strategy=strategy,
                parameters=parameters,
                dependencies=self._analyze_dependencies(factory),
                max_instances=max_instances,
                pool_size=pool_size or self._default_pool_size,
                initialization_func=initialization_func,
                cleanup_func=cleanup_func
            )
            
            self._factories[type_name] = factory
            self._descriptors[type_name] = descriptor
            
            # Inicializar pool si es necesario
            if strategy == CreationStrategy.POOLED:
                self._pools[type_name] = []
                asyncio.create_task(self._initialize_pool(type_name))
            
            logger.debug(f"Factory registrada: {type_name} -> {factory.__class__.__name__}")
    
    def register_singleton_factory(self, type_name: str, factory: Factory, **parameters):
        """Registrar factory singleton"""
        self.register_factory(type_name, factory, CreationStrategy.SINGLETON, **parameters)
    
    def register_prototype_factory(self, type_name: str, factory: Factory, **parameters):
        """Registrar factory prototype"""
        self.register_factory(type_name, factory, CreationStrategy.PROTOTYPE, **parameters)
    
    def register_pooled_factory(self, type_name: str, factory: Factory, 
                               pool_size: int = 5, **parameters):
        """Registrar factory con pool"""
        self.register_factory(
            type_name, factory, CreationStrategy.POOLED, 
            pool_size=pool_size, **parameters
        )
    
    async def create(self, type_name: str, **kwargs) -> Any:
        """Crear objeto usando factory registrada"""
        
        if type_name not in self._factories:
            raise ValueError(f"No hay factory registrada para tipo: {type_name}")
        
        factory = self._factories[type_name]
        descriptor = self._descriptors[type_name]
        
        # Verificar límite de instancias
        if (descriptor.max_instances and 
            descriptor.created_instances >= descriptor.max_instances):
            raise RuntimeError(f"Límite de instancias alcanzado para {type_name}")
        
        # Aplicar estrategia de creación
        instance = await self._create_with_strategy(type_name, factory, descriptor, **kwargs)
        
        # Inicialización personalizada
        if descriptor.initialization_func:
            await self._run_initialization(instance, descriptor.initialization_func)
        
        # Actualizar métricas
        if self._enable_metrics:
            descriptor.created_instances += 1
        
        logger.debug(f"Objeto creado: {type_name} (estrategia: {descriptor.strategy.value})")
        
        return instance
    
    async def _create_with_strategy(self, type_name: str, factory: Factory, 
                                   descriptor: FactoryDescriptor, **kwargs) -> Any:
        """Crear objeto según estrategia"""
        
        strategy = descriptor.strategy
        
        if strategy == CreationStrategy.SINGLETON:
            return await self._create_singleton(type_name, factory, **kwargs)
        
        elif strategy == CreationStrategy.PROTOTYPE:
            return await self._create_prototype(factory, **kwargs)
        
        elif strategy == CreationStrategy.POOLED:
            return await self._create_from_pool(type_name, factory, **kwargs)
        
        elif strategy == CreationStrategy.LAZY:
            return await self._create_lazy(type_name, factory, **kwargs)
        
        else:
            raise ValueError(f"Estrategia no soportada: {strategy}")
    
    async def _create_singleton(self, type_name: str, factory: Factory, **kwargs) -> Any:
        """Crear singleton"""
        with self._lock:
            if type_name in self._instances:
                return self._instances[type_name]
            
            instance = await factory.create(**kwargs)
            self._instances[type_name] = instance
            return instance
    
    async def _create_prototype(self, factory: Factory, **kwargs) -> Any:
        """Crear nueva instancia"""
        return await factory.create(**kwargs)
    
    async def _create_from_pool(self, type_name: str, factory: Factory, **kwargs) -> Any:
        """Crear desde pool"""
        with self._lock:
            pool = self._pools.get(type_name, [])
            
            if pool:
                instance = pool.pop()
                logger.debug(f"Objeto obtenido del pool: {type_name} (restantes: {len(pool)})")
                return instance
            
            # Pool vacío, crear nueva instancia
            logger.debug(f"Pool vacío para {type_name}, creando nueva instancia")
            return await factory.create(**kwargs)
    
    async def _create_lazy(self, type_name: str, factory: Factory, **kwargs) -> Any:
        """Crear lazy proxy"""
        return LazyProxy(factory, **kwargs)
    
    async def _initialize_pool(self, type_name: str):
        """Inicializar pool con objetos"""
        descriptor = self._descriptors[type_name]
        factory = self._factories[type_name]
        
        pool_size = descriptor.pool_size or self._default_pool_size
        
        for _ in range(pool_size):
            try:
                instance = await factory.create(**descriptor.parameters)
                self._pools[type_name].append(instance)
            except Exception as e:
                logger.error(f"Error inicializando pool para {type_name}: {e}")
                break
        
        logger.debug(f"Pool inicializado para {type_name}: {len(self._pools[type_name])} objetos")
    
    def return_to_pool(self, type_name: str, instance: Any):
        """Retornar objeto al pool"""
        if type_name in self._pools:
            descriptor = self._descriptors[type_name]
            pool = self._pools[type_name]
            
            if len(pool) < (descriptor.pool_size or self._default_pool_size):
                pool.append(instance)
                logger.debug(f"Objeto retornado al pool: {type_name}")
            else:
                # Pool lleno, ejecutar cleanup si está definido
                if descriptor.cleanup_func:
                    asyncio.create_task(self._run_cleanup(instance, descriptor.cleanup_func))
    
    async def _run_initialization(self, instance: Any, init_func: Callable):
        """Ejecutar función de inicialización"""
        try:
            if asyncio.iscoroutinefunction(init_func):
                await init_func(instance)
            else:
                init_func(instance)
        except Exception as e:
            logger.error(f"Error en inicialización: {e}")
    
    async def _run_cleanup(self, instance: Any, cleanup_func: Callable):
        """Ejecutar función de limpieza"""
        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func(instance)
            else:
                cleanup_func(instance)
        except Exception as e:
            logger.error(f"Error en cleanup: {e}")
    
    def _analyze_dependencies(self, factory: Factory) -> List[str]:
        """Analizar dependencias de la factory"""
        dependencies = []
        
        # Analizar constructor
        if hasattr(factory, '__init__'):
            signature = inspect.signature(factory.__init__)
            for param_name, param in signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    dependencies.append(param_name)
        
        return dependencies
    
    def get_factory(self, type_name: str) -> Optional[Factory]:
        """Obtener factory registrada"""
        return self._factories.get(type_name)
    
    def get_registered_types(self) -> List[str]:
        """Obtener tipos registrados"""
        return list(self._factories.keys())
    
    def is_registered(self, type_name: str) -> bool:
        """Verificar si tipo está registrado"""
        return type_name in self._factories
    
    def unregister(self, type_name: str):
        """Desregistrar factory"""
        with self._lock:
            if type_name in self._factories:
                # Cleanup de pools
                if type_name in self._pools:
                    pool = self._pools[type_name]
                    descriptor = self._descriptors[type_name]
                    
                    if descriptor.cleanup_func:
                        for instance in pool:
                            asyncio.create_task(
                                self._run_cleanup(instance, descriptor.cleanup_func)
                            )
                    
                    del self._pools[type_name]
                
                # Cleanup de singletons
                if type_name in self._instances:
                    del self._instances[type_name]
                
                del self._factories[type_name]
                del self._descriptors[type_name]
                
                logger.debug(f"Factory desregistrada: {type_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del registry"""
        metrics = {
            'registered_factories': len(self._factories),
            'singleton_instances': len(self._instances),
            'active_pools': len(self._pools),
            'factories': {}
        }
        
        for type_name, descriptor in self._descriptors.items():
            pool_info = {}
            if type_name in self._pools:
                pool_info = {
                    'pool_size': len(self._pools[type_name]),
                    'max_pool_size': descriptor.pool_size or self._default_pool_size
                }
            
            metrics['factories'][type_name] = {
                'strategy': descriptor.strategy.value,
                'created_instances': descriptor.created_instances,
                'max_instances': descriptor.max_instances,
                **pool_info
            }
        
        return metrics

class LazyProxy:
    """Proxy para creación lazy de objetos"""
    
    def __init__(self, factory: Factory, **kwargs):
        self._factory = factory
        self._kwargs = kwargs
        self._instance = None
        self._created = False
    
    async def _ensure_created(self):
        """Asegurar que el objeto está creado"""
        if not self._created:
            self._instance = await self._factory.create(**self._kwargs)
            self._created = True
    
    def __getattr__(self, name):
        if not self._created:
            # Esto requiere que la creación sea síncrona o use un mecanismo diferente
            raise RuntimeError("Objeto lazy no ha sido inicializado")
        return getattr(self._instance, name)

# Factories específicas para el dominio de trading

class ExchangeFactory(Factory):
    """Factory para crear conexiones a exchanges"""
    
    def __init__(self, di_container: DIContainer):
        self._di_container = di_container
        self._supported_exchanges = ["binance", "coinbase", "kraken", "okx"]
    
    @property
    def supported_types(self) -> List[str]:
        return self._supported_exchanges
    
    def can_create(self, type_name: str) -> bool:
        return type_name in self._supported_exchanges
    
    async def create(self, exchange_name: str = None, **config) -> Any:
        """Crear conexión a exchange"""
        if not exchange_name:
            raise ValueError("exchange_name requerido")
        
        if exchange_name not in self._supported_exchanges:
            raise ValueError(f"Exchange no soportado: {exchange_name}")
        
        # Aquí iría la lógica específica para crear cada exchange
        if exchange_name == "binance":
            return await self._create_binance_connection(**config)
        elif exchange_name == "coinbase":
            return await self._create_coinbase_connection(**config)
        elif exchange_name == "kraken":
            return await self._create_kraken_connection(**config)
        elif exchange_name == "okx":
            return await self._create_okx_connection(**config)
    
    async def _create_binance_connection(self, **config):
        """Crear conexión Binance"""
        from src.exchanges.binance_exchange import BinanceExchange
        return BinanceExchange(**config)
    
    async def _create_coinbase_connection(self, **config):
        """Crear conexión Coinbase"""
        from src.exchanges.coinbase_exchange import CoinbaseExchange
        return CoinbaseExchange(**config)
    
    async def _create_kraken_connection(self, **config):
        """Crear conexión Kraken"""
        from src.exchanges.kraken_exchange import KrakenExchange
        return KrakenExchange(**config)
    
    async def _create_okx_connection(self, **config):
        """Crear conexión OKX"""
        from src.exchanges.okx_exchange import OKXExchange
        return OKXExchange(**config)

class StrategyFactory(Factory):
    """Factory para crear estrategias de trading"""
    
    def __init__(self):
        self._strategies = {
            "arbitrage": "ArbitrageStrategy",
            "momentum": "MomentumStrategy", 
            "mean_reversion": "MeanReversionStrategy",
            "grid": "GridStrategy",
            "dca": "DCAStrategy"
        }
    
    @property
    def supported_types(self) -> List[str]:
        return list(self._strategies.keys())
    
    def can_create(self, type_name: str) -> bool:
        return type_name in self._strategies
    
    async def create(self, strategy_type: str = None, **config) -> Any:
        """Crear estrategia de trading"""
        if not strategy_type:
            raise ValueError("strategy_type requerido")
        
        if strategy_type not in self._strategies:
            raise ValueError(f"Estrategia no soportada: {strategy_type}")
        
        # Importación dinámica y creación
        module_name = f"src.strategies.{strategy_type}_strategy"
        class_name = self._strategies[strategy_type]
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            return strategy_class(**config)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error creando estrategia {strategy_type}: {e}")
            raise

class RepositoryFactory(Factory):
    """Factory para crear repositorios"""
    
    def __init__(self, db_pool=None):
        self._db_pool = db_pool
        self._repositories = [
            "user", "order", "portfolio", "transaction", 
            "market_data", "strategy", "backtest"
        ]
    
    @property
    def supported_types(self) -> List[str]:
        return self._repositories
    
    def can_create(self, type_name: str) -> bool:
        return type_name in self._repositories
    
    async def create(self, repo_type: str = None, **config) -> Any:
        """Crear repositorio"""
        if not repo_type:
            raise ValueError("repo_type requerido")
        
        if repo_type not in self._repositories:
            raise ValueError(f"Repositorio no soportado: {repo_type}")
        
        # Importación dinámica
        module_name = f"src.repositories.{repo_type}_repository"
        class_name = f"{repo_type.title()}Repository"
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            repo_class = getattr(module, class_name)
            
            # Inyectar dependencias
            if self._db_pool:
                config['db_pool'] = self._db_pool
            
            return repo_class(**config)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error creando repositorio {repo_type}: {e}")
            raise

# Utilidades para configuración masiva

def configure_trading_factories(registry: FactoryRegistry, di_container: DIContainer):
    """Configurar factories para trading"""
    
    # Exchange factory con pool
    exchange_factory = ExchangeFactory(di_container)
    registry.register_pooled_factory("exchange", exchange_factory, pool_size=3)
    
    # Strategy factory como prototype
    strategy_factory = StrategyFactory()
    registry.register_prototype_factory("strategy", strategy_factory)
    
    # Repository factory como singleton
    repo_factory = RepositoryFactory()
    registry.register_singleton_factory("repository", repo_factory)
    
    logger.info("Factories de trading configuradas")

# Decoradores para simplificar el uso

def factory_method(type_name: str, registry: FactoryRegistry):
    """Decorador para métodos que crean objetos vía factory"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Crear objeto usando factory
            instance = await registry.create(type_name, **kwargs)
            
            # Ejecutar función original con la instancia
            return await func(instance, *args, **kwargs)
        return wrapper
    return decorator