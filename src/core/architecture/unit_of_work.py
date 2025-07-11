"""
🔄 Unit of Work Pattern
Patrón para gestionar transacciones y coordinación entre repositorios
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar, Any, List, Optional
from contextlib import asynccontextmanager
import asyncpg
from .base_repository import BaseRepository
from .dependency_injection import injectable

logger = logging.getLogger(__name__)

T = TypeVar('T')

class UnitOfWorkBase(ABC):
    """Unidad de trabajo base abstracta"""
    
    def __init__(self):
        self._repositories: Dict[Type, BaseRepository] = {}
        self._is_committed = False
        self._is_rolled_back = False
    
    @abstractmethod
    async def __aenter__(self):
        """Iniciar unidad de trabajo"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Finalizar unidad de trabajo"""
        pass
    
    @abstractmethod
    async def commit(self):
        """Confirmar cambios"""
        pass
    
    @abstractmethod
    async def rollback(self):
        """Revertir cambios"""
        pass
    
    def add_repository(self, repo_type: Type[T], repository: BaseRepository[T, Any]):
        """Agregar repositorio a la unidad de trabajo"""
        self._repositories[repo_type] = repository
    
    def get_repository(self, repo_type: Type[T]) -> Optional[BaseRepository[T, Any]]:
        """Obtener repositorio por tipo"""
        return self._repositories.get(repo_type)

@injectable
class PostgreSQLUnitOfWork(UnitOfWorkBase):
    """Unit of Work para PostgreSQL con soporte completo de transacciones"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        super().__init__()
        self.db_pool = db_pool
        self._connection: Optional[asyncpg.Connection] = None
        self._transaction: Optional[asyncpg.Transaction] = None
        self._nested_transactions: List[asyncpg.Transaction] = []
    
    async def __aenter__(self):
        """Iniciar transacción"""
        self._connection = await self.db_pool.acquire()
        self._transaction = self._connection.transaction()
        await self._transaction.start()
        
        logger.debug("Iniciada transacción PostgreSQL")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Finalizar transacción automáticamente"""
        try:
            if exc_type is not None:
                # Hubo una excepción, hacer rollback
                await self.rollback()
                logger.debug(f"Rollback automático debido a excepción: {exc_type.__name__}")
            elif not self._is_committed and not self._is_rolled_back:
                # No se hizo commit explícito, hacer rollback por seguridad
                await self.rollback()
                logger.debug("Rollback automático - no se hizo commit explícito")
        finally:
            # Liberar conexión al pool
            if self._connection:
                await self.db_pool.release(self._connection)
                self._connection = None
                self._transaction = None
    
    async def commit(self):
        """Confirmar todos los cambios"""
        if self._is_committed:
            raise RuntimeError("La transacción ya fue confirmada")
        
        if self._is_rolled_back:
            raise RuntimeError("No se puede confirmar una transacción revertida")
        
        try:
            # Commit de transacciones anidadas primero
            while self._nested_transactions:
                nested_tx = self._nested_transactions.pop()
                await nested_tx.commit()
            
            # Commit de la transacción principal
            if self._transaction:
                await self._transaction.commit()
                self._is_committed = True
                logger.debug("Transacción PostgreSQL confirmada exitosamente")
        
        except Exception as e:
            await self.rollback()
            logger.error(f"Error al confirmar transacción: {e}")
            raise
    
    async def rollback(self):
        """Revertir todos los cambios"""
        if self._is_rolled_back:
            return  # Ya fue revertida
        
        try:
            # Rollback de transacciones anidadas
            while self._nested_transactions:
                nested_tx = self._nested_transactions.pop()
                try:
                    await nested_tx.rollback()
                except Exception as e:
                    logger.warning(f"Error en rollback de transacción anidada: {e}")
            
            # Rollback de la transacción principal
            if self._transaction:
                await self._transaction.rollback()
                self._is_rolled_back = True
                logger.debug("Transacción PostgreSQL revertida")
        
        except Exception as e:
            logger.error(f"Error al revertir transacción: {e}")
            raise
    
    async def create_savepoint(self, name: str) -> 'SavePoint':
        """Crear savepoint para transacciones anidadas"""
        if not self._connection:
            raise RuntimeError("No hay conexión activa")
        
        savepoint = await self._connection.transaction(savepoint=name)
        await savepoint.start()
        self._nested_transactions.append(savepoint)
        
        logger.debug(f"Savepoint '{name}' creado")
        return SavePoint(self, savepoint, name)
    
    def get_connection(self) -> asyncpg.Connection:
        """Obtener conexión actual"""
        if not self._connection:
            raise RuntimeError("No hay conexión activa en la unidad de trabajo")
        return self._connection

@injectable  
class InMemoryUnitOfWork(UnitOfWorkBase):
    """Unit of Work en memoria para testing"""
    
    def __init__(self):
        super().__init__()
        self._changes: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        """Iniciar unidad de trabajo en memoria"""
        self._changes.clear()
        logger.debug("Iniciada unidad de trabajo en memoria")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Finalizar unidad de trabajo en memoria"""
        if exc_type is not None:
            await self.rollback()
        elif not self._is_committed and not self._is_rolled_back:
            await self.rollback()
    
    async def commit(self):
        """Confirmar cambios en memoria"""
        if self._is_committed:
            raise RuntimeError("La unidad de trabajo ya fue confirmada")
        
        # En implementación real aquí se aplicarían los cambios
        self._is_committed = True
        logger.debug(f"Unidad de trabajo en memoria confirmada con {len(self._changes)} cambios")
    
    async def rollback(self):
        """Revertir cambios en memoria"""
        if self._is_rolled_back:
            return
        
        # En implementación real aquí se revertirían los cambios
        self._changes.clear()
        self._is_rolled_back = True
        logger.debug("Unidad de trabajo en memoria revertida")
    
    def track_change(self, operation: str, entity_type: str, entity_id: Any, data: Any):
        """Registrar cambio para aplicar en commit"""
        change = {
            'operation': operation,  # insert, update, delete
            'entity_type': entity_type,
            'entity_id': entity_id,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        }
        self._changes.append(change)

class SavePoint:
    """Contexto para manejar savepoints"""
    
    def __init__(self, uow: PostgreSQLUnitOfWork, transaction: asyncpg.Transaction, name: str):
        self.uow = uow
        self.transaction = transaction
        self.name = name
        self._is_committed = False
        self._is_rolled_back = False
    
    async def commit(self):
        """Confirmar savepoint"""
        if not self._is_committed and not self._is_rolled_back:
            await self.transaction.commit()
            self._is_committed = True
            logger.debug(f"Savepoint '{self.name}' confirmado")
    
    async def rollback(self):
        """Revertir savepoint"""
        if not self._is_rolled_back:
            await self.transaction.rollback()
            self._is_rolled_back = True
            logger.debug(f"Savepoint '{self.name}' revertido")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.rollback()
        elif not self._is_committed:
            await self.rollback()

class UnitOfWorkManager:
    """Gestor centralizado de unidades de trabajo"""
    
    def __init__(self):
        self._factories: Dict[str, Any] = {}
        self._default_type = "postgresql"
    
    def register_factory(self, uow_type: str, factory_func):
        """Registrar factory para tipo de UoW"""
        self._factories[uow_type] = factory_func
    
    def set_default_type(self, uow_type: str):
        """Establecer tipo por defecto"""
        self._default_type = uow_type
    
    @asynccontextmanager
    async def create_unit_of_work(self, uow_type: str = None) -> UnitOfWorkBase:
        """Crear y gestionar unidad de trabajo"""
        uow_type = uow_type or self._default_type
        
        if uow_type not in self._factories:
            raise ValueError(f"Tipo de UoW no registrado: {uow_type}")
        
        factory = self._factories[uow_type]
        uow = factory()
        
        async with uow:
            yield uow

# Patrón Repository con Unit of Work integrado
class RepositoryWithUoW(BaseRepository[T, Any]):
    """Repositorio que utiliza Unit of Work"""
    
    def __init__(self, base_repository: BaseRepository[T, Any], unit_of_work: UnitOfWorkBase):
        super().__init__()
        self._base_repo = base_repository
        self._uow = unit_of_work
    
    async def find_by_id(self, id: Any) -> Optional[T]:
        return await self._base_repo.find_by_id(id)
    
    async def find_all(self, page_request=None):
        return await self._base_repo.find_all(page_request)
    
    async def find_by_criteria(self, filters, page_request=None):
        return await self._base_repo.find_by_criteria(filters, page_request)
    
    async def save(self, entity: T) -> T:
        """Guardar con tracking de cambios en UoW"""
        result = await self._base_repo.save(entity)
        
        # Registrar cambio en UoW si es InMemory
        if isinstance(self._uow, InMemoryUnitOfWork):
            entity_id = getattr(entity, 'id', None)
            self._uow.track_change('save', type(entity).__name__, entity_id, result)
        
        return result
    
    async def save_all(self, entities: List[T]) -> List[T]:
        results = []
        for entity in entities:
            result = await self.save(entity)
            results.append(result)
        return results
    
    async def delete_by_id(self, id: Any) -> bool:
        result = await self._base_repo.delete_by_id(id)
        
        # Registrar cambio en UoW si es InMemory
        if isinstance(self._uow, InMemoryUnitOfWork) and result:
            self._uow.track_change('delete', 'unknown', id, None)
        
        return result
    
    async def delete(self, entity: T) -> bool:
        result = await self._base_repo.delete(entity)
        
        # Registrar cambio en UoW si es InMemory
        if isinstance(self._uow, InMemoryUnitOfWork) and result:
            entity_id = getattr(entity, 'id', None)
            self._uow.track_change('delete', type(entity).__name__, entity_id, entity)
        
        return result
    
    async def delete_all(self, entities: List[T]) -> bool:
        return await self._base_repo.delete_all(entities)
    
    async def exists_by_id(self, id: Any) -> bool:
        return await self._base_repo.exists_by_id(id)
    
    async def count(self, filters=None) -> int:
        return await self._base_repo.count(filters)

# Gestores de contexto para uso fácil
@asynccontextmanager
async def postgresql_unit_of_work(db_pool: asyncpg.Pool):
    """Context manager para UoW PostgreSQL"""
    async with PostgreSQLUnitOfWork(db_pool) as uow:
        yield uow

@asynccontextmanager  
async def memory_unit_of_work():
    """Context manager para UoW en memoria"""
    async with InMemoryUnitOfWork() as uow:
        yield uow

# Decorador para operaciones transaccionales
def transactional(uow_manager: UnitOfWorkManager, uow_type: str = None):
    """Decorador para métodos que requieren transacción"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with uow_manager.create_unit_of_work(uow_type) as uow:
                # Inyectar UoW como primer parámetro
                return await func(uow, *args, **kwargs)
        return wrapper
    return decorator

# Patrón Aggregate Root con UoW
class AggregateRoot:
    """Raíz de agregado que coordina cambios"""
    
    def __init__(self):
        self._domain_events: List[Any] = []
        self._is_dirty = False
    
    def mark_as_dirty(self):
        """Marcar agregado como modificado"""
        self._is_dirty = True
    
    def mark_as_clean(self):
        """Marcar agregado como limpio"""
        self._is_dirty = False
        self._domain_events.clear()
    
    def add_domain_event(self, event: Any):
        """Agregar evento de dominio"""
        self._domain_events.append(event)
        self.mark_as_dirty()
    
    def get_domain_events(self) -> List[Any]:
        """Obtener eventos de dominio"""
        return self._domain_events.copy()
    
    def clear_domain_events(self):
        """Limpiar eventos de dominio"""
        self._domain_events.clear()

# Utilidades para coordinación compleja
class TransactionScope:
    """Scope para transacciones distribuidas"""
    
    def __init__(self):
        self._units_of_work: List[UnitOfWorkBase] = []
        self._is_completed = False
    
    def enlist(self, uow: UnitOfWorkBase):
        """Enlistar UoW en la transacción distribuida"""
        if self._is_completed:
            raise RuntimeError("No se puede enlistar en transacción completada")
        self._units_of_work.append(uow)
    
    async def complete(self):
        """Confirmar todas las UoW"""
        if self._is_completed:
            raise RuntimeError("Transacción ya completada")
        
        # Patrón 2PC simplificado
        try:
            # Fase 1: Preparar todas las transacciones
            for uow in self._units_of_work:
                # En implementación real verificaríamos si pueden hacer commit
                pass
            
            # Fase 2: Confirmar todas
            for uow in self._units_of_work:
                await uow.commit()
            
            self._is_completed = True
            logger.debug(f"Transacción distribuida completada con {len(self._units_of_work)} UoW")
        
        except Exception as e:
            # Rollback de todas las UoW
            for uow in self._units_of_work:
                try:
                    await uow.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error en rollback distribuido: {rollback_error}")
            
            logger.error(f"Error en transacción distribuida: {e}")
            raise

# Factory para crear repositorios con UoW
class RepositoryFactory:
    """Factory para crear repositorios integrados con UoW"""
    
    @staticmethod
    def create_repository_with_uow(
        repository_class: Type[BaseRepository], 
        uow: UnitOfWorkBase,
        *args, **kwargs
    ) -> RepositoryWithUoW:
        """Crear repositorio integrado con UoW"""
        base_repo = repository_class(*args, **kwargs)
        return RepositoryWithUoW(base_repo, uow)