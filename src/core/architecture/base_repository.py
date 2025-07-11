"""
📊 Base Repository Pattern
Patrón Repository para acceso abstracto a datos
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
TId = TypeVar('TId')

class SortDirection(Enum):
    """Dirección de ordenamiento"""
    ASC = "asc"
    DESC = "desc"

@dataclass
class QueryFilter:
    """Filtro para consultas"""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, like, between
    value: Any
    
@dataclass
class SortOrder:
    """Orden de consulta"""
    field: str
    direction: SortDirection = SortDirection.ASC

@dataclass
class PageRequest:
    """Solicitud de paginación"""
    page: int = 1
    size: int = 20
    sort: List[SortOrder] = None
    
    def __post_init__(self):
        if self.sort is None:
            self.sort = []

@dataclass
class PageResult(Generic[T]):
    """Resultado paginado"""
    items: List[T]
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_previous: bool

class BaseRepository(ABC, Generic[T, TId]):
    """Repositorio base abstracto"""
    
    def __init__(self):
        self._entity_type: type = None
        self._id_type: type = None
    
    @abstractmethod
    async def find_by_id(self, id: TId) -> Optional[T]:
        """Buscar entidad por ID"""
        pass
    
    @abstractmethod
    async def find_all(self, page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar todas las entidades"""
        pass
    
    @abstractmethod
    async def find_by_criteria(self, filters: List[QueryFilter], 
                              page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar entidades por criterios"""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Guardar entidad"""
        pass
    
    @abstractmethod
    async def save_all(self, entities: List[T]) -> List[T]:
        """Guardar múltiples entidades"""
        pass
    
    @abstractmethod
    async def delete_by_id(self, id: TId) -> bool:
        """Eliminar entidad por ID"""
        pass
    
    @abstractmethod
    async def delete(self, entity: T) -> bool:
        """Eliminar entidad"""
        pass
    
    @abstractmethod
    async def delete_all(self, entities: List[T]) -> bool:
        """Eliminar múltiples entidades"""
        pass
    
    @abstractmethod
    async def exists_by_id(self, id: TId) -> bool:
        """Verificar si existe entidad por ID"""
        pass
    
    @abstractmethod
    async def count(self, filters: List[QueryFilter] = None) -> int:
        """Contar entidades"""
        pass
    
    # Métodos de conveniencia
    async def find_first(self, filters: List[QueryFilter] = None) -> Optional[T]:
        """Buscar primera entidad que coincida"""
        page_request = PageRequest(page=1, size=1)
        result = await self.find_by_criteria(filters or [], page_request)
        
        if isinstance(result, PageResult):
            return result.items[0] if result.items else None
        else:
            return result[0] if result else None
    
    async def find_by_field(self, field: str, value: Any) -> List[T]:
        """Buscar entidades por campo específico"""
        filter = QueryFilter(field=field, operator="eq", value=value)
        result = await self.find_by_criteria([filter])
        
        if isinstance(result, PageResult):
            return result.items
        else:
            return result

class AsyncPostgreSQLRepository(BaseRepository[T, TId]):
    """Repositorio PostgreSQL asíncrono"""
    
    def __init__(self, db_pool, table_name: str, entity_class: type):
        super().__init__()
        self.db_pool = db_pool
        self.table_name = table_name
        self.entity_class = entity_class
        self._entity_type = entity_class
    
    async def find_by_id(self, id: TId) -> Optional[T]:
        """Buscar por ID en PostgreSQL"""
        async with self.db_pool.acquire() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            row = await conn.fetchrow(query, id)
            
            if row:
                return self._row_to_entity(row)
            return None
    
    async def find_all(self, page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar todas las entidades"""
        async with self.db_pool.acquire() as conn:
            if page_request:
                # Consulta paginada
                offset = (page_request.page - 1) * page_request.size
                limit = page_request.size
                
                # Construir ORDER BY
                order_by = self._build_order_by(page_request.sort)
                
                # Consulta de datos
                query = f"SELECT * FROM {self.table_name} {order_by} LIMIT $1 OFFSET $2"
                rows = await conn.fetch(query, limit, offset)
                
                # Consulta de total
                count_query = f"SELECT COUNT(*) FROM {self.table_name}"
                total = await conn.fetchval(count_query)
                
                items = [self._row_to_entity(row) for row in rows]
                
                return PageResult(
                    items=items,
                    total_items=total,
                    total_pages=(total + page_request.size - 1) // page_request.size,
                    current_page=page_request.page,
                    page_size=page_request.size,
                    has_next=page_request.page * page_request.size < total,
                    has_previous=page_request.page > 1
                )
            else:
                # Consulta simple
                query = f"SELECT * FROM {self.table_name}"
                rows = await conn.fetch(query)
                return [self._row_to_entity(row) for row in rows]
    
    async def find_by_criteria(self, filters: List[QueryFilter], 
                              page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar por criterios"""
        async with self.db_pool.acquire() as conn:
            where_clause, params = self._build_where_clause(filters)
            
            if page_request:
                # Consulta paginada
                offset = (page_request.page - 1) * page_request.size
                limit = page_request.size
                order_by = self._build_order_by(page_request.sort)
                
                # Consulta de datos
                query = f"SELECT * FROM {self.table_name} {where_clause} {order_by} LIMIT ${len(params)+1} OFFSET ${len(params)+2}"
                rows = await conn.fetch(query, *params, limit, offset)
                
                # Consulta de total
                count_query = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
                total = await conn.fetchval(count_query, *params)
                
                items = [self._row_to_entity(row) for row in rows]
                
                return PageResult(
                    items=items,
                    total_items=total,
                    total_pages=(total + page_request.size - 1) // page_request.size,
                    current_page=page_request.page,
                    page_size=page_request.size,
                    has_next=page_request.page * page_request.size < total,
                    has_previous=page_request.page > 1
                )
            else:
                # Consulta simple
                query = f"SELECT * FROM {self.table_name} {where_clause}"
                rows = await conn.fetch(query, *params)
                return [self._row_to_entity(row) for row in rows]
    
    async def save(self, entity: T) -> T:
        """Guardar entidad"""
        async with self.db_pool.acquire() as conn:
            # Determinar si es insert o update
            entity_dict = self._entity_to_dict(entity)
            
            if 'id' in entity_dict and entity_dict['id']:
                # Update
                set_clause, params = self._build_update_clause(entity_dict)
                query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ${len(params)+1} RETURNING *"
                row = await conn.fetchrow(query, *params, entity_dict['id'])
            else:
                # Insert
                columns = ', '.join(entity_dict.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(entity_dict)))
                query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders}) RETURNING *"
                row = await conn.fetchrow(query, *entity_dict.values())
            
            return self._row_to_entity(row)
    
    async def save_all(self, entities: List[T]) -> List[T]:
        """Guardar múltiples entidades"""
        results = []
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                for entity in entities:
                    saved_entity = await self.save(entity)
                    results.append(saved_entity)
        return results
    
    async def delete_by_id(self, id: TId) -> bool:
        """Eliminar por ID"""
        async with self.db_pool.acquire() as conn:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            result = await conn.execute(query, id)
            return result == "DELETE 1"
    
    async def delete(self, entity: T) -> bool:
        """Eliminar entidad"""
        entity_dict = self._entity_to_dict(entity)
        if 'id' not in entity_dict:
            raise ValueError("Entidad debe tener ID para eliminar")
        return await self.delete_by_id(entity_dict['id'])
    
    async def delete_all(self, entities: List[T]) -> bool:
        """Eliminar múltiples entidades"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                for entity in entities:
                    await self.delete(entity)
        return True
    
    async def exists_by_id(self, id: TId) -> bool:
        """Verificar existencia por ID"""
        async with self.db_pool.acquire() as conn:
            query = f"SELECT 1 FROM {self.table_name} WHERE id = $1"
            result = await conn.fetchval(query, id)
            return result is not None
    
    async def count(self, filters: List[QueryFilter] = None) -> int:
        """Contar entidades"""
        async with self.db_pool.acquire() as conn:
            if filters:
                where_clause, params = self._build_where_clause(filters)
                query = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
                return await conn.fetchval(query, *params)
            else:
                query = f"SELECT COUNT(*) FROM {self.table_name}"
                return await conn.fetchval(query)
    
    def _row_to_entity(self, row) -> T:
        """Convertir row de DB a entidad"""
        if hasattr(self.entity_class, 'from_dict'):
            return self.entity_class.from_dict(dict(row))
        else:
            return self.entity_class(**dict(row))
    
    def _entity_to_dict(self, entity: T) -> dict:
        """Convertir entidad a diccionario"""
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        elif hasattr(entity, '__dict__'):
            return entity.__dict__.copy()
        else:
            raise ValueError(f"No se puede convertir {type(entity)} a dict")
    
    def _build_where_clause(self, filters: List[QueryFilter]) -> tuple:
        """Construir cláusula WHERE"""
        if not filters:
            return "", []
        
        conditions = []
        params = []
        
        for i, filter in enumerate(filters):
            param_index = i + 1
            
            if filter.operator == "eq":
                conditions.append(f"{filter.field} = ${param_index}")
                params.append(filter.value)
            elif filter.operator == "ne":
                conditions.append(f"{filter.field} != ${param_index}")
                params.append(filter.value)
            elif filter.operator == "gt":
                conditions.append(f"{filter.field} > ${param_index}")
                params.append(filter.value)
            elif filter.operator == "lt":
                conditions.append(f"{filter.field} < ${param_index}")
                params.append(filter.value)
            elif filter.operator == "gte":
                conditions.append(f"{filter.field} >= ${param_index}")
                params.append(filter.value)
            elif filter.operator == "lte":
                conditions.append(f"{filter.field} <= ${param_index}")
                params.append(filter.value)
            elif filter.operator == "like":
                conditions.append(f"{filter.field} LIKE ${param_index}")
                params.append(filter.value)
            elif filter.operator == "in":
                placeholders = ', '.join(f'${param_index + j}' for j in range(len(filter.value)))
                conditions.append(f"{filter.field} IN ({placeholders})")
                params.extend(filter.value)
            # Agregar más operadores según necesidad
        
        where_clause = "WHERE " + " AND ".join(conditions)
        return where_clause, params
    
    def _build_order_by(self, sort_orders: List[SortOrder]) -> str:
        """Construir cláusula ORDER BY"""
        if not sort_orders:
            return ""
        
        order_parts = []
        for sort_order in sort_orders:
            direction = "ASC" if sort_order.direction == SortDirection.ASC else "DESC"
            order_parts.append(f"{sort_order.field} {direction}")
        
        return "ORDER BY " + ", ".join(order_parts)
    
    def _build_update_clause(self, entity_dict: dict) -> tuple:
        """Construir cláusula UPDATE SET"""
        # Excluir ID del update
        update_dict = {k: v for k, v in entity_dict.items() if k != 'id'}
        
        set_parts = []
        params = []
        
        for i, (field, value) in enumerate(update_dict.items()):
            set_parts.append(f"{field} = ${i+1}")
            params.append(value)
        
        return ", ".join(set_parts), params

class InMemoryRepository(BaseRepository[T, TId]):
    """Repositorio en memoria para testing"""
    
    def __init__(self, id_field: str = 'id'):
        super().__init__()
        self._data: Dict[TId, T] = {}
        self._id_field = id_field
        self._next_id = 1
    
    async def find_by_id(self, id: TId) -> Optional[T]:
        """Buscar por ID"""
        return self._data.get(id)
    
    async def find_all(self, page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar todas las entidades"""
        items = list(self._data.values())
        
        if page_request:
            # Aplicar ordenamiento
            if page_request.sort:
                for sort_order in reversed(page_request.sort):
                    reverse = sort_order.direction == SortDirection.DESC
                    items.sort(key=lambda x: getattr(x, sort_order.field), reverse=reverse)
            
            # Aplicar paginación
            total = len(items)
            start = (page_request.page - 1) * page_request.size
            end = start + page_request.size
            page_items = items[start:end]
            
            return PageResult(
                items=page_items,
                total_items=total,
                total_pages=(total + page_request.size - 1) // page_request.size,
                current_page=page_request.page,
                page_size=page_request.size,
                has_next=end < total,
                has_previous=page_request.page > 1
            )
        else:
            return items
    
    async def find_by_criteria(self, filters: List[QueryFilter], 
                              page_request: PageRequest = None) -> Union[List[T], PageResult[T]]:
        """Buscar por criterios"""
        items = []
        
        for entity in self._data.values():
            if self._matches_filters(entity, filters):
                items.append(entity)
        
        if page_request:
            # Aplicar ordenamiento y paginación similar a find_all
            if page_request.sort:
                for sort_order in reversed(page_request.sort):
                    reverse = sort_order.direction == SortDirection.DESC
                    items.sort(key=lambda x: getattr(x, sort_order.field), reverse=reverse)
            
            total = len(items)
            start = (page_request.page - 1) * page_request.size
            end = start + page_request.size
            page_items = items[start:end]
            
            return PageResult(
                items=page_items,
                total_items=total,
                total_pages=(total + page_request.size - 1) // page_request.size,
                current_page=page_request.page,
                page_size=page_request.size,
                has_next=end < total,
                has_previous=page_request.page > 1
            )
        else:
            return items
    
    def _matches_filters(self, entity: T, filters: List[QueryFilter]) -> bool:
        """Verificar si entidad coincide con filtros"""
        for filter in filters:
            entity_value = getattr(entity, filter.field, None)
            
            if filter.operator == "eq" and entity_value != filter.value:
                return False
            elif filter.operator == "ne" and entity_value == filter.value:
                return False
            elif filter.operator == "gt" and entity_value <= filter.value:
                return False
            elif filter.operator == "lt" and entity_value >= filter.value:
                return False
            elif filter.operator == "gte" and entity_value < filter.value:
                return False
            elif filter.operator == "lte" and entity_value > filter.value:
                return False
            elif filter.operator == "like" and filter.value not in str(entity_value):
                return False
            elif filter.operator == "in" and entity_value not in filter.value:
                return False
        
        return True
    
    async def save(self, entity: T) -> T:
        """Guardar entidad"""
        entity_id = getattr(entity, self._id_field, None)
        
        if entity_id is None:
            # Asignar nuevo ID
            entity_id = self._next_id
            setattr(entity, self._id_field, entity_id)
            self._next_id += 1
        
        self._data[entity_id] = entity
        return entity
    
    async def save_all(self, entities: List[T]) -> List[T]:
        """Guardar múltiples entidades"""
        results = []
        for entity in entities:
            saved_entity = await self.save(entity)
            results.append(saved_entity)
        return results
    
    async def delete_by_id(self, id: TId) -> bool:
        """Eliminar por ID"""
        if id in self._data:
            del self._data[id]
            return True
        return False
    
    async def delete(self, entity: T) -> bool:
        """Eliminar entidad"""
        entity_id = getattr(entity, self._id_field, None)
        if entity_id is not None:
            return await self.delete_by_id(entity_id)
        return False
    
    async def delete_all(self, entities: List[T]) -> bool:
        """Eliminar múltiples entidades"""
        for entity in entities:
            await self.delete(entity)
        return True
    
    async def exists_by_id(self, id: TId) -> bool:
        """Verificar existencia por ID"""
        return id in self._data
    
    async def count(self, filters: List[QueryFilter] = None) -> int:
        """Contar entidades"""
        if filters:
            count = 0
            for entity in self._data.values():
                if self._matches_filters(entity, filters):
                    count += 1
            return count
        else:
            return len(self._data)