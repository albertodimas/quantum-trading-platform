"""
User Repository

Handles database operations for users, roles, and permissions.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
import asyncpg
from asyncpg import Pool

from ..core.architecture import BaseRepository, injectable
from ..core.observability import get_logger
from .models import (
    User, UserCreate, UserUpdate,
    Role, Permission, UserRole,
    APIKey, PermissionType
)

logger = get_logger(__name__)


@injectable
class UserRepository(BaseRepository[User]):
    """Repository for user management"""
    
    def __init__(self, db_pool: Pool):
        super().__init__(db_pool, User, "quantum.users")
        self.schema = "quantum"
        
    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create new user
        
        Args:
            user_data: User creation data
            
        Returns:
            Created user object
        """
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name
        )
        user.set_password(user_data.password)
        
        async with self.db_pool.acquire() as conn:
            # Insert user
            row = await conn.fetchrow("""
                INSERT INTO quantum.users 
                (id, email, username, password_hash, full_name, is_active, is_superuser)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            """, user.id, user.email, user.username, user.password_hash,
                user.full_name, user.is_active, user.is_superuser)
            
            if row:
                logger.info(f"User created: {user.username}")
                return User(**dict(row))
            
            raise Exception("Failed to create user")
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT u.*, 
                       array_agg(DISTINCT r.id) FILTER (WHERE r.id IS NOT NULL) as role_ids
                FROM quantum.users u
                LEFT JOIN quantum.user_roles ur ON u.id = ur.user_id
                LEFT JOIN quantum.roles r ON ur.role_id = r.id
                WHERE u.username = $1
                GROUP BY u.id
            """, username)
            
            if row:
                user_dict = dict(row)
                user = User(**user_dict)
                
                # Load roles if any
                if user_dict.get('role_ids'):
                    user.roles = await self._get_roles_by_ids(user_dict['role_ids'])
                    
                return user
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT u.*, 
                       array_agg(DISTINCT r.id) FILTER (WHERE r.id IS NOT NULL) as role_ids
                FROM quantum.users u
                LEFT JOIN quantum.user_roles ur ON u.id = ur.user_id
                LEFT JOIN quantum.roles r ON ur.role_id = r.id
                WHERE u.email = $1
                GROUP BY u.id
            """, email)
            
            if row:
                user_dict = dict(row)
                user = User(**user_dict)
                
                # Load roles if any
                if user_dict.get('role_ids'):
                    user.roles = await self._get_roles_by_ids(user_dict['role_ids'])
                    
                return user
            return None
    
    async def get_with_roles(self, user_id: UUID) -> Optional[User]:
        """Get user with roles and permissions loaded"""
        async with self.db_pool.acquire() as conn:
            # Get user with roles
            row = await conn.fetchrow("""
                SELECT u.*, 
                       array_agg(DISTINCT r.id) FILTER (WHERE r.id IS NOT NULL) as role_ids
                FROM quantum.users u
                LEFT JOIN quantum.user_roles ur ON u.id = ur.user_id
                LEFT JOIN quantum.roles r ON ur.role_id = r.id
                WHERE u.id = $1
                GROUP BY u.id
            """, user_id)
            
            if row:
                user_dict = dict(row)
                user = User(**user_dict)
                
                # Load roles with permissions
                if user_dict.get('role_ids'):
                    user.roles = await self._get_roles_by_ids(user_dict['role_ids'])
                    
                return user
            return None
    
    async def update_user(
        self, 
        user_id: UUID, 
        update_data: UserUpdate
    ) -> Optional[User]:
        """Update user information"""
        update_fields = []
        values = []
        param_count = 1
        
        # Build dynamic update query
        for field, value in update_data.dict(exclude_unset=True).items():
            if value is not None:
                update_fields.append(f"{field} = ${param_count}")
                values.append(value)
                param_count += 1
        
        if not update_fields:
            return await self.get_with_roles(user_id)
        
        # Add updated_at
        update_fields.append(f"updated_at = ${param_count}")
        values.append(datetime.utcnow())
        param_count += 1
        
        # Add user_id for WHERE clause
        values.append(user_id)
        
        async with self.db_pool.acquire() as conn:
            query = f"""
                UPDATE quantum.users 
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
                RETURNING *
            """
            
            row = await conn.fetchrow(query, *values)
            
            if row:
                logger.info(f"User updated: {user_id}")
                return await self.get_with_roles(user_id)
            return None
    
    async def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE quantum.users 
                SET last_login = $1
                WHERE id = $2
            """, datetime.utcnow(), user_id)
            
            logger.info(f"Updated last login for user: {user_id}")
    
    async def assign_role(self, user_id: UUID, role_id: UUID) -> bool:
        """Assign role to user"""
        async with self.db_pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO quantum.user_roles (user_id, role_id)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id, role_id) DO NOTHING
                """, user_id, role_id)
                
                logger.info(f"Role {role_id} assigned to user {user_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to assign role: {str(e)}")
                return False
    
    async def remove_role(self, user_id: UUID, role_id: UUID) -> bool:
        """Remove role from user"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM quantum.user_roles
                WHERE user_id = $1 AND role_id = $2
            """, user_id, role_id)
            
            if result.split()[-1] == '1':
                logger.info(f"Role {role_id} removed from user {user_id}")
                return True
            return False
    
    async def _get_roles_by_ids(self, role_ids: List[UUID]) -> List[Role]:
        """Get roles by IDs with permissions"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT r.*, 
                       array_agg(
                           json_build_object(
                               'id', p.id,
                               'name', p.name,
                               'description', p.description
                           )
                       ) FILTER (WHERE p.id IS NOT NULL) as permissions
                FROM quantum.roles r
                LEFT JOIN quantum.role_permissions rp ON r.id = rp.role_id
                LEFT JOIN quantum.permissions p ON rp.permission_id = p.id
                WHERE r.id = ANY($1::uuid[])
                GROUP BY r.id
            """, role_ids)
            
            roles = []
            for row in rows:
                role_dict = dict(row)
                permissions_data = role_dict.pop('permissions', [])
                
                role = Role(**role_dict)
                if permissions_data:
                    role.permissions = [
                        Permission(**perm) for perm in permissions_data
                    ]
                    
                roles.append(role)
                
            return roles
    
    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        include_inactive: bool = False
    ) -> List[User]:
        """List users with pagination"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT u.*, 
                       array_agg(DISTINCT r.id) FILTER (WHERE r.id IS NOT NULL) as role_ids
                FROM quantum.users u
                LEFT JOIN quantum.user_roles ur ON u.id = ur.user_id
                LEFT JOIN quantum.roles r ON ur.role_id = r.id
            """
            
            if not include_inactive:
                query += " WHERE u.is_active = TRUE"
                
            query += """
                GROUP BY u.id
                ORDER BY u.created_at DESC
                LIMIT $1 OFFSET $2
            """
            
            rows = await conn.fetch(query, limit, skip)
            
            users = []
            for row in rows:
                user_dict = dict(row)
                user = User(**user_dict)
                
                if user_dict.get('role_ids'):
                    user.roles = await self._get_roles_by_ids(user_dict['role_ids'])
                    
                users.append(user)
                
            return users
    
    async def search_users(
        self,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Search users by username or email"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT u.*, 
                       array_agg(DISTINCT r.id) FILTER (WHERE r.id IS NOT NULL) as role_ids
                FROM quantum.users u
                LEFT JOIN quantum.user_roles ur ON u.id = ur.user_id
                LEFT JOIN quantum.roles r ON ur.role_id = r.id
                WHERE u.username ILIKE $1 OR u.email ILIKE $1
                GROUP BY u.id
                ORDER BY u.created_at DESC
                LIMIT $2 OFFSET $3
            """, f"%{search_term}%", limit, skip)
            
            users = []
            for row in rows:
                user_dict = dict(row)
                user = User(**user_dict)
                
                if user_dict.get('role_ids'):
                    user.roles = await self._get_roles_by_ids(user_dict['role_ids'])
                    
                users.append(user)
                
            return users


@injectable
class RoleRepository(BaseRepository[Role]):
    """Repository for role management"""
    
    def __init__(self, db_pool: Pool):
        super().__init__(db_pool, Role, "quantum.roles")
        
    async def create_role(
        self,
        name: str,
        description: str,
        permissions: List[PermissionType]
    ) -> Role:
        """Create new role with permissions"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Create role
                role_row = await conn.fetchrow("""
                    INSERT INTO quantum.roles (name, description)
                    VALUES ($1, $2)
                    RETURNING *
                """, name, description)
                
                role = Role(**dict(role_row))
                
                # Assign permissions
                if permissions:
                    for perm_type in permissions:
                        await conn.execute("""
                            INSERT INTO quantum.role_permissions (role_id, permission_id)
                            SELECT $1, id FROM quantum.permissions WHERE name = $2
                        """, role.id, perm_type.value)
                    
                    # Load permissions
                    role.permissions = await self._get_role_permissions(role.id)
                
                logger.info(f"Role created: {name}")
                return role
    
    async def get_by_name(self, name: str) -> Optional[Role]:
        """Get role by name"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM quantum.roles WHERE name = $1
            """, name)
            
            if row:
                role = Role(**dict(row))
                role.permissions = await self._get_role_permissions(role.id)
                return role
            return None
    
    async def _get_role_permissions(self, role_id: UUID) -> List[Permission]:
        """Get permissions for a role"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT p.* 
                FROM quantum.permissions p
                JOIN quantum.role_permissions rp ON p.id = rp.permission_id
                WHERE rp.role_id = $1
            """, role_id)
            
            return [Permission(**dict(row)) for row in rows]
    
    async def update_role_permissions(
        self,
        role_id: UUID,
        permissions: List[PermissionType]
    ) -> bool:
        """Update role permissions"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Remove existing permissions
                await conn.execute("""
                    DELETE FROM quantum.role_permissions WHERE role_id = $1
                """, role_id)
                
                # Add new permissions
                for perm_type in permissions:
                    await conn.execute("""
                        INSERT INTO quantum.role_permissions (role_id, permission_id)
                        SELECT $1, id FROM quantum.permissions WHERE name = $2
                    """, role_id, perm_type.value)
                
                logger.info(f"Updated permissions for role: {role_id}")
                return True


@injectable
class APIKeyRepository:
    """Repository for API key management"""
    
    def __init__(self, db_pool: Pool):
        self.db_pool = db_pool
        
    async def create_api_key(
        self,
        user_id: UUID,
        key_hash: str,
        name: str,
        permissions: List[PermissionType],
        expires_at: Optional[datetime] = None
    ) -> APIKey:
        """Create new API key"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO quantum.api_keys 
                (user_id, key_hash, name, permissions, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """, user_id, key_hash, name, 
                [p.value for p in permissions], expires_at)
            
            if row:
                logger.info(f"API key created: {name} for user {user_id}")
                return APIKey(**dict(row))
                
            raise Exception("Failed to create API key")
    
    async def get_by_key_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM quantum.api_keys
                WHERE key_hash = $1 AND is_active = TRUE
            """, key_hash)
            
            if row:
                api_key = APIKey(**dict(row))
                
                # Update last used
                await conn.execute("""
                    UPDATE quantum.api_keys 
                    SET last_used_at = $1
                    WHERE id = $2
                """, datetime.utcnow(), api_key.id)
                
                return api_key
            return None
    
    async def list_user_api_keys(
        self,
        user_id: UUID,
        include_inactive: bool = False
    ) -> List[APIKey]:
        """List user's API keys"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM quantum.api_keys
                WHERE user_id = $1
            """
            
            if not include_inactive:
                query += " AND is_active = TRUE"
                
            query += " ORDER BY created_at DESC"
            
            rows = await conn.fetch(query, user_id)
            return [APIKey(**dict(row)) for row in rows]
    
    async def revoke_api_key(self, key_id: UUID, user_id: UUID) -> bool:
        """Revoke API key"""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE quantum.api_keys 
                SET is_active = FALSE
                WHERE id = $1 AND user_id = $2
            """, key_id, user_id)
            
            if result.split()[-1] == '1':
                logger.info(f"API key revoked: {key_id}")
                return True
            return False