"""
Authentication Dependencies

FastAPI dependency injection functions for authentication.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from asyncpg.pool import Pool

from ..core.observability import get_logger
from ..core.database import get_db_pool
from .models import User
from .service import AuthService
from .repository import UserRepository, RoleRepository, APIKeyRepository
from .jwt_handler import get_jwt_handler

logger = get_logger(__name__)

# OAuth2 scheme for JWT tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_auth_service(
    db_pool: Pool = Depends(get_db_pool)
) -> AuthService:
    """
    Get authentication service instance
    
    Creates AuthService with all required dependencies.
    """
    user_repo = UserRepository(db_pool)
    role_repo = RoleRepository(db_pool)
    api_key_repo = APIKeyRepository(db_pool)
    
    return AuthService(
        user_repo=user_repo,
        role_repo=role_repo,
        api_key_repo=api_key_repo
    )


async def get_current_user_from_token(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[User]:
    """
    Get current user from JWT token
    
    Validates JWT token and returns user object.
    """
    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


async def get_current_user_from_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[User]:
    """
    Get current user from API key
    
    Validates API key and returns associated user object.
    """
    if not api_key:
        return None
        
    result = await auth_service.validate_api_key(api_key)
    if not result:
        return None
        
    api_key_obj, user = result
    return user


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """
    Get current authenticated user
    
    Tries JWT token first, then API key. Returns authenticated user
    or raises 401 if neither authentication method succeeds.
    """
    # Try token authentication first
    if token_user:
        return token_user
        
    # Then try API key authentication
    if api_key_user:
        return api_key_user
        
    # No valid authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user
    
    Ensures the authenticated user is active.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current superuser
    
    Ensures the authenticated user is a superuser.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


class PermissionChecker:
    """
    Permission checker dependency
    
    Usage:
        @router.get("/protected")
        async def protected_endpoint(
            user: User = Depends(PermissionChecker(PermissionType.EXECUTE_TRADES))
        ):
            ...
    """
    
    def __init__(self, required_permission: str):
        self.required_permission = required_permission
        
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """
        Check if user has required permission
        
        Returns user if permission check passes, raises 403 otherwise.
        """
        if not current_user.has_permission(self.required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{self.required_permission}' required"
            )
        return current_user


class RoleChecker:
    """
    Role checker dependency
    
    Usage:
        @router.get("/admin")
        async def admin_endpoint(
            user: User = Depends(RoleChecker("admin"))
        ):
            ...
    """
    
    def __init__(self, required_role: str):
        self.required_role = required_role
        
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """
        Check if user has required role
        
        Returns user if role check passes, raises 403 otherwise.
        """
        if not current_user.has_role(self.required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{self.required_role}' required"
            )
        return current_user


class MultiPermissionChecker:
    """
    Multiple permission checker dependency
    
    Checks if user has ANY of the required permissions.
    
    Usage:
        @router.get("/trading")
        async def trading_endpoint(
            user: User = Depends(
                MultiPermissionChecker([
                    PermissionType.EXECUTE_TRADES,
                    PermissionType.ADMIN_ACCESS
                ])
            )
        ):
            ...
    """
    
    def __init__(self, required_permissions: list):
        self.required_permissions = required_permissions
        
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """
        Check if user has any of the required permissions
        
        Returns user if any permission check passes, raises 403 otherwise.
        """
        for permission in self.required_permissions:
            if current_user.has_permission(permission):
                return current_user
                
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"One of these permissions required: {', '.join(self.required_permissions)}"
        )


# Convenience dependency instances for common permissions
require_read_portfolio = PermissionChecker("read_portfolio")
require_write_portfolio = PermissionChecker("write_portfolio")
require_execute_trades = PermissionChecker("execute_trades")
require_manage_strategies = PermissionChecker("manage_strategies")
require_admin_access = PermissionChecker("admin_access")
require_manage_users = PermissionChecker("manage_users")

# Convenience dependency instances for common roles
require_admin_role = RoleChecker("admin")
require_trader_role = RoleChecker("trader")
require_viewer_role = RoleChecker("viewer")