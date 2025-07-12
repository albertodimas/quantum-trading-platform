"""
Authentication and Authorization Module

This module provides:
- User authentication with JWT tokens
- Role-based access control (RBAC)
- API key management
- Permission system
"""

from .models import (
    User, Role, Permission, UserRole,
    UserCreate, UserUpdate, UserLogin,
    Token, TokenData, APIKey,
    PermissionType
)
from .jwt_handler import (
    JWTHandler, get_jwt_handler,
    create_access_token, create_refresh_token,
    decode_access_token, decode_refresh_token
)
from .service import AuthService
from .repository import UserRepository, RoleRepository, APIKeyRepository
from .dependencies import (
    get_current_user, get_current_active_user,
    get_current_superuser, get_auth_service,
    PermissionChecker, RoleChecker
)

__all__ = [
    # Models
    'User',
    'Role', 
    'Permission',
    'UserRole',
    'UserCreate',
    'UserUpdate',
    'UserLogin',
    'Token',
    'TokenData',
    'APIKey',
    'PermissionType',
    
    # JWT Handler
    'JWTHandler',
    'get_jwt_handler',
    'create_access_token',
    'create_refresh_token',
    'decode_access_token',
    'decode_refresh_token',
    
    # Service & Repositories
    'AuthService',
    'UserRepository',
    'RoleRepository',
    'APIKeyRepository',
    
    # Dependencies
    'get_current_user',
    'get_current_active_user',
    'get_current_superuser',
    'get_auth_service',
    'PermissionChecker',
    'RoleChecker'
]