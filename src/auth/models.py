"""
Authentication Models

Defines user, role, and permission models for the authentication system.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
from enum import Enum

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PermissionType(str, Enum):
    """Permission types for the system"""
    # Portfolio permissions
    READ_PORTFOLIO = "read_portfolio"
    WRITE_PORTFOLIO = "write_portfolio"
    
    # Trading permissions
    EXECUTE_TRADES = "execute_trades"
    MANAGE_STRATEGIES = "manage_strategies"
    
    # System permissions
    ADMIN_ACCESS = "admin_access"
    MANAGE_USERS = "manage_users"
    VIEW_ALL_PORTFOLIOS = "view_all_portfolios"
    
    # API permissions
    API_FULL_ACCESS = "api_full_access"
    API_READ_ONLY = "api_read_only"


class Permission(BaseModel):
    """Permission model"""
    id: UUID = Field(default_factory=uuid4)
    name: PermissionType
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "execute_trades",
                "description": "Permission to execute trades"
            }
        }


class Role(BaseModel):
    """Role model"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    permissions: List[Permission] = []
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "trader",
                "description": "Can execute trades and manage strategies",
                "permissions": ["execute_trades", "manage_strategies"]
            }
        }


class UserRole(BaseModel):
    """User-Role mapping"""
    user_id: UUID
    role_id: UUID
    assigned_at: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel):
    """User model"""
    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    username: str
    password_hash: Optional[str] = Field(None, exclude=True)  # Exclude from API responses
    is_active: bool = True
    is_superuser: bool = False
    roles: List[Role] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Additional fields
    full_name: Optional[str] = None
    preferences: Dict[str, Any] = {}
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum() or '_' in v or '-' in v, 'Username must be alphanumeric'
        assert len(v) >= 3, 'Username must be at least 3 characters'
        return v
    
    def set_password(self, password: str):
        """Hash and set user password"""
        self.password_hash = pwd_context.hash(password)
        
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        if not self.password_hash:
            return False
        return pwd_context.verify(password, self.password_hash)
    
    def has_permission(self, permission: PermissionType) -> bool:
        """Check if user has specific permission"""
        if self.is_superuser:
            return True
            
        for role in self.roles:
            for perm in role.permissions:
                if perm.name == permission:
                    return True
        return False
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return any(role.name == role_name for role in self.roles)
    
    def get_permissions(self) -> List[PermissionType]:
        """Get all user permissions"""
        if self.is_superuser:
            return list(PermissionType)
            
        permissions = set()
        for role in self.roles:
            for perm in role.permissions:
                permissions.add(perm.name)
        return list(permissions)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "trader123",
                "full_name": "John Doe",
                "is_active": True
            }
        }


class UserCreate(BaseModel):
    """User creation schema"""
    email: EmailStr
    username: str
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserUpdate(BaseModel):
    """User update schema"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None


class UserLogin(BaseModel):
    """User login schema"""
    username: str
    password: str


class Token(BaseModel):
    """JWT Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    email: str
    permissions: List[str] = []
    exp: Optional[datetime] = None


class APIKey(BaseModel):
    """API Key model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    key_hash: str
    name: str
    permissions: List[PermissionType] = []
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    
    def has_permission(self, permission: PermissionType) -> bool:
        """Check if API key has specific permission"""
        return permission in self.permissions
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        return self.is_active and not self.is_expired()


# Default roles and permissions
DEFAULT_PERMISSIONS = {
    PermissionType.READ_PORTFOLIO: "View portfolio data",
    PermissionType.WRITE_PORTFOLIO: "Modify portfolio data",
    PermissionType.EXECUTE_TRADES: "Execute trading orders",
    PermissionType.MANAGE_STRATEGIES: "Create and manage trading strategies",
    PermissionType.ADMIN_ACCESS: "Full administrative access",
    PermissionType.MANAGE_USERS: "Manage user accounts",
    PermissionType.VIEW_ALL_PORTFOLIOS: "View all user portfolios",
    PermissionType.API_FULL_ACCESS: "Full API access",
    PermissionType.API_READ_ONLY: "Read-only API access"
}

DEFAULT_ROLES = {
    "admin": {
        "description": "Full system access",
        "permissions": list(PermissionType)
    },
    "trader": {
        "description": "Can execute trades and manage strategies",
        "permissions": [
            PermissionType.READ_PORTFOLIO,
            PermissionType.WRITE_PORTFOLIO,
            PermissionType.EXECUTE_TRADES,
            PermissionType.MANAGE_STRATEGIES,
            PermissionType.API_FULL_ACCESS
        ]
    },
    "viewer": {
        "description": "Read-only access",
        "permissions": [
            PermissionType.READ_PORTFOLIO,
            PermissionType.API_READ_ONLY
        ]
    },
    "bot": {
        "description": "Automated trading bot access",
        "permissions": [
            PermissionType.READ_PORTFOLIO,
            PermissionType.EXECUTE_TRADES,
            PermissionType.API_FULL_ACCESS
        ]
    }
}