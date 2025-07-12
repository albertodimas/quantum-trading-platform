"""
Authentication Service

High-level authentication operations including login, logout, registration,
and token management.
"""

from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import secrets
import hashlib

from ..core.architecture import injectable
from ..core.observability import get_logger
from .models import (
    User, UserCreate, UserLogin, Token,
    Role, Permission, PermissionType,
    APIKey, DEFAULT_ROLES
)
from .repository import UserRepository, RoleRepository, APIKeyRepository
from .jwt_handler import JWTHandler, get_jwt_handler

logger = get_logger(__name__)


@injectable
class AuthService:
    """Service for authentication and authorization operations"""
    
    def __init__(
        self,
        user_repo: UserRepository,
        role_repo: RoleRepository,
        api_key_repo: APIKeyRepository,
        jwt_handler: Optional[JWTHandler] = None
    ):
        self.user_repo = user_repo
        self.role_repo = role_repo
        self.api_key_repo = api_key_repo
        self.jwt_handler = jwt_handler or get_jwt_handler()
        
    async def register(
        self,
        user_data: UserCreate,
        role_name: str = "viewer"
    ) -> User:
        """
        Register new user with default role
        
        Args:
            user_data: User registration data
            role_name: Default role to assign (default: "viewer")
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If username/email already exists
        """
        # Check if username exists
        existing_user = await self.user_repo.get_by_username(user_data.username)
        if existing_user:
            raise ValueError(f"Username '{user_data.username}' already exists")
            
        # Check if email exists
        existing_email = await self.user_repo.get_by_email(user_data.email)
        if existing_email:
            raise ValueError(f"Email '{user_data.email}' already registered")
            
        # Create user
        user = await self.user_repo.create_user(user_data)
        
        # Assign default role
        default_role = await self.role_repo.get_by_name(role_name)
        if default_role:
            await self.user_repo.assign_role(user.id, default_role.id)
            user = await self.user_repo.get_with_roles(user.id)
            
        logger.info(f"User registered: {user.username}")
        return user
        
    async def login(
        self,
        credentials: UserLogin
    ) -> Tuple[User, Token]:
        """
        Authenticate user and generate tokens
        
        Args:
            credentials: Login credentials
            
        Returns:
            Tuple of (User, Token)
            
        Raises:
            ValueError: If credentials are invalid
        """
        # Get user by username
        user = await self.user_repo.get_by_username(credentials.username)
        if not user:
            logger.warning(f"Login attempt with invalid username: {credentials.username}")
            raise ValueError("Invalid username or password")
            
        # Verify password
        if not user.verify_password(credentials.password):
            logger.warning(f"Login attempt with invalid password for user: {credentials.username}")
            raise ValueError("Invalid username or password")
            
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {credentials.username}")
            raise ValueError("Account is not active")
            
        # Update last login
        await self.user_repo.update_last_login(user.id)
        
        # Create tokens
        access_token = self.jwt_handler.create_access_token(user)
        refresh_token = self.jwt_handler.create_refresh_token(user)
        
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.jwt_handler.access_token_expire_minutes * 60
        )
        
        logger.info(f"User logged in: {user.username}")
        return user, token
        
    async def refresh_token(
        self,
        refresh_token: str
    ) -> Token:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New Token object
            
        Raises:
            ValueError: If refresh token is invalid
        """
        # Decode refresh token
        token_data = self.jwt_handler.verify_token(refresh_token, token_type="refresh")
        if not token_data:
            raise ValueError("Invalid refresh token")
            
        # Get user
        user = await self.user_repo.get(UUID(token_data.user_id))
        if not user:
            raise ValueError("User not found")
            
        # Check if user is active
        if not user.is_active:
            raise ValueError("Account is not active")
            
        # Create new access token
        access_token = self.jwt_handler.create_access_token(user)
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.jwt_handler.access_token_expire_minutes * 60
        )
        
    async def logout(self, user_id: UUID) -> bool:
        """
        Logout user (placeholder for session management)
        
        In a stateless JWT system, logout is typically handled client-side
        by removing the token. This method can be extended to:
        - Blacklist tokens
        - Clear server-side sessions
        - Log security events
        
        Args:
            user_id: User ID to logout
            
        Returns:
            Success status
        """
        logger.info(f"User logged out: {user_id}")
        # TODO: Implement token blacklisting if needed
        return True
        
    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password
        
        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            Success status
            
        Raises:
            ValueError: If current password is incorrect
        """
        user = await self.user_repo.get_with_roles(user_id)
        if not user:
            raise ValueError("User not found")
            
        # Verify current password
        if not user.verify_password(current_password):
            raise ValueError("Current password is incorrect")
            
        # Set new password
        user.set_password(new_password)
        
        # Update in database
        await self.user_repo.update(
            user_id,
            {"password_hash": user.password_hash}
        )
        
        logger.info(f"Password changed for user: {user_id}")
        return True
        
    async def reset_password(
        self,
        email: str,
        reset_token: str,
        new_password: str
    ) -> bool:
        """
        Reset user password with reset token
        
        Note: This is a simplified implementation. In production,
        you would:
        - Generate and send reset tokens via email
        - Store tokens with expiration in database
        - Validate token before allowing reset
        
        Args:
            email: User email
            reset_token: Password reset token
            new_password: New password
            
        Returns:
            Success status
        """
        # TODO: Implement proper reset token validation
        user = await self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("User not found")
            
        # Set new password
        user.set_password(new_password)
        
        # Update in database
        await self.user_repo.update(
            user.id,
            {"password_hash": user.password_hash}
        )
        
        logger.info(f"Password reset for user: {email}")
        return True
        
    async def create_api_key(
        self,
        user_id: UUID,
        name: str,
        permissions: List[PermissionType],
        expires_in_days: Optional[int] = None
    ) -> Tuple[APIKey, str]:
        """
        Create API key for user
        
        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions for the key
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple of (APIKey object, raw key string)
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
        # Create API key
        api_key = await self.api_key_repo.create_api_key(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            expires_at=expires_at
        )
        
        logger.info(f"API key created: {name} for user {user_id}")
        
        # Return both the object and raw key (raw key shown only once)
        return api_key, raw_key
        
    async def validate_api_key(self, raw_key: str) -> Optional[Tuple[APIKey, User]]:
        """
        Validate API key and return associated user
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            Tuple of (APIKey, User) if valid, None otherwise
        """
        # Hash the key
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Get API key
        api_key = await self.api_key_repo.get_by_key_hash(key_hash)
        if not api_key or not api_key.is_valid():
            return None
            
        # Get associated user
        user = await self.user_repo.get_with_roles(api_key.user_id)
        if not user or not user.is_active:
            return None
            
        return api_key, user
        
    async def list_user_api_keys(
        self,
        user_id: UUID,
        include_inactive: bool = False
    ) -> List[APIKey]:
        """
        List user's API keys
        
        Args:
            user_id: User ID
            include_inactive: Include revoked keys
            
        Returns:
            List of API keys
        """
        return await self.api_key_repo.list_user_api_keys(
            user_id,
            include_inactive
        )
        
    async def revoke_api_key(
        self,
        user_id: UUID,
        key_id: UUID
    ) -> bool:
        """
        Revoke API key
        
        Args:
            user_id: User ID (for ownership verification)
            key_id: API key ID
            
        Returns:
            Success status
        """
        return await self.api_key_repo.revoke_api_key(key_id, user_id)
        
    async def assign_role_to_user(
        self,
        user_id: UUID,
        role_name: str
    ) -> bool:
        """
        Assign role to user
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            Success status
        """
        role = await self.role_repo.get_by_name(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found")
            
        return await self.user_repo.assign_role(user_id, role.id)
        
    async def remove_role_from_user(
        self,
        user_id: UUID,
        role_name: str
    ) -> bool:
        """
        Remove role from user
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            Success status
        """
        role = await self.role_repo.get_by_name(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found")
            
        return await self.user_repo.remove_role(user_id, role.id)
        
    async def get_user_permissions(self, user_id: UUID) -> List[PermissionType]:
        """
        Get all permissions for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of permissions
        """
        user = await self.user_repo.get_with_roles(user_id)
        if not user:
            return []
            
        return user.get_permissions()
        
    async def check_permission(
        self,
        user_id: UUID,
        permission: PermissionType
    ) -> bool:
        """
        Check if user has specific permission
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user = await self.user_repo.get_with_roles(user_id)
        if not user:
            return False
            
        return user.has_permission(permission)
        
    async def initialize_default_roles(self) -> None:
        """
        Initialize default roles and permissions
        
        This should be called during system initialization
        to ensure default roles exist.
        """
        for role_name, role_config in DEFAULT_ROLES.items():
            existing_role = await self.role_repo.get_by_name(role_name)
            if not existing_role:
                await self.role_repo.create_role(
                    name=role_name,
                    description=role_config["description"],
                    permissions=role_config["permissions"]
                )
                logger.info(f"Created default role: {role_name}")
        
    async def get_current_user(self, token: str) -> Optional[User]:
        """
        Get current user from access token
        
        Args:
            token: JWT access token
            
        Returns:
            User object if valid, None otherwise
        """
        token_data = self.jwt_handler.verify_token(token, token_type="access")
        if not token_data:
            return None
            
        user = await self.user_repo.get_with_roles(UUID(token_data.user_id))
        if not user or not user.is_active:
            return None
            
        return user