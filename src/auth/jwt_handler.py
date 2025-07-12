"""
JWT Token Handler

Manages JWT token creation, validation, and decoding for authentication.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from uuid import UUID

from ..core.observability import get_logger
from .models import TokenData, User

logger = get_logger(__name__)


class JWTHandler:
    """Handle JWT token operations"""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "your-secret-key-here")
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        if self.secret_key == "your-secret-key-here":
            logger.warning("Using default secret key - CHANGE THIS IN PRODUCTION!")
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token for user
        
        Args:
            user: User object
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # Create token payload
        token_data = {
            "sub": str(user.id),  # Subject (user ID)
            "username": user.username,
            "email": user.email,
            "permissions": [perm.value for perm in user.get_permissions()],
            "is_superuser": user.is_superuser,
            "exp": expire,
            "iat": datetime.utcnow(),  # Issued at
            "type": "access"
        }
        
        # Encode token
        encoded_jwt = jwt.encode(
            token_data,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        logger.info(f"Access token created for user: {user.username}")
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token for user
        
        Args:
            user: User object
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT refresh token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        # Create minimal payload for refresh token
        token_data = {
            "sub": str(user.id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        # Encode token
        encoded_jwt = jwt.encode(
            token_data,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        logger.info(f"Refresh token created for user: {user.username}")
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
            
        except JWTError as e:
            logger.error(f"Invalid token: {str(e)}")
            return None
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """
        Verify token and return token data
        
        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            TokenData object or None if invalid
        """
        payload = self.decode_token(token)
        if not payload:
            return None
        
        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
            return None
        
        # Create TokenData object
        try:
            token_data = TokenData(
                user_id=payload.get("sub"),
                username=payload.get("username", ""),
                email=payload.get("email", ""),
                permissions=payload.get("permissions", []),
                exp=datetime.fromtimestamp(payload.get("exp"))
            )
            return token_data
            
        except Exception as e:
            logger.error(f"Error creating TokenData: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str, user: User) -> Optional[str]:
        """
        Create new access token from refresh token
        
        Args:
            refresh_token: Valid refresh token
            user: User object (should be fetched based on refresh token)
            
        Returns:
            New access token or None if refresh token invalid
        """
        # Verify refresh token
        token_data = self.verify_token(refresh_token, token_type="refresh")
        if not token_data:
            return None
        
        # Verify user ID matches
        if str(user.id) != token_data.user_id:
            logger.warning("User ID mismatch in refresh token")
            return None
        
        # Create new access token
        return self.create_access_token(user)
    
    def create_api_key_token(
        self,
        api_key_id: UUID,
        user_id: UUID,
        permissions: list,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT token for API key authentication
        
        Args:
            api_key_id: API key ID
            user_id: User ID who owns the API key
            permissions: List of permissions for the API key
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=365)  # API keys have longer expiry
        
        token_data = {
            "sub": str(api_key_id),
            "user_id": str(user_id),
            "permissions": permissions,
            "type": "api_key",
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(
            token_data,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt


# Global JWT handler instance
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get global JWT handler instance"""
    global _jwt_handler
    
    if _jwt_handler is None:
        _jwt_handler = JWTHandler(
            secret_key=os.getenv("SECRET_KEY"),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(
                os.getenv("JWT_EXPIRATION_MINUTES", "30")
            )
        )
    
    return _jwt_handler


# Convenience functions
def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token for user"""
    return get_jwt_handler().create_access_token(user, expires_delta)


def create_refresh_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create refresh token for user"""
    return get_jwt_handler().create_refresh_token(user, expires_delta)


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and verify access token"""
    return get_jwt_handler().verify_token(token, token_type="access")


def decode_refresh_token(token: str) -> Optional[TokenData]:
    """Decode and verify refresh token"""
    return get_jwt_handler().verify_token(token, token_type="refresh")