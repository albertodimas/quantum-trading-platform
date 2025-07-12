"""
Authentication API Endpoints

FastAPI router for authentication endpoints including login, logout,
registration, and token management.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from uuid import UUID

from ..core.observability import get_logger
from .models import (
    User, UserCreate, UserLogin, Token,
    PermissionType, APIKey
)
from .service import AuthService
from .dependencies import get_current_user, get_auth_service

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"]
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register new user
    
    Creates a new user account with default 'viewer' role.
    """
    try:
        user = await auth_service.register(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login user
    
    Authenticate user with username/password and return JWT tokens.
    """
    try:
        credentials = UserLogin(
            username=form_data.username,
            password=form_data.password
        )
        user, token = await auth_service.login(credentials)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token
    
    Use refresh token to get new access token.
    """
    try:
        token = await auth_service.refresh_token(refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout user
    
    Invalidate user session (client should discard tokens).
    """
    try:
        await auth_service.logout(current_user.id)
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user info
    
    Returns information about the authenticated user.
    """
    return current_user


@router.put("/me", response_model=User)
async def update_current_user(
    update_data: dict,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update current user info
    
    Update user profile information.
    """
    # TODO: Use UserUpdate model instead of dict
    # Filter allowed fields
    allowed_fields = {"full_name", "preferences"}
    filtered_data = {k: v for k, v in update_data.items() if k in allowed_fields}
    
    # Update user
    updated_user = await auth_service.user_repo.update_user(
        current_user.id,
        filtered_data
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return updated_user


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change user password
    
    Change password for the authenticated user.
    """
    try:
        success = await auth_service.change_password(
            current_user.id,
            current_password,
            new_password
        )
        if success:
            return {"message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.get("/permissions", response_model=List[str])
async def get_my_permissions(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user permissions
    
    Returns list of permissions for the authenticated user.
    """
    return [perm.value for perm in current_user.get_permissions()]


# API Key Management Endpoints

@router.post("/api-keys", response_model=dict)
async def create_api_key(
    name: str,
    permissions: List[PermissionType],
    expires_in_days: int = None,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Create API key
    
    Create a new API key for the authenticated user.
    Returns the API key only once - store it securely!
    """
    try:
        api_key, raw_key = await auth_service.create_api_key(
            current_user.id,
            name,
            permissions,
            expires_in_days
        )
        
        return {
            "id": str(api_key.id),
            "name": api_key.name,
            "key": raw_key,  # Only returned once!
            "permissions": [p.value for p in api_key.permissions],
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "message": "Store this key securely - it won't be shown again!"
        }
    except Exception as e:
        logger.error(f"API key creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key creation failed"
        )


@router.get("/api-keys", response_model=List[dict])
async def list_api_keys(
    include_inactive: bool = False,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    List user's API keys
    
    Returns list of API keys for the authenticated user.
    """
    api_keys = await auth_service.list_user_api_keys(
        current_user.id,
        include_inactive
    )
    
    return [
        {
            "id": str(key.id),
            "name": key.name,
            "permissions": [p.value for p in key.permissions],
            "is_active": key.is_active,
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "created_at": key.created_at.isoformat(),
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None
        }
        for key in api_keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Revoke API key
    
    Revoke an API key owned by the authenticated user.
    """
    try:
        success = await auth_service.revoke_api_key(
            current_user.id,
            key_id
        )
        if success:
            return {"message": "API key revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
    except Exception as e:
        logger.error(f"API key revocation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key revocation failed"
        )


# Admin Endpoints (require admin permissions)

@router.get("/users", response_model=List[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    include_inactive: bool = False,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    List all users (admin only)
    
    Returns paginated list of users.
    """
    if not current_user.has_permission(PermissionType.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    users = await auth_service.user_repo.list_users(
        skip=skip,
        limit=limit,
        include_inactive=include_inactive
    )
    
    return users


@router.post("/users/{user_id}/roles/{role_name}")
async def assign_role(
    user_id: UUID,
    role_name: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Assign role to user (admin only)
    
    Assign a role to a specific user.
    """
    if not current_user.has_permission(PermissionType.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        success = await auth_service.assign_role_to_user(user_id, role_name)
        if success:
            return {"message": f"Role '{role_name}' assigned successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Role assignment error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role assignment failed"
        )


@router.delete("/users/{user_id}/roles/{role_name}")
async def remove_role(
    user_id: UUID,
    role_name: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Remove role from user (admin only)
    
    Remove a role from a specific user.
    """
    if not current_user.has_permission(PermissionType.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        success = await auth_service.remove_role_from_user(user_id, role_name)
        if success:
            return {"message": f"Role '{role_name}' removed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Role removal error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role removal failed"
        )