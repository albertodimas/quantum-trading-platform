"""
Authentication Middleware

FastAPI middleware for authentication, rate limiting, and security headers.
"""

import time
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.observability import get_logger
from ..core.architecture import RateLimiter, TokenBucketStrategy
from .jwt_handler import get_jwt_handler
from .models import User

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware that validates JWT tokens
    
    This middleware:
    - Validates JWT tokens on protected routes
    - Adds user information to request state
    - Handles authentication errors
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list] = None,
        optional_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.jwt_handler = get_jwt_handler()
        
        # Paths that don't require authentication
        self.exclude_paths = exclude_paths or [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        # Paths where authentication is optional
        self.optional_paths = optional_paths or []
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and validate authentication
        """
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
            
        # Check if authentication is optional for this path
        is_optional = any(
            request.url.path.startswith(path) 
            for path in self.optional_paths
        )
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header and not is_optional:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header missing"},
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        if auth_header:
            try:
                # Validate Bearer token format
                scheme, token = auth_header.split()
                if scheme.lower() != "bearer":
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid authentication scheme"},
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                    
                # Decode and validate token
                token_data = self.jwt_handler.verify_token(token, token_type="access")
                if not token_data:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or expired token"},
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                    
                # Add user info to request state
                request.state.user_id = token_data.user_id
                request.state.username = token_data.username
                request.state.permissions = token_data.permissions
                
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid authorization header format"},
                    headers={"WWW-Authenticate": "Bearer"}
                )
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Authentication failed"}
                )
        elif not is_optional:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Process request
        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    
    Implements per-user and per-IP rate limiting.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        super().__init__(app)
        
        # Create rate limiters
        self.minute_limiter = RateLimiter(
            strategy=TokenBucketStrategy(
                capacity=requests_per_minute,
                refill_rate=requests_per_minute / 60,  # per second
                refill_amount=1
            )
        )
        
        self.hour_limiter = RateLimiter(
            strategy=TokenBucketStrategy(
                capacity=requests_per_hour,
                refill_rate=requests_per_hour / 3600,  # per second
                refill_amount=1
            )
        )
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits before processing request
        """
        # Get client identifier (user ID or IP)
        client_id = None
        if hasattr(request.state, "user_id"):
            client_id = f"user:{request.state.user_id}"
        else:
            # Use IP address as fallback
            client_id = f"ip:{request.client.host}"
            
        # Check minute rate limit
        if not await self.minute_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
            
        # Check hour rate limit
        if not await self.hour_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Hourly rate limit exceeded",
                    "retry_after": 3600
                },
                headers={"Retry-After": "3600"}
            )
            
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(
            self.minute_limiter.strategy.capacity
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(
            self.hour_limiter.strategy.capacity
        )
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware
    
    Adds security headers to all responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware with authentication support
    
    Handles CORS headers for cross-origin requests.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: Optional[list] = None,
        allowed_methods: Optional[list] = None,
        allowed_headers: Optional[list] = None,
        allow_credentials: bool = True
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or [
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"
        ]
        self.allowed_headers = allowed_headers or [
            "Authorization", "Content-Type", "X-API-Key"
        ]
        self.allow_credentials = allow_credentials
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle CORS headers
        """
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)
            
        # Get origin
        origin = request.headers.get("origin")
        
        # Set CORS headers
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.allowed_methods
            )
            response.headers["Access-Control-Allow-Headers"] = ", ".join(
                self.allowed_headers
            )
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
                
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware
    
    Logs all requests with timing information.
    """
    
    def __init__(self, app: ASGIApp, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response information
        """
        start_time = time.time()
        
        # Log request
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "client": request.client.host if request.client else None
        }
        
        if hasattr(request.state, "user_id"):
            log_data["user_id"] = request.state.user_id
            
        logger.info(f"Request started: {log_data}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {duration:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        return response


def setup_middleware(app: ASGIApp) -> None:
    """
    Setup all middleware for the application
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is first executed)
    
    # Request logging (outermost)
    app.add_middleware(RequestLoggingMiddleware)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allowed_origins=["http://localhost:3000", "https://app.quantumtrading.io"],
        allow_credentials=True
    )
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
    
    # Authentication (innermost)
    app.add_middleware(AuthenticationMiddleware)