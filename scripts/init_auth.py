#!/usr/bin/env python3
"""
Initialize Authentication System

This script:
1. Runs authentication database migrations
2. Creates default roles and permissions
3. Sets up initial admin user
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import init_db_pool, close_db_pool
from src.core.observability import get_logger
from src.auth.service import AuthService
from src.auth.repository import UserRepository, RoleRepository, APIKeyRepository
from src.auth.models import UserCreate

logger = get_logger(__name__)


async def run_auth_migration(pool):
    """Run authentication schema migration"""
    migration_file = Path(__file__).parent.parent / "migrations" / "002_auth_schema.sql"
    
    if not migration_file.exists():
        logger.error(f"Migration file not found: {migration_file}")
        return False
        
    logger.info(f"Running migration: {migration_file}")
    
    async with pool.acquire() as conn:
        try:
            # Read migration SQL
            with open(migration_file, 'r') as f:
                sql = f.read()
                
            # Execute migration
            await conn.execute(sql)
            logger.info("Authentication schema migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False


async def verify_setup(pool):
    """Verify authentication setup"""
    async with pool.acquire() as conn:
        # Check tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'quantum' 
            AND table_name IN ('users', 'roles', 'permissions', 'api_keys')
            ORDER BY table_name
        """)
        
        logger.info(f"Found {len(tables)} authentication tables")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
            
        # Check permissions
        perms = await conn.fetch("SELECT name FROM quantum.permissions ORDER BY name")
        logger.info(f"Found {len(perms)} permissions:")
        for perm in perms:
            logger.info(f"  - {perm['name']}")
            
        # Check roles
        roles = await conn.fetch("SELECT name, description FROM quantum.roles ORDER BY name")
        logger.info(f"Found {len(roles)} roles:")
        for role in roles:
            logger.info(f"  - {role['name']}: {role['description']}")
            
        # Check admin user
        admin = await conn.fetchrow("SELECT username, email FROM quantum.users WHERE username = 'admin'")
        if admin:
            logger.info(f"Admin user exists: {admin['username']} ({admin['email']})")
        else:
            logger.warning("Admin user not found!")
            
        return len(tables) == 4


async def create_test_users(auth_service):
    """Create some test users for development"""
    test_users = [
        {
            "data": UserCreate(
                email="trader@example.com",
                username="trader1",
                password="Trader123!",
                full_name="Test Trader"
            ),
            "role": "trader"
        },
        {
            "data": UserCreate(
                email="viewer@example.com",
                username="viewer1",
                password="Viewer123!",
                full_name="Test Viewer"
            ),
            "role": "viewer"
        },
        {
            "data": UserCreate(
                email="bot@example.com",
                username="tradingbot",
                password="Bot123!@#",
                full_name="Trading Bot"
            ),
            "role": "bot"
        }
    ]
    
    for user_info in test_users:
        try:
            user = await auth_service.register(
                user_info["data"],
                role_name=user_info["role"]
            )
            logger.info(f"Created test user: {user.username} with role: {user_info['role']}")
        except ValueError as e:
            logger.info(f"User already exists: {user_info['data'].username}")
        except Exception as e:
            logger.error(f"Failed to create user {user_info['data'].username}: {str(e)}")


async def main():
    """Main initialization function"""
    logger.info("Starting authentication system initialization...")
    
    # Initialize database pool
    pool = await init_db_pool()
    
    try:
        # Run migration
        if not await run_auth_migration(pool):
            logger.error("Migration failed, aborting initialization")
            return
            
        # Initialize services
        user_repo = UserRepository(pool)
        role_repo = RoleRepository(pool)
        api_key_repo = APIKeyRepository(pool)
        
        auth_service = AuthService(
            user_repo=user_repo,
            role_repo=role_repo,
            api_key_repo=api_key_repo
        )
        
        # Initialize default roles (idempotent)
        await auth_service.initialize_default_roles()
        
        # Verify setup
        if await verify_setup(pool):
            logger.info("Authentication system initialized successfully!")
            
            # Create test users in development
            if os.getenv("ENVIRONMENT", "development") == "development":
                logger.info("Creating test users for development...")
                await create_test_users(auth_service)
        else:
            logger.error("Authentication system verification failed!")
            
    finally:
        await close_db_pool()
        
    logger.info("Initialization complete!")
    
    # Print important information
    print("\n" + "="*60)
    print("AUTHENTICATION SYSTEM INITIALIZED")
    print("="*60)
    print("\nDefault Admin Credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print("\n⚠️  IMPORTANT: Change the admin password immediately!")
    print("\nTest Users (Development Only):")
    print("  trader1 / Trader123!  (trader role)")
    print("  viewer1 / Viewer123!  (viewer role)")
    print("  tradingbot / Bot123!@# (bot role)")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())