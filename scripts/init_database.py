#!/usr/bin/env python3
"""
Initialize Quantum Trading Platform Database

This script:
1. Creates the database if it doesn't exist
2. Runs all migrations in order
3. Validates the schema
4. Seeds initial data if needed
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handle database initialization and migrations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
        
    async def create_database(self):
        """Create database if it doesn't exist"""
        # Parse database URL
        parts = self.database_url.replace("postgresql://", "").split("/")
        db_name = parts[-1].split("?")[0]
        base_url = "postgresql://" + parts[0]
        
        try:
            # Connect to postgres database
            conn = await asyncpg.connect(base_url + "/postgres")
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)",
                db_name
            )
            
            if not exists:
                logger.info(f"Creating database: {db_name}")
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database {db_name} created successfully")
            else:
                logger.info(f"Database {db_name} already exists")
                
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
            
    async def get_applied_migrations(self, conn: asyncpg.Connection) -> set:
        """Get list of already applied migrations"""
        # Check if migrations table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'quantum' 
                AND table_name = 'schema_migrations'
            );
        """)
        
        if not table_exists:
            # Create migrations table
            await conn.execute("""
                CREATE SCHEMA IF NOT EXISTS quantum;
                CREATE TABLE IF NOT EXISTS quantum.schema_migrations (
                    filename VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            return set()
            
        # Get applied migrations
        rows = await conn.fetch(
            "SELECT filename FROM quantum.schema_migrations ORDER BY applied_at"
        )
        return {row['filename'] for row in rows}
        
    async def apply_migration(self, conn: asyncpg.Connection, migration_file: Path):
        """Apply a single migration file"""
        logger.info(f"Applying migration: {migration_file.name}")
        
        try:
            # Read migration SQL
            with open(migration_file, 'r') as f:
                sql = f.read()
                
            # Execute migration
            await conn.execute(sql)
            
            # Record migration
            await conn.execute("""
                INSERT INTO quantum.schema_migrations (filename) 
                VALUES ($1)
            """, migration_file.name)
            
            logger.info(f"Migration {migration_file.name} applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying migration {migration_file.name}: {e}")
            raise
            
    async def run_migrations(self):
        """Run all pending migrations"""
        conn = await asyncpg.connect(self.database_url)
        
        try:
            # Get applied migrations
            applied = await self.get_applied_migrations(conn)
            
            # Get migration files
            migration_files = sorted(
                [f for f in self.migrations_dir.glob("*.sql") if f.is_file()]
            )
            
            # Apply pending migrations
            pending_count = 0
            for migration_file in migration_files:
                if migration_file.name not in applied:
                    await self.apply_migration(conn, migration_file)
                    pending_count += 1
                    
            if pending_count == 0:
                logger.info("No pending migrations")
            else:
                logger.info(f"Applied {pending_count} migrations")
                
        finally:
            await conn.close()
            
    async def validate_schema(self):
        """Validate that all expected tables exist"""
        conn = await asyncpg.connect(self.database_url)
        
        try:
            # Expected tables
            expected_tables = [
                'users', 'roles', 'user_roles', 'api_keys',
                'exchanges', 'user_exchange_credentials', 'symbols',
                'strategies', 'strategy_instances',
                'orders', 'trades', 'positions', 'position_history',
                'portfolio_snapshots', 'market_ticks', 'market_candles',
                'system_events', 'trading_signals'
            ]
            
            # Check each table
            missing_tables = []
            for table in expected_tables:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'quantum' 
                        AND table_name = $1
                    );
                """, table)
                
                if not exists:
                    missing_tables.append(table)
                    
            if missing_tables:
                logger.error(f"Missing tables: {', '.join(missing_tables)}")
                return False
                
            logger.info("Schema validation passed - all tables present")
            
            # Check TimescaleDB hypertables
            hypertables = await conn.fetch("""
                SELECT hypertable_name 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_schema = 'quantum'
            """)
            
            hypertable_names = {row['hypertable_name'] for row in hypertables}
            expected_hypertables = {
                'orders', 'trades', 'position_history', 'portfolio_snapshots',
                'market_ticks', 'market_candles', 'system_events', 'trading_signals'
            }
            
            missing_hypertables = expected_hypertables - hypertable_names
            if missing_hypertables:
                logger.warning(
                    f"Missing TimescaleDB hypertables: {', '.join(missing_hypertables)}"
                )
                logger.warning("TimescaleDB extension might not be installed")
            else:
                logger.info("All TimescaleDB hypertables configured correctly")
                
            return True
            
        finally:
            await conn.close()
            
    async def seed_test_data(self):
        """Seed initial test data if in development mode"""
        if settings.ENVIRONMENT != "development":
            logger.info("Skipping test data seeding (not in development mode)")
            return
            
        conn = await asyncpg.connect(self.database_url)
        
        try:
            # Check if we already have test data
            user_count = await conn.fetchval(
                "SELECT COUNT(*) FROM quantum.users"
            )
            
            if user_count > 0:
                logger.info("Test data already exists, skipping seeding")
                return
                
            logger.info("Seeding test data...")
            
            # Create test user
            await conn.execute("""
                INSERT INTO quantum.users (email, username, password_hash)
                VALUES ($1, $2, $3)
            """, "test@quantum.trading", "testuser", 
                # This is a bcrypt hash of "testpassword"
                "$2b$12$YIhH5x.SNkcGKZW8LAweOOqJTXCqp9fJjDv1bgL/7cLaL3hqD2Yt."
            )
            
            logger.info("Test data seeded successfully")
            
        finally:
            await conn.close()
            
    async def initialize(self):
        """Run full initialization process"""
        logger.info("Starting database initialization...")
        
        # Create database
        await self.create_database()
        
        # Run migrations
        await self.run_migrations()
        
        # Validate schema
        schema_valid = await self.validate_schema()
        if not schema_valid:
            raise Exception("Schema validation failed")
            
        # Seed test data
        await self.seed_test_data()
        
        logger.info("Database initialization completed successfully!")


async def main():
    """Main initialization function"""
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set in environment")
        sys.exit(1)
        
    # Initialize database
    initializer = DatabaseInitializer(database_url)
    
    try:
        await initializer.initialize()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())