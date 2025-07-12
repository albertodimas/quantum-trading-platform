#!/usr/bin/env python3
"""
Test connections to external services

This script validates connectivity to:
- PostgreSQL database
- Redis cache
- RabbitMQ message broker
- Exchange APIs (testnet)
"""

import os
import sys
import asyncio
import asyncpg
import redis
import aio_pika
from pathlib import Path
from typing import Dict, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionTester:
    """Test connections to all external services"""
    
    def __init__(self):
        self.results: Dict[str, Tuple[bool, str]] = {}
        
    async def test_postgresql(self) -> Tuple[bool, str]:
        """Test PostgreSQL connection"""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return False, "DATABASE_URL not configured"
            
        try:
            # Try to connect
            conn = await asyncpg.connect(database_url)
            
            # Test query
            version = await conn.fetchval("SELECT version()")
            
            # Check for TimescaleDB
            timescale = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                )
            """)
            
            await conn.close()
            
            msg = f"Connected to PostgreSQL: {version.split(',')[0]}"
            if timescale:
                msg += " (TimescaleDB enabled)"
            else:
                msg += " (TimescaleDB NOT installed)"
                
            return True, msg
            
        except Exception as e:
            return False, f"PostgreSQL connection failed: {str(e)}"
            
    def test_redis(self) -> Tuple[bool, str]:
        """Test Redis connection"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_password = os.getenv("REDIS_PASSWORD")
        
        try:
            # Parse URL and create client
            if redis_password:
                redis_url = redis_url.replace("://", f"://:{redis_password}@")
                
            client = redis.from_url(redis_url)
            
            # Test connection
            client.ping()
            
            # Get info
            info = client.info()
            version = info.get('redis_version', 'unknown')
            
            # Test set/get
            client.set('test_key', 'test_value', ex=10)
            value = client.get('test_key')
            client.delete('test_key')
            
            return True, f"Connected to Redis v{version}"
            
        except Exception as e:
            return False, f"Redis connection failed: {str(e)}"
            
    async def test_rabbitmq(self) -> Tuple[bool, str]:
        """Test RabbitMQ connection"""
        rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        
        try:
            # Connect to RabbitMQ
            connection = await aio_pika.connect_robust(rabbitmq_url)
            
            # Create channel
            channel = await connection.channel()
            
            # Declare test queue
            queue = await channel.declare_queue("test_queue", auto_delete=True)
            
            # Get server properties
            props = connection.connection.server_properties
            version = props.get(b'version', b'unknown').decode()
            
            await connection.close()
            
            return True, f"Connected to RabbitMQ v{version}"
            
        except Exception as e:
            return False, f"RabbitMQ connection failed: {str(e)}"
            
    def test_binance_testnet(self) -> Tuple[bool, str]:
        """Test Binance testnet API"""
        import requests
        
        api_key = os.getenv("BINANCE_API_KEY")
        if not api_key or api_key == "your-binance-testnet-api-key":
            return False, "Binance testnet API key not configured"
            
        try:
            # Test public endpoint first
            base_url = "https://testnet.binance.vision"
            response = requests.get(f"{base_url}/api/v3/ping")
            
            if response.status_code != 200:
                return False, f"Binance testnet ping failed: {response.status_code}"
                
            # Test time endpoint
            response = requests.get(f"{base_url}/api/v3/time")
            if response.status_code == 200:
                server_time = response.json()['serverTime']
                return True, f"Connected to Binance testnet (server time: {server_time})"
            else:
                return False, f"Binance testnet time check failed: {response.status_code}"
                
        except Exception as e:
            return False, f"Binance testnet connection failed: {str(e)}"
            
    async def test_all_connections(self):
        """Test all connections"""
        logger.info("Testing connections to external services...")
        print("\n" + "="*60)
        print("QUANTUM TRADING PLATFORM - CONNECTION TEST")
        print("="*60 + "\n")
        
        # Test PostgreSQL
        print("1. Testing PostgreSQL...")
        success, msg = await self.test_postgresql()
        self.results['PostgreSQL'] = (success, msg)
        print(f"   {'✓' if success else '✗'} {msg}\n")
        
        # Test Redis
        print("2. Testing Redis...")
        success, msg = self.test_redis()
        self.results['Redis'] = (success, msg)
        print(f"   {'✓' if success else '✗'} {msg}\n")
        
        # Test RabbitMQ
        print("3. Testing RabbitMQ...")
        success, msg = await self.test_rabbitmq()
        self.results['RabbitMQ'] = (success, msg)
        print(f"   {'✓' if success else '✗'} {msg}\n")
        
        # Test Binance testnet
        print("4. Testing Binance Testnet API...")
        success, msg = self.test_binance_testnet()
        self.results['Binance Testnet'] = (success, msg)
        print(f"   {'✓' if success else '✗'} {msg}\n")
        
        # Summary
        print("="*60)
        print("SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for success, _ in self.results.values() if success)
        
        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        
        if passed_tests == total_tests:
            print("\n✅ All connections successful!")
            return True
        else:
            print("\n❌ Some connections failed. Please check your configuration.")
            print("\nFailed services:")
            for service, (success, msg) in self.results.items():
                if not success:
                    print(f"  - {service}: {msg}")
            return False
            
    def print_docker_commands(self):
        """Print helpful Docker commands"""
        print("\n" + "="*60)
        print("HELPFUL COMMANDS")
        print("="*60)
        print("\nTo start required services with Docker:")
        print("  docker-compose up -d postgres redis rabbitmq")
        print("\nTo check service status:")
        print("  docker-compose ps")
        print("\nTo view service logs:")
        print("  docker-compose logs -f [service_name]")
        print("\nTo stop services:")
        print("  docker-compose down")
        print("\nTo initialize the database:")
        print("  python scripts/init_database.py")


async def main():
    """Main test function"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    tester = ConnectionTester()
    success = await tester.test_all_connections()
    
    # Print helpful commands
    tester.print_docker_commands()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())