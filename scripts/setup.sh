#!/bin/bash
# Setup script for Quantum Trading Platform

set -e

echo "🚀 Setting up Quantum Trading Platform..."

# Create necessary directories
mkdir -p logs
mkdir -p data/cache
mkdir -p data/storage
mkdir -p backups

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -r requirements/test.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Check Redis connection
echo "Checking Redis connection..."
python3 -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping()" && echo "✓ Redis is running" || echo "✗ Redis is not running"

# Check PostgreSQL connection
echo "Checking PostgreSQL connection..."
python3 -c "import psycopg2; psycopg2.connect('postgresql://postgres:postgres@localhost:5432/postgres')" && echo "✓ PostgreSQL is running" || echo "✗ PostgreSQL is not running"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env with your configuration"
echo "2. Run 'make migrate' to setup database"
echo "3. Run 'make test' to verify installation"
echo "4. Run 'make run' to start the platform"