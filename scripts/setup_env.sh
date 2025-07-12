#!/bin/bash

# Quantum Trading Platform - Environment Setup Script

echo "🚀 Quantum Trading Platform - Environment Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if command -v python3.11 &> /dev/null; then
    python3.11 -m venv quantum_venv
elif command -v python3.12 &> /dev/null; then
    # For Python 3.12, you may need to install python3.12-venv first:
    # sudo apt install python3.12-venv
    python3.12 -m venv quantum_venv 2>/dev/null || {
        echo "⚠️  Warning: python3-venv not installed."
        echo "   Please run: sudo apt install python3.12-venv"
        echo "   Then run this script again."
        exit 1
    }
else
    python3 -m venv quantum_venv
fi

# Activate virtual environment
echo "✓ Virtual environment created"
source quantum_venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install test dependencies
echo ""
echo "📦 Installing test dependencies..."
pip install -r requirements-test.txt

# Create necessary directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/historical
mkdir -p logs
mkdir -p models
mkdir -p migrations

# Copy .env.example to .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created from .env.example"
    echo "⚠️  Please update .env with your configuration"
fi

# Generate secret key
echo ""
echo "🔐 Generating secret key..."
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
sed -i "s/your-secret-key-here-change-in-production/$SECRET_KEY/g" .env
echo "✓ Secret key generated"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source quantum_venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Update .env with your database credentials"
echo "2. Update .env with your exchange API keys (testnet)"
echo "3. Run: docker-compose up -d postgres redis"
echo "4. Run: python scripts/init_database.py"
echo ""