-- Quantum Trading Platform - Initial Database Schema
-- Version: 1.0.0
-- Date: 2025-07-12

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create schema
CREATE SCHEMA IF NOT EXISTS quantum;

-- Set default schema
SET search_path TO quantum, public;

-- =====================================================
-- USERS AND AUTHENTICATION
-- =====================================================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Roles table
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User roles mapping
CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

-- API keys for system access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    permissions JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- =====================================================
-- EXCHANGES AND TRADING CONFIGURATION
-- =====================================================

-- Supported exchanges
CREATE TABLE exchanges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_testnet BOOLEAN DEFAULT FALSE,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User exchange credentials
CREATE TABLE user_exchange_credentials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange_id UUID REFERENCES exchanges(id) ON DELETE CASCADE,
    api_key_encrypted TEXT NOT NULL,
    api_secret_encrypted TEXT NOT NULL,
    additional_credentials JSONB DEFAULT '{}'::jsonb, -- For passphrase, etc.
    is_active BOOLEAN DEFAULT TRUE,
    is_testnet BOOLEAN DEFAULT FALSE,
    permissions JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, exchange_id, is_testnet)
);

-- Trading pairs/symbols
CREATE TABLE symbols (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exchange_id UUID REFERENCES exchanges(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    min_quantity DECIMAL(20, 8),
    max_quantity DECIMAL(20, 8),
    step_size DECIMAL(20, 8),
    min_price DECIMAL(20, 8),
    max_price DECIMAL(20, 8),
    tick_size DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(exchange_id, symbol)
);

-- =====================================================
-- TRADING STRATEGIES
-- =====================================================

-- Strategy definitions
CREATE TABLE strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- Strategy instances (running strategies)
CREATE TABLE strategy_instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID REFERENCES strategies(id) ON DELETE CASCADE,
    exchange_id UUID REFERENCES exchanges(id) ON DELETE CASCADE,
    symbol_id UUID REFERENCES symbols(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'stopped', -- stopped, starting, running, stopping, error
    started_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    error_message TEXT,
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- ORDERS AND TRADES
-- =====================================================

-- Orders table (using TimescaleDB hypertable)
CREATE TABLE orders (
    id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_instance_id UUID REFERENCES strategy_instances(id) ON DELETE SET NULL,
    exchange_id UUID NOT NULL REFERENCES exchanges(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    exchange_order_id VARCHAR(100),
    client_order_id VARCHAR(100) UNIQUE,
    type VARCHAR(20) NOT NULL, -- market, limit, stop_loss, etc.
    side VARCHAR(10) NOT NULL, -- buy, sell
    status VARCHAR(20) NOT NULL, -- pending, open, filled, cancelled, rejected
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

-- Convert orders to hypertable
SELECT create_hypertable('orders', 'created_at', chunk_time_interval => INTERVAL '1 day');

-- Trades/fills table (using TimescaleDB hypertable)
CREATE TABLE trades (
    id UUID NOT NULL,
    order_id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exchange_id UUID NOT NULL REFERENCES exchanges(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    exchange_trade_id VARCHAR(100),
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),
    is_maker BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, executed_at)
);

-- Convert trades to hypertable
SELECT create_hypertable('trades', 'executed_at', chunk_time_interval => INTERVAL '1 day');

-- =====================================================
-- POSITIONS AND PORTFOLIO
-- =====================================================

-- Current positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_instance_id UUID REFERENCES strategy_instances(id) ON DELETE SET NULL,
    exchange_id UUID NOT NULL REFERENCES exchanges(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    side VARCHAR(10) NOT NULL, -- long, short
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, exchange_id, symbol_id, strategy_instance_id)
);

-- Position history (snapshots)
CREATE TABLE position_history (
    id UUID NOT NULL,
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    realized_pnl DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    snapshot_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, snapshot_at)
);

-- Convert position_history to hypertable
SELECT create_hypertable('position_history', 'snapshot_at', chunk_time_interval => INTERVAL '1 day');

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_value DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    positions_value DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8),
    total_pnl DECIMAL(20, 8),
    snapshot_at TIMESTAMPTZ NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, snapshot_at)
);

-- Convert portfolio_snapshots to hypertable
SELECT create_hypertable('portfolio_snapshots', 'snapshot_at', chunk_time_interval => INTERVAL '1 day');

-- =====================================================
-- MARKET DATA
-- =====================================================

-- Price ticks (using TimescaleDB hypertable)
CREATE TABLE market_ticks (
    exchange_id UUID NOT NULL REFERENCES exchanges(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8),
    side VARCHAR(10), -- buy, sell
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (exchange_id, symbol_id, timestamp)
);

-- Convert market_ticks to hypertable
SELECT create_hypertable('market_ticks', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- OHLCV candles (using TimescaleDB hypertable)
CREATE TABLE market_candles (
    exchange_id UUID NOT NULL REFERENCES exchanges(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    timeframe VARCHAR(10) NOT NULL, -- 1m, 5m, 15m, 1h, 4h, 1d
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trades_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (exchange_id, symbol_id, timeframe, open_time)
);

-- Convert market_candles to hypertable
SELECT create_hypertable('market_candles', 'open_time', chunk_time_interval => INTERVAL '1 day');

-- =====================================================
-- SYSTEM AND MONITORING
-- =====================================================

-- System events log
CREATE TABLE system_events (
    id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- info, warning, error, critical
    source VARCHAR(100),
    message TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, occurred_at)
);

-- Convert system_events to hypertable
SELECT create_hypertable('system_events', 'occurred_at', chunk_time_interval => INTERVAL '1 day');

-- Trading signals log
CREATE TABLE trading_signals (
    id UUID NOT NULL,
    strategy_instance_id UUID REFERENCES strategy_instances(id) ON DELETE CASCADE,
    signal_type VARCHAR(50) NOT NULL,
    strength DECIMAL(5, 4), -- 0.0000 to 1.0000
    action VARCHAR(20), -- buy, sell, hold
    details JSONB DEFAULT '{}'::jsonb,
    generated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, generated_at)
);

-- Convert trading_signals to hypertable
SELECT create_hypertable('trading_signals', 'generated_at', chunk_time_interval => INTERVAL '1 day');

-- =====================================================
-- INDEXES
-- =====================================================

-- Users and auth indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);

-- Exchange and trading indexes
CREATE INDEX idx_symbols_exchange ON symbols(exchange_id);
CREATE INDEX idx_symbols_active ON symbols(is_active) WHERE is_active = TRUE;

-- Orders indexes
CREATE INDEX idx_orders_user ON orders(user_id, created_at DESC);
CREATE INDEX idx_orders_status ON orders(status, created_at DESC);
CREATE INDEX idx_orders_exchange ON orders(exchange_id, created_at DESC);
CREATE INDEX idx_orders_symbol ON orders(symbol_id, created_at DESC);
CREATE INDEX idx_orders_client_order_id ON orders(client_order_id);

-- Trades indexes
CREATE INDEX idx_trades_user ON trades(user_id, executed_at DESC);
CREATE INDEX idx_trades_order ON trades(order_id);

-- Positions indexes
CREATE INDEX idx_positions_user ON positions(user_id);
CREATE INDEX idx_positions_open ON positions(closed_at) WHERE closed_at IS NULL;

-- Market data indexes
CREATE INDEX idx_market_ticks_symbol ON market_ticks(symbol_id, timestamp DESC);
CREATE INDEX idx_market_candles_symbol ON market_candles(symbol_id, timeframe, open_time DESC);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update timestamp triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default roles
INSERT INTO roles (name, description) VALUES
    ('admin', 'Full system access'),
    ('trader', 'Can execute trades and manage strategies'),
    ('viewer', 'Read-only access'),
    ('bot', 'Automated trading bot access');

-- Insert supported exchanges
INSERT INTO exchanges (name, display_name, is_active, is_testnet) VALUES
    ('binance', 'Binance', TRUE, FALSE),
    ('binance_testnet', 'Binance Testnet', TRUE, TRUE),
    ('coinbase', 'Coinbase Pro', TRUE, FALSE),
    ('coinbase_sandbox', 'Coinbase Sandbox', TRUE, TRUE),
    ('kraken', 'Kraken', TRUE, FALSE);

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Grant permissions to quantum schema
GRANT ALL ON SCHEMA quantum TO quantum_user;
GRANT ALL ON ALL TABLES IN SCHEMA quantum TO quantum_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA quantum TO quantum_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA quantum TO quantum_user;

-- Create read-only user for analytics
CREATE USER quantum_readonly WITH PASSWORD 'readonly_pass';
GRANT CONNECT ON DATABASE quantum_trading TO quantum_readonly;
GRANT USAGE ON SCHEMA quantum TO quantum_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA quantum TO quantum_readonly;

-- =====================================================
-- COMMENTS
-- =====================================================

COMMENT ON SCHEMA quantum IS 'Quantum Trading Platform main schema';
COMMENT ON TABLE users IS 'Platform users with authentication information';
COMMENT ON TABLE orders IS 'Trading orders with TimescaleDB hypertable for time-series optimization';
COMMENT ON TABLE trades IS 'Executed trades/fills with TimescaleDB hypertable';
COMMENT ON TABLE positions IS 'Current open positions for users';
COMMENT ON TABLE market_ticks IS 'Real-time market tick data with TimescaleDB hypertable';
COMMENT ON TABLE market_candles IS 'OHLCV candle data with TimescaleDB hypertable';