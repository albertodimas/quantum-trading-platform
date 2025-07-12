-- Authentication Schema Migration
-- Creates tables for users, roles, permissions, and API keys

-- Create permissions table
CREATE TABLE IF NOT EXISTS quantum.permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create roles table
CREATE TABLE IF NOT EXISTS quantum.roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS quantum.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    preferences JSONB DEFAULT '{}',
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create role_permissions junction table
CREATE TABLE IF NOT EXISTS quantum.role_permissions (
    role_id UUID NOT NULL REFERENCES quantum.roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES quantum.permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (role_id, permission_id)
);

-- Create user_roles junction table
CREATE TABLE IF NOT EXISTS quantum.user_roles (
    user_id UUID NOT NULL REFERENCES quantum.users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES quantum.roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id)
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS quantum.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES quantum.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    permissions TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create refresh tokens table (for token blacklisting)
CREATE TABLE IF NOT EXISTS quantum.refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES quantum.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_blacklisted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create password reset tokens table
CREATE TABLE IF NOT EXISTS quantum.password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES quantum.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON quantum.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON quantum.users(username);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON quantum.users(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON quantum.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON quantum.api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON quantum.api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON quantum.user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role_id ON quantum.user_roles(role_id);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION quantum.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON quantum.users
    FOR EACH ROW EXECUTE FUNCTION quantum.update_updated_at_column();

CREATE TRIGGER update_roles_updated_at BEFORE UPDATE ON quantum.roles
    FOR EACH ROW EXECUTE FUNCTION quantum.update_updated_at_column();

-- Insert default permissions
INSERT INTO quantum.permissions (name, description) VALUES
    ('read_portfolio', 'View portfolio data'),
    ('write_portfolio', 'Modify portfolio data'),
    ('execute_trades', 'Execute trading orders'),
    ('manage_strategies', 'Create and manage trading strategies'),
    ('admin_access', 'Full administrative access'),
    ('manage_users', 'Manage user accounts'),
    ('view_all_portfolios', 'View all user portfolios'),
    ('api_full_access', 'Full API access'),
    ('api_read_only', 'Read-only API access')
ON CONFLICT (name) DO NOTHING;

-- Insert default roles
INSERT INTO quantum.roles (name, description) VALUES
    ('admin', 'Full system access'),
    ('trader', 'Can execute trades and manage strategies'),
    ('viewer', 'Read-only access'),
    ('bot', 'Automated trading bot access')
ON CONFLICT (name) DO NOTHING;

-- Assign permissions to roles
-- Admin role gets all permissions
INSERT INTO quantum.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM quantum.roles r, quantum.permissions p
WHERE r.name = 'admin'
ON CONFLICT DO NOTHING;

-- Trader role permissions
INSERT INTO quantum.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM quantum.roles r, quantum.permissions p
WHERE r.name = 'trader' 
AND p.name IN ('read_portfolio', 'write_portfolio', 'execute_trades', 'manage_strategies', 'api_full_access')
ON CONFLICT DO NOTHING;

-- Viewer role permissions
INSERT INTO quantum.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM quantum.roles r, quantum.permissions p
WHERE r.name = 'viewer' 
AND p.name IN ('read_portfolio', 'api_read_only')
ON CONFLICT DO NOTHING;

-- Bot role permissions
INSERT INTO quantum.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM quantum.roles r, quantum.permissions p
WHERE r.name = 'bot' 
AND p.name IN ('read_portfolio', 'execute_trades', 'api_full_access')
ON CONFLICT DO NOTHING;

-- Create default admin user (password: admin123 - CHANGE THIS!)
-- Password hash is for 'admin123' using bcrypt
INSERT INTO quantum.users (email, username, password_hash, is_superuser, full_name)
VALUES (
    'admin@quantumtrading.io',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGHTGXjzFxi',
    TRUE,
    'System Administrator'
)
ON CONFLICT (username) DO NOTHING;

-- Assign admin role to admin user
INSERT INTO quantum.user_roles (user_id, role_id)
SELECT u.id, r.id
FROM quantum.users u, quantum.roles r
WHERE u.username = 'admin' AND r.name = 'admin'
ON CONFLICT DO NOTHING;

-- Add table comments
COMMENT ON TABLE quantum.users IS 'User accounts for authentication';
COMMENT ON TABLE quantum.roles IS 'Roles for role-based access control';
COMMENT ON TABLE quantum.permissions IS 'Granular permissions for authorization';
COMMENT ON TABLE quantum.api_keys IS 'API keys for programmatic access';
COMMENT ON TABLE quantum.refresh_tokens IS 'JWT refresh tokens for session management';
COMMENT ON TABLE quantum.password_reset_tokens IS 'Tokens for password reset functionality';