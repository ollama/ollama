-- ============================================================================
-- OLLAMA LOCAL DEVELOPMENT DATABASE INITIALIZATION
-- ============================================================================
-- This script is automatically executed by PostgreSQL container on startup
-- It creates schemas, extensions, and initial tables

-- Create EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create SCHEMAS
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS logs;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions
GRANT USAGE ON SCHEMA public TO ollama;
GRANT USAGE ON SCHEMA logs TO ollama;
GRANT USAGE ON SCHEMA audit TO ollama;

-- Set defaults
ALTER SCHEMA public OWNER TO ollama;
