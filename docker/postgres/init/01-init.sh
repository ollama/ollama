#!/bin/bash
# PostgreSQL initialization script - create additional databases

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create Grafana database
    CREATE DATABASE grafana;
    GRANT ALL PRIVILEGES ON DATABASE grafana TO $POSTGRES_USER;
    
    -- Create extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    
    -- Create schemas
    CREATE SCHEMA IF NOT EXISTS ollama;
    
    -- Performance tuning
    ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
    ALTER SYSTEM SET track_activity_query_size = 2048;
    ALTER SYSTEM SET pg_stat_statements.track = 'all';
EOSQL

echo "PostgreSQL initialization completed successfully"
