# PostgreSQL & Cloud SQL Integration Guide

**Version**: 2.0.0 | **Date**: January 13, 2026 | **Status**: 🟢 PRODUCTION READY

---

## Overview

This guide covers integrating PostgreSQL/Cloud SQL with the Ollama Elite AI Platform for persistent data storage, user management, API key tracking, and conversation history.

---

## 📋 Quick Start

### 1. Local Development (Docker)

```bash
# Start PostgreSQL with Docker Compose
docker-compose up postgres

# Verify connection
psql postgresql://postgres:password@localhost:5432/ollama

# Run migrations
alembic upgrade head

# Check schema
\dt  # list tables
\d users  # describe table
```

### 2. Production (Cloud SQL)

```bash
# Create instance
gcloud sql instances create ollama-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create ollama --instance=ollama-db

# Run migrations via Cloud SQL Proxy
alembic upgrade head
```

---

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### API Keys Table
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    rate_limit INT DEFAULT 100,
    last_used_at TIMESTAMP,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Conversations Table
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    model VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    tokens_used INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Usage Tracking Table
```sql
CREATE TABLE usage (
    id UUID PRIMARY KEY,
    api_key_id UUID REFERENCES api_keys(id),
    endpoint VARCHAR(255),
    model VARCHAR(255),
    tokens_used INT,
    latency_ms INT,
    status_code INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Connection Management

### SQLAlchemy Configuration

```python
# config/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool

# Development (local PostgreSQL)
DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/ollama"

# Production (Cloud SQL)
DATABASE_URL = "postgresql+asyncpg://ollama:PASSWORD@/ollama?unix_sock=/cloudsql/elevatediq:us-central1:ollama-db/.s.PGSQL.5432"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connection before use
    pool_recycle=3600    # Recycle connections every hour
)

# Session factory
async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
```

### Connection Pooling Best Practices

```python
# Configuration for different environments

# Development: 5 connections
pool_size = 5
max_overflow = 10

# Production: 20 connections
pool_size = 20
max_overflow = 30

# Connection timeout
pool_timeout = 30

# Pre-ping for stale connections
pool_pre_ping = True

# Recycle old connections
pool_recycle = 3600  # 1 hour
```

---

## Migrations

### Create New Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add user fields"

# Edit migration file manually if needed
alembic/versions/001_add_user_fields.py

# Apply migration
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

### Migration Example

```python
# alembic/versions/001_initial_schema.py
"""Create initial schema.

Revision ID: 001
Revises:
Create Date: 2026-01-13 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import uuid

def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('username', sa.String(255), unique=True, nullable=False),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

def downgrade() -> None:
    op.drop_table('users')
```

---

## ORM Models

### User Model

```python
# ollama/models.py
from sqlalchemy import Column, String, Boolean, DateTime, UUID
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
```

### API Key Model

```python
class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True)  # Never store raw keys
    name: Mapped[str | None] = mapped_column(String(255))
    rate_limit: Mapped[int] = mapped_column(default=100)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="api_keys")
```

---

## Query Examples

### User Management

```python
# Create user
async def create_user(session: AsyncSession, username: str, email: str, password: str) -> User:
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user

# Get user by username
async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

# List all users (paginated)
async def list_users(session: AsyncSession, skip: int = 0, limit: int = 100) -> list[User]:
    result = await session.execute(
        select(User).offset(skip).limit(limit)
    )
    return result.scalars().all()
```

### API Key Management

```python
# Create API key
async def create_api_key(session: AsyncSession, user_id: UUID, name: str) -> str:
    raw_key = f"sk_{secrets.token_urlsafe(32)}"
    key_hash = hash_key(raw_key)

    api_key = APIKey(
        user_id=user_id,
        key_hash=key_hash,
        name=name
    )
    session.add(api_key)
    await session.commit()

    return raw_key  # Return only once to user

# Verify API key
async def verify_api_key(session: AsyncSession, raw_key: str) -> APIKey | None:
    key_hash = hash_key(raw_key)
    result = await session.execute(
        select(APIKey)
        .where(APIKey.key_hash == key_hash)
        .where(APIKey.revoked == False)
        .where((APIKey.expires_at == None) | (APIKey.expires_at > datetime.utcnow()))
    )
    api_key = result.scalar_one_or_none()

    if api_key:
        api_key.last_used_at = datetime.utcnow()
        await session.commit()

    return api_key
```

### Conversation History

```python
# Create conversation
async def create_conversation(session: AsyncSession, user_id: UUID, model: str, title: str = None) -> Conversation:
    conv = Conversation(
        user_id=user_id,
        model=model,
        title=title or f"Conversation with {model}"
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv

# Add message to conversation
async def add_message(session: AsyncSession, conversation_id: UUID, role: str, content: str, tokens_used: int = None) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tokens_used=tokens_used
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message

# Get conversation history
async def get_conversation_history(session: AsyncSession, conversation_id: UUID) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return result.scalars().all()
```

---

## Performance Optimization

### Indexing Strategy

```sql
-- User lookups
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- API key lookups
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);

-- Conversation queries
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- Message queries
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Usage tracking
CREATE INDEX idx_usage_api_key_id ON usage(api_key_id);
CREATE INDEX idx_usage_created_at ON usage(created_at);
CREATE INDEX idx_usage_model ON usage(model);
```

### Query Optimization

```python
# Use eager loading to avoid N+1 queries
from sqlalchemy.orm import joinedload

result = await session.execute(
    select(User)
    .options(joinedload(User.api_keys))
    .options(joinedload(User.conversations))
)
users = result.unique().scalars().all()

# Use select() for specific columns
result = await session.execute(
    select(User.id, User.username, User.email)
)
```

---

## Backup & Recovery

### Automated Backups

```bash
# Cloud SQL automatic backups (done by GCP)
gcloud sql backups create --instance=ollama-db

# Manual backup
pg_dump postgresql://ollama:password@cloudsql/ollama > backup.sql

# Restore from backup
psql postgresql://ollama:password@cloudsql/ollama < backup.sql

# Backup to Google Cloud Storage
gcloud sql export sql ollama-db gs://ollama-backups/backup-$(date +%Y%m%d).sql
```

### Point-in-Time Recovery

```bash
# Cloud SQL supports PITR by default
# Restore to specific point in time
gcloud sql backups describe <BACKUP_ID> --instance=ollama-db

# Restore with gcloud
gcloud sql backups restore <BACKUP_ID> \
  --backup-instance=ollama-db \
  --backup-configuration=default
```

---

## Monitoring & Maintenance

### Connection Monitoring

```sql
-- Check active connections
SELECT datname, count(*) as connection_count
FROM pg_stat_activity
GROUP BY datname
ORDER BY connection_count DESC;

-- Kill idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'ollama'
  AND state = 'idle'
  AND query_start < NOW() - INTERVAL '1 hour';
```

### Query Performance

```sql
-- Slowest queries (requires pg_stat_statements)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Reset statistics
SELECT pg_stat_statements_reset();
```

### Table Statistics

```sql
-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Troubleshooting

### Connection Issues

```bash
# Test connection
psql -h localhost -U postgres -d ollama -c "SELECT 1"

# Check if PostgreSQL is running
docker ps | grep postgres

# View PostgreSQL logs
docker logs <container_id>

# Restart container
docker restart <container_id>
```

### Performance Issues

```bash
# Profile queries
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'admin';

# Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

# Analyze table statistics
ANALYZE users;
```

### Migration Issues

```bash
# Check migration status
alembic current

# List all migrations
alembic history

# Rollback specific migration
alembic downgrade -1

# Validate migration
alembic upgrade --sql head
```

---

## Environment Configuration

### Development (.env.dev)

```bash
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/ollama
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434

# For local development with real IP
REAL_IP=$(hostname -I | awk '{print $1}')
PUBLIC_API_URL=http://$REAL_IP:8000
```

### Production (.env.prod - GCP Secret Manager)

```bash
DATABASE_URL=postgresql+asyncpg://ollama:${DB_PASSWORD}@/ollama?unix_sock=/cloudsql/${PROJECT}:${REGION}:${INSTANCE}/.s.PGSQL.5432
REDIS_URL=redis://:${REDIS_PASSWORD}@redis-instance:6379/0
QDRANT_URL=http://qdrant-instance:6333
OLLAMA_BASE_URL=http://ollama-instance:11434

PUBLIC_API_URL=https://ollama.elevatediq.ai
```

---

**For more information, see [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md)**
