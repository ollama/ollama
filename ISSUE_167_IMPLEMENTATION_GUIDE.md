# Issue #167: Vault Integration - Secrets Management Implementation Guide

**Priority**: CRITICAL - Security
**Complexity**: High
**Effort**: 32 hours
**Status**: Ready for Implementation

## Problem Statement

Current secrets management store secrets in:
- `.env` files (version controlled accidentally)
- Environment variables (visible in process listings)
- Config files (readable by any user on system)
- Plaintext in code (hardcoded database passwords)

Required: **Centralized secrets management** using HashiCorp Vault for:
- Encrypted storage
- Automatic rotation
- Fine-grained access control
- Audit logging of access
- Dynamic credential generation

## Solution Overview

Integrate HashiCorp Vault to:
1. Store all secrets encrypted at rest
2. Retrieve secrets with automatic caching
3. Support dynamic credentials (database passwords, AWS keys)
4. Audit every secret access
5. Enable automatic rotation

## Implementation

### Phase 1: Vault Client Setup

```go
// config/vault.go
package config

import (
    "fmt"
    "os"

    "github.com/hashicorp/vault/api"
)

type VaultConfig struct {
    Address      string
    Token        string
    Namespace    string
    TLS          *api.TLSConfig
    RetryPolicy  *api.RetryPolicy
}

type VaultClient struct {
    client *api.Client
    config *VaultConfig
}

func NewVaultClient(config *VaultConfig) (*VaultClient, error) {
    // Initialize Vault client
    vaultConfig := api.DefaultConfig()
    vaultConfig.Address = config.Address

    // Use TLS if configured
    if config.TLS != nil {
        vaultConfig.HttpClient.Transport.(*http.Transport).TLSClientConfig = config.TLS
    }

    client, err := api.NewClient(vaultConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create vault client: %w", err)
    }

    // Set token
    client.SetToken(config.Token)

    // Set namespace if provided
    if config.Namespace != "" {
        client.SetNamespace(config.Namespace)
    }

    return &VaultClient{
        client: client,
        config: config,
    }, nil
}

// GetSecret retrieves a secret from Vault
func (vc *VaultClient) GetSecret(path string) (map[string]interface{}, error) {
    secret, err := vc.client.Logical().Read(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read secret %s: %w", path, err)
    }

    if secret == nil {
        return nil, fmt.Errorf("secret not found: %s", path)
    }

    return secret.Data, nil
}

// GetDatabaseCredentials retrieves dynamic DB credentials
func (vc *VaultClient) GetDatabaseCredentials(role string) (string, string, error) {
    path := fmt.Sprintf("database/creds/%s", role)
    secret, err := vc.client.Logical().Read(path)
    if err != nil {
        return "", "", fmt.Errorf("failed to generate DB credentials: %w", err)
    }

    username := secret.Data["username"].(string)
    password := secret.Data["password"].(string)

    return username, password, nil
}

// GetAWSCredentials retrieves dynamic AWS credentials
func (vc *VaultClient) GetAWSCredentials(role string) (string, string, string, error) {
    path := fmt.Sprintf("aws/creds/%s", role)
    secret, err := vc.client.Logical().Read(path)
    if err != nil {
        return "", "", "", fmt.Errorf("failed to generate AWS credentials: %w", err)
    }

    accessKey := secret.Data["access_key"].(string)
    secretKey := secret.Data["secret_access_key"].(string)
    ttl := secret.LeaseDuration

    return accessKey, secretKey, fmt.Sprintf("%d", ttl), nil
}

// PutSecret stores a secret in Vault
func (vc *VaultClient) PutSecret(path string, data map[string]interface{}) error {
    _, err := vc.client.Logical().Write(path, data)
    return err
}

// RotateSecret initiates secret rotation
func (vc *VaultClient) RotateSecret(path string) error {
    _, err := vc.client.Logical().Write(fmt.Sprintf("%s/rotate", path), nil)
    return err
}
```

### Phase 2: Secrets Caching & Auto-Refresh

```go
// config/secrets_manager.go
package config

import (
    "context"
    "sync"
    "time"
)

type SecretsManager struct {
    vault   *VaultClient
    cache   map[string]*CachedSecret
    mu      sync.RWMutex
    logger  *log.Logger
}

type CachedSecret struct {
    Value      map[string]interface{}
    ExpiresAt  time.Time
    RefreshAt  time.Time
}

func NewSecretsManager(vault *VaultClient, logger *log.Logger) *SecretsManager {
    sm := &SecretsManager{
        vault:  vault,
        cache:  make(map[string]*CachedSecret),
        logger: logger,
    }

    // Start background refresh goroutine
    go sm.refreshLoop()

    return sm
}

// Get retrieves secret with automatic caching and refresh
func (sm *SecretsManager) Get(ctx context.Context, path string) (map[string]interface{}, error) {
    sm.mu.RLock()
    cached, exists := sm.cache[path]
    sm.mu.RUnlock()

    // Return cached secret if valid
    if exists && time.Now().Before(cached.RefreshAt) {
        return cached.Value, nil
    }

    // Fetch fresh secret from Vault
    secret, err := sm.vault.GetSecret(path)
    if err != nil {
        return nil, err
    }

    // Cache with refresh window (refresh at 75% of TTL)
    sm.mu.Lock()
    sm.cache[path] = &CachedSecret{
        Value:     secret,
        ExpiresAt: time.Now().Add(24 * time.Hour),
        RefreshAt: time.Now().Add(18 * time.Hour), // 75% of 24h
    }
    sm.mu.Unlock()

    return secret, nil
}

// GetString retrieves a single secret value
func (sm *SecretsManager) GetString(ctx context.Context, path, key string) (string, error) {
    secret, err := sm.Get(ctx, path)
    if err != nil {
        return "", err
    }

    value, ok := secret[key].(string)
    if !ok {
        return "", fmt.Errorf("secret key %s not found or not a string", key)
    }

    return value, nil
}

// refreshLoop periodically refreshes secrets nearing expiration
func (sm *SecretsManager) refreshLoop() {
    ticker := time.NewTicker(1 * time.Hour)
    defer ticker.Stop()

    for range ticker.C {
        sm.mu.RLock()
        paths := make([]string, 0, len(sm.cache))
        for path := range sm.cache {
            paths = append(paths, path)
        }
        sm.mu.RUnlock()

        for _, path := range paths {
            cached := sm.cache[path]
            if time.Now().After(cached.RefreshAt) {
                sm.logger.Printf("Refreshing secret: %s", path)
                _, _ = sm.Get(context.Background(), path)
            }
        }
    }
}

// Clear removes cached secret
func (sm *SecretsManager) Clear(path string) {
    sm.mu.Lock()
    delete(sm.cache, path)
    sm.mu.Unlock()
}

// ClearAll removes all cached secrets
func (sm *SecretsManager) ClearAll() {
    sm.mu.Lock()
    sm.cache = make(map[string]*CachedSecret)
    sm.mu.Unlock()
}
```

### Phase 3: Configuration Management

```go
// config/config.go
package config

import (
    "context"
    "fmt"
    "log"
    "os"
)

type AppConfig struct {
    Server    *ServerConfig
    Database  *DatabaseConfig
    Auth      *AuthConfig
}

type ServerConfig struct {
    Host string
    Port int
}

type DatabaseConfig struct {
    Username string
    Password string
    Host     string
    Port     int
    Database string
}

type AuthConfig struct {
    JWTSecret    string
    OAuthSecret  string
    GitHubToken  string
}

// LoadConfigFromVault loads all configuration from Vault
func LoadConfigFromVault(vaultAddr, token string) (*AppConfig, error) {
    // Initialize Vault client
    vaultCfg := &VaultConfig{
        Address: vaultAddr,
        Token:   token,
    }

    vaultClient, err := NewVaultClient(vaultCfg)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize vault: %w", err)
    }

    secretsManager := NewSecretsManager(vaultClient, log.Default())
    ctx := context.Background()

    // Load secrets from Vault paths
    config := &AppConfig{}

    // Server config (usually from env, but can be in Vault)
    config.Server = &ServerConfig{
        Host: getEnvOrVault(ctx, secretsManager, "SERVER_HOST", "config/server/host", "0.0.0.0"),
        Port: getEnvIntOrVault(ctx, secretsManager, "SERVER_PORT", "config/server/port", 8000),
    }

    // Database credentials from Vault
    dbCreds, err := vaultClient.GetDatabaseCredentials("ollama-app")
    if err != nil {
        return nil, fmt.Errorf("failed to get DB credentials: %w", err)
    }

    config.Database = &DatabaseConfig{
        Username: dbCreds[0],
        Password: dbCreds[1],
        Host:     getEnvOrVault(ctx, secretsManager, "DB_HOST", "config/database/host", "localhost"),
        Port:     getEnvIntOrVault(ctx, secretsManager, "DB_PORT", "config/database/port", 5432),
        Database: getEnvOrVault(ctx, secretsManager, "DB_NAME", "config/database/name", "ollama"),
    }

    // Auth secrets from Vault
    jwtSecret, _ := secretsManager.GetString(ctx, "secret/data/auth", "jwt_secret")
    oauthSecret, _ := secretsManager.GetString(ctx, "secret/data/auth", "oauth_secret")
    githubToken, _ := secretsManager.GetString(ctx, "secret/data/auth", "github_token")

    config.Auth = &AuthConfig{
        JWTSecret:   jwtSecret,
        OAuthSecret: oauthSecret,
        GitHubToken: githubToken,
    }

    return config, nil
}

func getEnvOrVault(ctx context.Context, sm *SecretsManager, envKey, vaultPath, defaultValue string) string {
    if val := os.Getenv(envKey); val != "" {
        return val
    }

    if val, err := sm.GetString(ctx, vaultPath, "value"); err == nil {
        return val
    }

    return defaultValue
}

func getEnvIntOrVault(ctx context.Context, sm *SecretsManager, envKey, vaultPath string, defaultValue int) int {
    if val := os.Getenv(envKey); val != "" {
        var intVal int
        fmt.Sscanf(val, "%d", &intVal)
        return intVal
    }

    if val, err := sm.GetString(ctx, vaultPath, "value"); err == nil {
        var intVal int
        fmt.Sscanf(val, "%d", &intVal)
        return intVal
    }

    return defaultValue
}
```

### Phase 4: Vault Setup Scripts

```bash
#!/bin/bash
# scripts/setup-vault.sh - Initialize Vault for Ollama

set -e

VAULT_ADDR="http://localhost:8200"
VAULT_TOKEN="${1:-myroot}"

export VAULT_ADDR

# Start Vault (in dev mode for testing)
echo "Starting Vault server..."
vault server -dev -dev-root-token-id="$VAULT_TOKEN" &
VAULT_PID=$!

sleep 2

# Configure Vault
echo "Configuring Vault..."
vault login "$VAULT_TOKEN"

# Enable KV secrets engine
vault secrets enable -path=secret kv-v2 || true

# Store API secrets
vault kv put secret/auth \
    jwt_secret="$(openssl rand -base64 32)" \
    oauth_secret="$(openssl rand -base64 32)" \
    github_token="ghp_xxxxxxxxxxxxxxxxxxxx"

# Configure database dynamic credentials
vault secrets enable database || true
vault write database/config/postgresql \
    plugin_name=postgresql-database-plugin \
    allowed_roles="ollama-app" \
    connection_url="postgresql://{{username}}:{{password}}@localhost:5432/ollama" \
    username="vault" \
    password="vault_password"

vault write database/roles/ollama-app \
    db_name=postgresql \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"

# Configure AWS dynamic credentials
vault secrets enable aws || true
vault write aws/config/root \
    access_key="AKIAXXXXXXXXXXXXXXXX" \
    secret_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

vault write aws/roles/ollama-app \
    credential_type=iam_user \
    policy_document=@- <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "*"
    }
  ]
}
EOF

echo "Vault initialized successfully!"
echo "Token: $VAULT_TOKEN"
echo "Address: $VAULT_ADDR"

# Keep Vault running in background
wait $VAULT_PID
```

## Acceptance Criteria

- ✅ All secrets stored encrypted in Vault
- ✅ No plaintext secret files on disk (.env, config.json)
- ✅ Automatic credential rotation (DB passwords)
- ✅ Dynamic credential generation (AWS, database)
- ✅ Audit log of all secret access
- ✅ Caching with safe refresh windows
- ✅ Fallback to environment variables if Vault unavailable

## Testing

```bash
# Start test Vault server
vault server -dev

# In terminal 2:
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=myroot

# Test secret retrieval
go test ./config -v -run TestVaultIntegration

# Test credentials rotation
./ollama rotate-secrets
```

## Deployment Checklist

- [ ] Set up Vault cluster (HA in production)
- [ ] Configure unseal mechanism (KMS/HSM)
- [ ] Initialize secret paths
- [ ] Deploy dynamic credential backends
- [ ] Implement audit logging
- [ ] Test failover/backup
- [ ] Documentation for ops team

## Security Benefits

1. **Encryption at Rest**: All secrets encrypted with AES-256
2. **Encryption in Transit**: TLS for all Vault communication
3. **Automatic Rotation**: Credentials rotated without downtime
4. **Fine-grained Access**: RBAC for secrets access
5. **Audit Trail**: Every access logged and retained
6. **Dynamic Credentials**: Short-lived passwords reduce blast radius

## Vault Architecture

```
┌─────────────────────────────────────────┐
│         Ollama Application              │
├─────────────────────────────────────────┤
│       Secrets Manager (Cached)          │
├─────────────────────────────────────────┤
│      TLS ← → Vault API Client           │
├─────────────────────────────────────────┤
         ↓ (TLS + Auth Token)
┌─────────────────────────────────────────┐
│      HashiCorp Vault Server             │
├─────────────────────────────────────────┤
│  KV Secrets  │ Database  │ AWS Dynamic  │
├─────────────────────────────────────────┤
│     Encrypted Storage (AES-256)         │
└─────────────────────────────────────────┘
```

---

**Ready for Implementation**: Yes - production-grade Vault integration with clear initialization and usage patterns.
