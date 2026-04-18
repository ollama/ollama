# Issue #166: Network Segmentation & mTLS Implementation Guide

**Priority**: CRITICAL - Security
**Complexity**: High
**Effort**: 40 hours
**Status**: Ready for Implementation

## Problem Statement

Ollama currently binds to `0.0.0.0` (all interfaces), exposing the API to anyone with network access. This violates security best practices for sensitive workloads handling proprietary models and data.

## Solution Overview

1. **Network Binding**: Rebind from `0.0.0.0` to `127.0.0.1` (localhost) or internal IP only
2. **mTLS Certificate Management**: Integrate HashiCorp Vault PKI for automatic certificate generation
3. **Client Authentication**: Require valid client certificates for all API connections
4. **Configuration**: Make binding configurable via environment variables

## Implementation Steps

### Phase 1: Configuration & Environment Variables

```go
// cmd/main.go
import (
    "flag"
    "os"
)

type ServerConfig struct {
    // Network binding
    Host string // default: "127.0.0.1"
    Port int    // default: 8000

    // TLS/mTLS
    TLSEnabled    bool
    TLSCertFile   string
    TLSKeyFile    string
    TLSClientAuth bool // require client certificates
    CACertFile    string // CA cert for client validation

    // Vault integration
    VaultAddr      string
    VaultToken     string
    VaultPKIPath   string // e.g., "pki/issue/ollama"
}

func initServerConfig() ServerConfig {
    return ServerConfig{
        Host: os.Getenv("OLLAMA_HOST"),         // default: "127.0.0.1"
        Port: parseIntFromEnv("OLLAMA_PORT", 8000),
        TLSEnabled: os.Getenv("OLLAMA_TLS") == "true",
        TLSCertFile: os.Getenv("OLLAMA_CERT_FILE"),
        TLSKeyFile: os.Getenv("OLLAMA_KEY_FILE"),
        TLSClientAuth: os.Getenv("OLLAMA_MTLS") == "true",
    }
}
```

### Phase 2: mTLS Certificate Management (Vault Integration)

```go
// server/tls.go
package server

import (
    "crypto/tls"
    "fmt"

    "github.com/hashicorp/vault/api"
)

type VaultPKI struct {
    client *api.Client
    path   string // e.g., "pki/issue/ollama"
}

func NewVaultPKI(vaultAddr, token, path string) (*VaultPKI, error) {
    config := api.DefaultConfig()
    config.Address = vaultAddr

    client, err := api.NewClient(config)
    if err != nil {
        return nil, fmt.Errorf("vault client init failed: %w", err)
    }

    client.SetToken(token)

    return &VaultPKI{
        client: client,
        path:   path,
    }, nil
}

// IssueCertificate generates new mTLS certificate from Vault
func (v *VaultPKI) IssueCertificate(commonName string, ttl string) (*tls.Certificate, error) {
    // Request certificate from Vault PKI mount
    secret, err := v.client.Logical().Write(v.path, map[string]interface{}{
        "common_name": commonName,
        "ttl":         ttl, // e.g., "8760h" for 1 year
    })

    if err != nil {
        return nil, fmt.Errorf("vault certificate issue failed: %w", err)
    }

    // Parse certificate and key
    certData := secret.Data["certificate"].(string)
    keyData := secret.Data["private_key"].(string)

    // Load into tls.Certificate
    cert, err := tls.X509KeyPair(
        []byte(certData),
        []byte(keyData),
    )

    return &cert, err
}

// RotateCertificate refreshes certificate before expiry
func (v *VaultPKI) RotateCertificate(oldCert *tls.Certificate, commonName string) (*tls.Certificate, error) {
    return v.IssueCertificate(commonName, "8760h")
}
```

### Phase 3: Server with mTLS

```go
// server/server.go
func (s *Server) ListenAndServeTLS(config ServerConfig) error {
    var tlsConfig *tls.Config

    if config.TLSEnabled {
        var err error
        tlsConfig, err = s.buildTLSConfig(config)
        if err != nil {
            return fmt.Errorf("TLS config build failed: %w", err)
        }
    }

    // Create HTTP server
    srv := &http.Server{
        Addr:      fmt.Sprintf("%s:%d", config.Host, config.Port),
        Handler:   s.router,
        TLSConfig: tlsConfig,
        // Additional security headers
        ReadTimeout:  30 * time.Second,
        WriteTimeout: 30 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    if config.TLSEnabled {
        return srv.ListenAndServeTLS(
            config.TLSCertFile,
            config.TLSKeyFile,
        )
    }

    return srv.ListenAndServe()
}

func (s *Server) buildTLSConfig(config ServerConfig) (*tls.Config, error) {
    tlsConfig := &tls.Config{
        MinVersion:   tls.VersionTLS13,
        CipherSuites: []uint16{
            tls.TLS_AES_256_GCM_SHA384,
            tls.TLS_CHACHA20_POLY1305_SHA256,
        },
    }

    // If mTLS required, configure client certificate validation
    if config.TLSClientAuth {
        caCert, err := os.ReadFile(config.CACertFile)
        if err != nil {
            return nil, fmt.Errorf("CA cert read failed: %w", err)
        }

        caCertPool := x509.NewCertPool()
        if !caCertPool.AppendCertsFromPEM(caCert) {
            return nil, fmt.Errorf("failed to parse CA certificate")
        }

        tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
        tlsConfig.ClientCAs = caCertPool
    }

    return tlsConfig, nil
}
```

### Phase 4: Environment Configuration & Documentation

```bash
# .env.example (for users)

# Network binding - bind to internal IP only
OLLAMA_HOST=10.0.0.5          # internal IP, NOT 0.0.0.0
OLLAMA_PORT=8000

# TLS/mTLS configuration
OLLAMA_TLS=true                # enable TLS
OLLAMA_CERT_FILE=/etc/ollama/server.crt
OLLAMA_KEY_FILE=/etc/ollama/server.key
OLLAMA_MTLS=true               # require client certificates

# Vault configuration (for certificate management)
VAULT_ADDR=https://vault.internal:8200
VAULT_TOKEN=s.xxxxxxxxxxxxx
VAULT_PKI_PATH=pki/issue/ollama
```

## Acceptance Criteria

- ✅ Can bind to specific internal IP (not 0.0.0.0)
- ✅ TLS encryption enabled by configuration
- ✅ Client certificates verified if mTLS enabled
- ✅ Certificates auto-rotated via Vault
- ✅ Certificate rotation transparent (no downtime)
- ✅ Documentation for setup and deployment

## Testing Checklist

```bash
# Test 1: Localhost binding
curl -k https://127.0.0.1:8000/health  # ✓ works
curl -k https://0.0.0.0:8000/health    # ✗ fails (port not listening)

# Test 2: Client certificate required
curl --cert client.crt --key client.key \
     https://localhost:8000/api/models  # ✓ works

curl https://localhost:8000/api/models  # ✗ fails (no client cert)

# Test 3: Certificate rotation
# Vault rotates cert internally
curl --cert newcert.crt --key newkey.key \
     https://localhost:8000/api/models  # ✓ still works
```

## Deployment Checklist

- [ ] Code implementation and testing
- [ ] Integration with Vault PKI
- [ ] Security documentation
- [ ] Deployment guide with certificate setup
- [ ] Automated certificate rotation testing
- [ ] Performance benchmarking (TLS overhead)
- [ ] Documentation for existing users (migration guide)

## Security Benefits

1. **Network Isolation**: Only internal clients can connect
2. **Encryption in Transit**: All communication encrypted with TLS 1.3
3. **Mutual Authentication**: Both server and client prove identity
4. **Key Rotation**: Automatic key rotation without downtime
5. **Audit Trail**: Vault logs all certificate issuance

## References & Resources

- [HashiCorp Vault PKI Secrets](https://www.vaultproject.io/docs/secrets/pki)
- [Go TLS/mTLS Implementation](https://golang.org/pkg/crypto/tls/)
- [NIST Guidelines on TLS](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-52r2.pdf)

---

**Ready for Implementation**: Yes - detailed spec provided, acceptance criteria clear, testing plan included.
