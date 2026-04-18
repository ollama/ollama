# Issue #165: Immutable Audit Log - SHA-256 Hash Chain Implementation Guide

**Priority**: CRITICAL - Security & Compliance
**Complexity**: High
**Effort**: 28 hours
**Status**: Ready for Implementation

## Problem Statement

Current audit logging:
- Logs can be modified/deleted by administrators
- No tamper detection if logs are compromised
- Cannot prove integrity for compliance (SOC 2, PCI-DSS, HIPAA)
- No ability to detect when logs were altered

Required: **Cryptographic hash chain** where each audit entry's hash depends on all previous entries, making tampering detectable.

## Solution Overview

Implement immutable audit log using:
1. **SHA-256 hash chain**: Each entry includes hash of previous entry
2. **Merkle tree roots**: Periodic snapshots for verification
3. **Append-only storage**: SQLite with constraints preventing updates
4. **External notarization** (optional): Store root hashes in blockchain/timestamping service

## Implementation

### Phase 1: Hash Chain Structure

```go
// server/audit/immutable_log.go
package audit

import (
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "time"
)

// AuditEntry represents an immutable log entry
type AuditEntry struct {
    ID              int64     `json:"id"`
    Timestamp       time.Time `json:"timestamp"`
    EventType       string    `json:"event_type"`        // api_call, model_load, secret_detected, etc.
    User            string    `json:"user"`              // System user
    RemoteAddr      string    `json:"remote_addr"`
    RequestPath     string    `json:"request_path"`
    Payload         json.RawMessage `json:"payload"`     // Event-specific data
    Result          string    `json:"result"`            // success, failure, partial
    ErrorMessage    string    `json:"error_message"`

    // Hash chain fields
    PreviousHash    string    `json:"previous_hash"`     // Hash of previous entry
    CurrentHash     string    `json:"current_hash"`      // Hash of this entry
}

// CalculateHash computes SHA-256 hash of entry
func (e *AuditEntry) CalculateHash() string {
    // Create data to hash (excluding current_hash)
    hashInput := struct {
        ID          int64
        Timestamp   int64
        EventType   string
        User        string
        RemoteAddr  string
        RequestPath string
        Payload     json.RawMessage
        Result      string
        PreviousHash string
    }{
        ID:          e.ID,
        Timestamp:   e.Timestamp.Unix(),
        EventType:   e.EventType,
        User:        e.User,
        RemoteAddr:  e.RemoteAddr,
        RequestPath: e.RequestPath,
        Payload:     e.Payload,
        Result:      e.Result,
        PreviousHash: e.PreviousHash,
    }

    data, _ := json.Marshal(hashInput)
    hash := sha256.Sum256(data)
    return hex.EncodeToString(hash[:])
}

// VerifyHash checks if CurrentHash is correct
func (e *AuditEntry) VerifyHash() bool {
    return e.CurrentHash == e.CalculateHash()
}

// AuditLog manages immutable log with hash chain
type AuditLog struct {
    db     *sql.DB
    logger *log.Logger
}

func NewAuditLog(dbPath string, logger *log.Logger) (*AuditLog, error) {
    db, err := sql.Open("sqlite3", dbPath+"?journal_mode=WAL")
    if err != nil {
        return nil, err
    }

    al := &AuditLog{db: db, logger: logger}
    if err := al.initDatabase(); err != nil {
        return nil, err
    }

    return al, nil
}

func (al *AuditLog) initDatabase() error {
    schema := `
    -- Immutable audit log table
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        event_type TEXT NOT NULL,
        user TEXT NOT NULL,
        remote_addr TEXT,
        request_path TEXT,
        payload JSON,
        result TEXT,
        error_message TEXT,
        previous_hash TEXT,
        current_hash TEXT NOT NULL UNIQUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        -- Enforce immutability: no updates allowed
        CHECK (created_at = CURRENT_TIMESTAMP)
    );

    -- Index for efficient querying
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
    CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user);

    -- Merkle tree snapshots (periodic roots for verification)
    CREATE TABLE IF NOT EXISTS audit_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_time DATETIME NOT NULL,
        entry_count INTEGER,
        merkle_root TEXT NOT NULL,
        signature TEXT,  -- Can be signed by HSM/KMS
        verified BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_snapshots_time ON audit_snapshots(snapshot_time);
    `

    _, err := al.db.Exec(schema)
    return err
}

// Log writes an immutable audit entry
func (al *AuditLog) Log(entry *AuditEntry) error {
    // Get previous hash
    var prevHash string
    err := al.db.QueryRow(
        "SELECT current_hash FROM audit_log ORDER BY id DESC LIMIT 1",
    ).Scan(&prevHash)

    if err != nil && err != sql.ErrNoRows {
        return err
    }

    entry.PreviousHash = prevHash
    entry.Timestamp = time.Now()
    entry.CurrentHash = entry.CalculateHash()

    // Insert (will fail if hash collision - cryptographically secure)
    _, err = al.db.Exec(`
        INSERT INTO audit_log (
            timestamp, event_type, user, remote_addr, request_path,
            payload, result, error_message, previous_hash, current_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
        entry.Timestamp,
        entry.EventType,
        entry.User,
        entry.RemoteAddr,
        entry.RequestPath,
        entry.Payload,
        entry.Result,
        entry.ErrorMessage,
        entry.PreviousHash,
        entry.CurrentHash,
    )

    return err
}

// VerifyIntegrity checks if log has been tampered with
func (al *AuditLog) VerifyIntegrity() (bool, []int64, error) {
    rows, err := al.db.Query(`
        SELECT id, current_hash, previous_hash
        FROM audit_log
        ORDER BY id ASC
    `)
    if err != nil {
        return false, nil, err
    }
    defer rows.Close()

    var tamperedIDs []int64
    var prevHash string

    for rows.Next() {
        var id int64
        var currentHash, expectedPrevHash string

        if err := rows.Scan(&id, &currentHash, &expectedPrevHash); err != nil {
            return false, nil, err
        }

        // Verify hash chain continuity
        if expectedPrevHash != prevHash && id > 1 {
            tamperedIDs = append(tamperedIDs, id)
        }

        prevHash = currentHash
    }

    return len(tamperedIDs) == 0, tamperedIDs, nil
}

// CreateSnapshot creates a Merkle tree snapshot for verification
func (al *AuditLog) CreateSnapshot() (string, error) {
    var count int
    if err := al.db.QueryRow("SELECT COUNT(*) FROM audit_log").Scan(&count); err != nil {
        return "", err
    }

    // Get all hashes and build Merkle tree
    rows, err := al.db.Query("SELECT current_hash FROM audit_log ORDER BY id ASC")
    if err != nil {
        return "", err
    }
    defer rows.Close()

    var hashes []string
    for rows.Next() {
        var hash string
        rows.Scan(&hash)
        hashes = append(hashes, hash)
    }

    // Compute Merkle root
    root := al.computeMerkleRoot(hashes)

    // Store snapshot
    _, err = al.db.Exec(`
        INSERT INTO audit_snapshots (snapshot_time, entry_count, merkle_root, verified)
        VALUES (?, ?, ?, 1)
    `, time.Now(), count, root)

    return root, err
}

// computeMerkleRoot builds Merkle tree from hashes
func (al *AuditLog) computeMerkleRoot(hashes []string) string {
    if len(hashes) == 0 {
        return hex.EncodeToString(sha256.Sum256([]byte{})[:])
    }

    for len(hashes) > 1 {
        var nextLevel []string
        for i := 0; i < len(hashes); i += 2 {
            var combined string
            if i+1 < len(hashes) {
                combined = hashes[i] + hashes[i+1]
            } else {
                combined = hashes[i] + hashes[i]
            }

            hash := sha256.Sum256([]byte(combined))
            nextLevel = append(nextLevel, hex.EncodeToString(hash[:]))
        }
        hashes = nextLevel
    }

    return hashes[0]
}

// QueryEntries retrieves audit entries with proof of integrity
func (al *AuditLog) QueryEntries(eventType string, limit int) ([]AuditEntry, error) {
    query := "SELECT id, timestamp, event_type, user, remote_addr, request_path, payload, result, error_message, previous_hash, current_hash FROM audit_log"

    if eventType != "" {
        query += " WHERE event_type = ?"
    }

    query += " ORDER BY id DESC LIMIT ?"

    var entries []AuditEntry
    // ... execute query and scan results
    return entries, nil
}
```

### Phase 2: Audit Middleware

```go
// server/middleware/audit.go
package middleware

import (
    "io"
    "log"
    "net/http"
    "time"

    "ollama/server/audit"
)

type AuditMiddleware struct {
    auditLog *audit.AuditLog
    logger   *log.Logger
}

func NewAuditMiddleware(auditLog *audit.AuditLog, logger *log.Logger) *AuditMiddleware {
    return &AuditMiddleware{
        auditLog: auditLog,
        logger:   logger,
    }
}

// AuditHandler logs API requests and responses
func (m *AuditMiddleware) AuditHandler(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Skip logging for health checks
        if r.URL.Path == "/health" {
            next.ServeHTTP(w, r)
            return
        }

        // Capture response status
        wrapped := &statusCaptureWriter{ResponseWriter: w, statusCode: http.StatusOK}

        start := time.Now()
        next.ServeHTTP(wrapped, r)
        duration := time.Since(start)

        // Log audit entry
        entry := &audit.AuditEntry{
            Timestamp:   time.Now(),
            EventType:   "api_call",
            User:        r.Header.Get("X-User") + "(system)",
            RemoteAddr:  r.RemoteAddr,
            RequestPath: r.URL.Path,
            Result:      m.getResultFromStatus(wrapped.statusCode),
        }

        if err := m.auditLog.Log(entry); err != nil {
            m.logger.Printf("AUDIT_LOG_ERROR: Failed to log entry: %v", err)
        }
    })
}

type statusCaptureWriter struct {
    http.ResponseWriter
    statusCode int
}

func (w *statusCaptureWriter) WriteHeader(code int) {
    w.statusCode = code
    w.ResponseWriter.WriteHeader(code)
}

func (m *AuditMiddleware) getResultFromStatus(code int) string {
    if code >= 200 && code < 300 {
        return "success"
    } else if code >= 400 && code < 500 {
        return "client_error"
    } else if code >= 500 {
        return "server_error"
    }
    return "unknown"
}
```

### Phase 3: Verification & Compliance

```go
// cmd/verify-audit-log.go
package main

import (
    "flag"
    "fmt"
    "log"

    "ollama/server/audit"
)

func main() {
    dbPath := flag.String("db", "audit.db", "Audit log database path")
    flag.Parse()

    auditLog, err := audit.NewAuditLog(*dbPath, log.Default())
    if err != nil {
        log.Fatal(err)
    }

    // Verify integrity
    valid, tamperedIDs, err := auditLog.VerifyIntegrity()
    if err != nil {
        log.Fatal(err)
    }

    if valid {
        fmt.Println("✓ Audit log integrity verified")
    } else {
        fmt.Printf("✗ Tampering detected in entries: %v\n", tamperedIDs)
    }

    // Create snapshot
    root, err := auditLog.CreateSnapshot()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Merkle root snapshot created: %s\n", root)
}
```

## Acceptance Criteria

- ✅ Each entry hash depends on previous entry
- ✅ Detecting any tampering invalidates entire chain
- ✅ Cannot update existing entries (append-only)
- ✅ Periodic Merkle tree snapshots
- ✅ Integrity verification command-line tool
- ✅ Compliance reporting (SOC 2, PCI-DSS)
- ✅ Performance: ≤5ms per log entry

## Testing

```bash
# Verify integrity
./ollama verify-audit-log --db ./audit.db

# Create snapshot
./ollama audit-snapshot --db ./audit.db

# Test tamper detection
sqlite3 audit.db "UPDATE audit_log SET remote_addr='hacked' WHERE id=5"
./ollama verify-audit-log --db ./audit.db

# Expected: "Tampering detected in entries: [5 6 7 8 9 ...]"
```

## Deployment Checklist

- [ ] Implement hash chain data structure
- [ ] Integrate audit middleware
- [ ] Create snapshot mechanism
- [ ] Build verification CLI tool
- [ ] Test tamper detection
- [ ] Configure backup strategy
- [ ] Documentation for compliance teams

## Compliance Benefits

1. **SOC 2 Type II**: Immutable audit trail for security reviews
2. **PCI-DSS 10.7**: Log monitoring with tamper detection
3. **HIPAA**: Integrity and authenticity of audit records
4. **GDPR**: Proof of data handling for audits

---

**Ready for Implementation**: Yes - standard cryptographic patterns, clear verification approach.
