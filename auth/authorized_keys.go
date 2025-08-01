package auth

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

type KeyEntry struct {
	Name      string
	PublicKey string
	Endpoints []string
}

type KeyPermission struct {
	Name      string
	Endpoints []string
}

type APIPermissions struct {
	permissions  map[string]*KeyPermission
	lastModified time.Time
	mutex        sync.RWMutex
}

var ws = regexp.MustCompile(`\s+`)

func authkeyPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", "authorized_keys"), nil
}

func NewAPIPermissions() *APIPermissions {
	return &APIPermissions{
		permissions: make(map[string]*KeyPermission),
		mutex:       sync.RWMutex{},
	}
}

func (ap *APIPermissions) ReloadIfNeeded() error {
	ap.mutex.Lock()
	defer ap.mutex.Unlock()

	filename, err := authkeyPath()
	if err != nil {
		return err
	}

	fileInfo, err := os.Stat(filename)
	if err != nil {
		return fmt.Errorf("failed to stat file: %v", err)
	}

	if !fileInfo.ModTime().After(ap.lastModified) {
		return nil
	}

	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	ap.lastModified = fileInfo.ModTime()
	return ap.parse(file)
}

func (ap *APIPermissions) parse(r io.Reader) error {
	ap.permissions = make(map[string]*KeyPermission)

	scanner := bufio.NewScanner(r)
	var cnt int
	for scanner.Scan() {
		cnt += 1
		line := strings.TrimSpace(scanner.Text())

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		line = ws.ReplaceAllString(line, " ")

		entry, err := ap.parseLine(line)
		if err != nil {
			slog.Warn(fmt.Sprintf("authorized_keys line %d: skipping invalid line: %v\n", cnt, err))
			continue
		}

		var pubKeyStr string

		if entry.PublicKey == "*" {
			pubKeyStr = "*"
		} else {
			pubKey, err := ap.validateAndDecodeKey(entry)
			if err != nil {
				slog.Warn(fmt.Sprintf("authorized_keys line %d: invalid key for %s: %v\n", cnt, entry.Name, err))
				continue
			}
			pubKeyStr = pubKey
		}

		if perm, exists := ap.permissions[pubKeyStr]; exists {
			if perm.Name == "default" {
				perm.Name = entry.Name
			}
			if len(perm.Endpoints) == 1 && perm.Endpoints[0] == "*" {
				// skip redundant entries
				continue
			} else if len(entry.Endpoints) == 1 && entry.Endpoints[0] == "*" {
				// overwrite redundant entries
				perm.Endpoints = entry.Endpoints
			} else {
				perm.Endpoints = append(perm.Endpoints, entry.Endpoints...)
			}
		} else {
			ap.permissions[pubKeyStr] = &KeyPermission{
				Name:      entry.Name,
				Endpoints: entry.Endpoints,
			}
		}
	}

	return scanner.Err()
}

func (ap *APIPermissions) parseLine(line string) (*KeyEntry, error) {
	parts := strings.SplitN(line, " ", 4)
	if len(parts) < 2 {
		return nil, fmt.Errorf("key type and public key not found")
	}

	kind, b64Key := parts[0], parts[1]
	name := "default"
	eps := "*"

	if len(parts) >= 3 && parts[2] != "" {
		if parts[2] != "*" {
			name = parts[2]
		}
	}

	if len(parts) == 4 && parts[3] != "" {
		eps = parts[3]
	}

	if kind != "ssh-ed25519" && kind != "*" {
		return nil, fmt.Errorf("unsupported key type %s", kind)
	}

	if kind == "*" && b64Key != "*" {
		return nil, fmt.Errorf("unsupported key type")
	}

	var endpoints []string
	if eps == "*" {
		endpoints = []string{"*"}
	} else {
		for _, e := range strings.Split(eps, ",") {
			e = strings.TrimSpace(e)
			if e == "" {
				return nil, fmt.Errorf("empty endpoint in list")
			} else if e == "*" {
				endpoints = []string{"*"}
				break
			}
			endpoints = append(endpoints, e)
		}
	}

	return &KeyEntry{
		PublicKey: b64Key,
		Name:      name,
		Endpoints: endpoints,
	}, nil
}

func (ap *APIPermissions) validateAndDecodeKey(entry *KeyEntry) (string, error) {
	keyBlob, err := base64.StdEncoding.DecodeString(entry.PublicKey)
	if err != nil {
		return "", fmt.Errorf("base64 decode: %w", err)
	}
	pub, err := ssh.ParsePublicKey(keyBlob)
	if err != nil {
		return "", fmt.Errorf("parse key: %w", err)
	}
	if pub.Type() != ssh.KeyAlgoED25519 {
		return "", fmt.Errorf("key is not Ed25519")
	}

	return entry.PublicKey, nil
}

func (ap *APIPermissions) Authorize(pubKey ssh.PublicKey, endpoint string) (bool, string, error) {
	if err := ap.ReloadIfNeeded(); err != nil {
		return false, "unknown", err
	}

	ap.mutex.RLock()
	defer ap.mutex.RUnlock()

	if wildcardPerm, exists := ap.permissions["*"]; exists {
		if len(wildcardPerm.Endpoints) == 1 && wildcardPerm.Endpoints[0] == "*" {
			return true, wildcardPerm.Name, nil
		}

		for _, allowedEndpoint := range wildcardPerm.Endpoints {
			if allowedEndpoint == endpoint {
				return true, wildcardPerm.Name, nil
			}
		}
	}

	keyString := string(ssh.MarshalAuthorizedKey(pubKey))
	parts := strings.SplitN(keyString, " ", 2)
	var base64Key string
	if len(parts) > 1 {
		base64Key = parts[1]
	} else {
		base64Key = parts[0]
	}

	base64Key = strings.TrimSpace(base64Key)

	perm, exists := ap.permissions[base64Key]
	if !exists {
		return false, "unknown", nil
	}

	if len(perm.Endpoints) == 1 && perm.Endpoints[0] == "*" {
		return true, perm.Name, nil
	}

	for _, allowedEndpoint := range perm.Endpoints {
		if allowedEndpoint == endpoint {
			return true, perm.Name, nil
		}
	}

	return false, "unknown", nil
}
