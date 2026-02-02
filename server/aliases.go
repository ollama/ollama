package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const (
	routerConfigFilename = "router.json"
	routerConfigVersion  = 1
)

var errAliasCycle = errors.New("alias cycle detected")

type aliasEntry struct {
	Alias  string `json:"alias"`
	Target string `json:"target"`
}

type routerConfig struct {
	Version int          `json:"version"`
	Aliases []aliasEntry `json:"aliases"`
}

type aliasStore struct {
	mu      sync.RWMutex
	path    string
	entries map[string]aliasEntry // normalized alias -> entry
}

func newAliasStore(path string) (*aliasStore, error) {
	store := &aliasStore{
		path:    path,
		entries: make(map[string]aliasEntry),
	}
	if err := store.load(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *aliasStore) load() error {
	data, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}

	var cfg routerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return err
	}

	if cfg.Version != 0 && cfg.Version != routerConfigVersion {
		return fmt.Errorf("unsupported router config version %d", cfg.Version)
	}

	for _, entry := range cfg.Aliases {
		aliasName := model.ParseName(entry.Alias)
		if !aliasName.IsValid() {
			slog.Warn("invalid alias name in router config", "alias", entry.Alias)
			continue
		}
		targetName := model.ParseName(entry.Target)
		if !targetName.IsValid() {
			slog.Warn("invalid alias target in router config", "target", entry.Target)
			continue
		}

		canonicalAlias := aliasName.String()
		canonicalTarget := targetName.String()
		s.entries[normalizeAliasKey(aliasName)] = aliasEntry{
			Alias:  canonicalAlias,
			Target: canonicalTarget,
		}
	}

	return nil
}

func (s *aliasStore) saveLocked() error {
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	entries := make([]aliasEntry, 0, len(s.entries))
	for _, entry := range s.entries {
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool {
		return strings.Compare(entries[i].Alias, entries[j].Alias) < 0
	})

	cfg := routerConfig{
		Version: routerConfigVersion,
		Aliases: entries,
	}

	f, err := os.CreateTemp(dir, "router-*.json")
	if err != nil {
		return err
	}

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cfg); err != nil {
		_ = f.Close()
		_ = os.Remove(f.Name())
		return err
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(f.Name())
		return err
	}

	if err := os.Chmod(f.Name(), 0o644); err != nil {
		_ = os.Remove(f.Name())
		return err
	}

	return os.Rename(f.Name(), s.path)
}

func (s *aliasStore) ResolveName(name model.Name) (model.Name, bool, error) {
	key := normalizeAliasKey(name)

	s.mu.RLock()
	entry, ok := s.entries[key]
	s.mu.RUnlock()
	if !ok {
		return name, false, nil
	}

	// If a local model exists, do not allow alias shadowing.
	exists, err := localModelExists(name)
	if err != nil {
		return name, false, err
	}
	if exists {
		return name, false, nil
	}

	visited := map[string]struct{}{key: {}}
	targetKey := strings.ToLower(entry.Target)
	current := entry.Target

	for {
		targetName := model.ParseName(current)
		if !targetName.IsValid() {
			return name, false, fmt.Errorf("alias target %q is invalid", current)
		}

		if _, seen := visited[targetKey]; seen {
			return name, false, errAliasCycle
		}
		visited[targetKey] = struct{}{}

		s.mu.RLock()
		next, ok := s.entries[targetKey]
		s.mu.RUnlock()
		if !ok {
			return targetName, true, nil
		}

		current = next.Target
		targetKey = strings.ToLower(current)
	}
}

func (s *aliasStore) Set(alias, target model.Name) error {
	aliasKey := normalizeAliasKey(alias)
	targetKey := normalizeAliasKey(target)

	if aliasKey == targetKey {
		return fmt.Errorf("alias cannot point to itself")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	visited := map[string]struct{}{aliasKey: {}}
	currentKey := targetKey
	for {
		if _, seen := visited[currentKey]; seen {
			return errAliasCycle
		}
		visited[currentKey] = struct{}{}

		next, ok := s.entries[currentKey]
		if !ok {
			break
		}
		currentKey = strings.ToLower(next.Target)
	}

	s.entries[aliasKey] = aliasEntry{
		Alias:  alias.String(),
		Target: target.String(),
	}

	return s.saveLocked()
}

func (s *aliasStore) Delete(alias model.Name) (bool, error) {
	aliasKey := normalizeAliasKey(alias)

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.entries[aliasKey]; !ok {
		return false, nil
	}

	delete(s.entries, aliasKey)
	return true, s.saveLocked()
}

func (s *aliasStore) List() []aliasEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries := make([]aliasEntry, 0, len(s.entries))
	for _, entry := range s.entries {
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool {
		return strings.Compare(entries[i].Alias, entries[j].Alias) < 0
	})
	return entries
}

func normalizeAliasKey(name model.Name) string {
	return strings.ToLower(name.String())
}

func localModelExists(name model.Name) (bool, error) {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return false, err
	}
	needle := name.String()
	for existing := range manifests {
		if strings.EqualFold(existing.String(), needle) {
			return true, nil
		}
	}
	return false, nil
}

func routerConfigPath() string {
	return filepath.Join(envconfig.Models(), routerConfigFilename)
}

func (s *Server) aliasStore() (*aliasStore, error) {
	s.aliasesOnce.Do(func() {
		s.aliases, s.aliasesErr = newAliasStore(routerConfigPath())
	})

	return s.aliases, s.aliasesErr
}

func (s *Server) resolveModelAliasName(name model.Name) (model.Name, bool, error) {
	store, err := s.aliasStore()
	if err != nil {
		return name, false, err
	}

	if store == nil {
		return name, false, nil
	}

	return store.ResolveName(name)
}
