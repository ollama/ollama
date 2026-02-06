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

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const (
	serverConfigFilename = "server.json"
	serverConfigVersion  = 1
)

var errAliasCycle = errors.New("alias cycle detected")

type aliasEntry struct {
	Alias          string `json:"alias"`
	Target         string `json:"target"`
	PrefixMatching bool   `json:"prefix_matching,omitempty"`
}

type serverConfig struct {
	Version int          `json:"version"`
	Aliases []aliasEntry `json:"aliases"`
}

type store struct {
	mu            sync.RWMutex
	path          string
	entries       map[string]aliasEntry // normalized alias -> entry (exact matches)
	prefixEntries []aliasEntry          // prefix matches, sorted longest-first
}

func createStore(path string) (*store, error) {
	store := &store{
		path:    path,
		entries: make(map[string]aliasEntry),
	}
	if err := store.load(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *store) load() error {
	data, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}

	var cfg serverConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return err
	}

	if cfg.Version != 0 && cfg.Version != serverConfigVersion {
		return fmt.Errorf("unsupported router config version %d", cfg.Version)
	}

	for _, entry := range cfg.Aliases {
		targetName := model.ParseName(entry.Target)
		if !targetName.IsValid() {
			slog.Warn("invalid alias target in router config", "target", entry.Target)
			continue
		}
		canonicalTarget := displayAliasName(targetName)

		if entry.PrefixMatching {
			// Prefix aliases don't need to be valid model names
			alias := strings.TrimSpace(entry.Alias)
			if alias == "" {
				slog.Warn("empty prefix alias in router config")
				continue
			}
			s.prefixEntries = append(s.prefixEntries, aliasEntry{
				Alias:          alias,
				Target:         canonicalTarget,
				PrefixMatching: true,
			})
		} else {
			aliasName := model.ParseName(entry.Alias)
			if !aliasName.IsValid() {
				slog.Warn("invalid alias name in router config", "alias", entry.Alias)
				continue
			}
			canonicalAlias := displayAliasName(aliasName)
			s.entries[normalizeAliasKey(aliasName)] = aliasEntry{
				Alias:  canonicalAlias,
				Target: canonicalTarget,
			}
		}
	}

	// Sort prefix entries by alias length descending (longest prefix wins)
	s.sortPrefixEntriesLocked()

	return nil
}

func (s *store) saveLocked() error {
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	// Combine exact and prefix entries
	entries := make([]aliasEntry, 0, len(s.entries)+len(s.prefixEntries))
	for _, entry := range s.entries {
		entries = append(entries, entry)
	}
	entries = append(entries, s.prefixEntries...)

	sort.Slice(entries, func(i, j int) bool {
		return strings.Compare(entries[i].Alias, entries[j].Alias) < 0
	})

	cfg := serverConfig{
		Version: serverConfigVersion,
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

func (s *store) ResolveName(name model.Name) (model.Name, bool, error) {
	// If a local model exists, do not allow alias shadowing (highest priority).
	exists, err := localModelExists(name)
	if err != nil {
		return name, false, err
	}
	if exists {
		return name, false, nil
	}

	key := normalizeAliasKey(name)

	s.mu.RLock()
	entry, exactMatch := s.entries[key]
	var prefixMatch *aliasEntry
	if !exactMatch {
		// Try prefix matching - prefixEntries is sorted longest-first
		nameStr := strings.ToLower(displayAliasName(name))
		for i := range s.prefixEntries {
			prefix := strings.ToLower(s.prefixEntries[i].Alias)
			if strings.HasPrefix(nameStr, prefix) {
				prefixMatch = &s.prefixEntries[i]
				break // First match is longest due to sorting
			}
		}
	}
	s.mu.RUnlock()

	if !exactMatch && prefixMatch == nil {
		return name, false, nil
	}

	var current string
	var visited map[string]struct{}

	if exactMatch {
		visited = map[string]struct{}{key: {}}
		current = entry.Target
	} else {
		// For prefix match, use the target as-is
		visited = map[string]struct{}{}
		current = prefixMatch.Target
	}

	targetKey := normalizeAliasKeyString(current)

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
		targetKey = normalizeAliasKeyString(current)
	}
}

func (s *store) Set(alias, target model.Name, prefixMatching bool) error {
	targetKey := normalizeAliasKey(target)

	s.mu.Lock()
	defer s.mu.Unlock()

	if prefixMatching {
		// For prefix aliases, we skip cycle detection since prefix matching
		// works differently and the target is a specific model
		aliasStr := displayAliasName(alias)

		// Remove any existing prefix entry with the same alias
		for i, e := range s.prefixEntries {
			if strings.EqualFold(e.Alias, aliasStr) {
				s.prefixEntries = append(s.prefixEntries[:i], s.prefixEntries[i+1:]...)
				break
			}
		}

		s.prefixEntries = append(s.prefixEntries, aliasEntry{
			Alias:          aliasStr,
			Target:         displayAliasName(target),
			PrefixMatching: true,
		})
		s.sortPrefixEntriesLocked()
		return s.saveLocked()
	}

	aliasKey := normalizeAliasKey(alias)

	if aliasKey == targetKey {
		return fmt.Errorf("alias cannot point to itself")
	}

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
		currentKey = normalizeAliasKeyString(next.Target)
	}

	s.entries[aliasKey] = aliasEntry{
		Alias:  displayAliasName(alias),
		Target: displayAliasName(target),
	}

	return s.saveLocked()
}

func (s *store) Delete(alias model.Name) (bool, error) {
	aliasKey := normalizeAliasKey(alias)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Try exact match first
	if _, ok := s.entries[aliasKey]; ok {
		delete(s.entries, aliasKey)
		return true, s.saveLocked()
	}

	// Try prefix entries
	aliasStr := displayAliasName(alias)
	for i, e := range s.prefixEntries {
		if strings.EqualFold(e.Alias, aliasStr) {
			s.prefixEntries = append(s.prefixEntries[:i], s.prefixEntries[i+1:]...)
			return true, s.saveLocked()
		}
	}

	return false, nil
}

// DeleteByString deletes an alias by its raw string value, useful for prefix
// aliases that may not be valid model names.
func (s *store) DeleteByString(alias string) (bool, error) {
	alias = strings.TrimSpace(alias)
	aliasLower := strings.ToLower(alias)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Try prefix entries first (since this is mainly for prefix aliases)
	for i, e := range s.prefixEntries {
		if strings.EqualFold(e.Alias, alias) {
			s.prefixEntries = append(s.prefixEntries[:i], s.prefixEntries[i+1:]...)
			return true, s.saveLocked()
		}
	}

	// Also check exact entries by normalized key
	if _, ok := s.entries[aliasLower]; ok {
		delete(s.entries, aliasLower)
		return true, s.saveLocked()
	}

	return false, nil
}

func (s *store) List() []aliasEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries := make([]aliasEntry, 0, len(s.entries)+len(s.prefixEntries))
	for _, entry := range s.entries {
		entries = append(entries, entry)
	}
	entries = append(entries, s.prefixEntries...)

	sort.Slice(entries, func(i, j int) bool {
		return strings.Compare(entries[i].Alias, entries[j].Alias) < 0
	})
	return entries
}

func normalizeAliasKey(name model.Name) string {
	return strings.ToLower(displayAliasName(name))
}

func (s *store) sortPrefixEntriesLocked() {
	sort.Slice(s.prefixEntries, func(i, j int) bool {
		// Sort by length descending (longest prefix first)
		return len(s.prefixEntries[i].Alias) > len(s.prefixEntries[j].Alias)
	})
}

func normalizeAliasKeyString(value string) string {
	n := model.ParseName(value)
	if !n.IsValid() {
		return strings.ToLower(strings.TrimSpace(value))
	}
	return normalizeAliasKey(n)
}

func displayAliasName(n model.Name) string {
	display := n.DisplayShortest()
	if strings.EqualFold(n.Tag, "latest") {
		if idx := strings.LastIndex(display, ":"); idx != -1 {
			return display[:idx]
		}
	}
	return display
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

func serverConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".ollama", serverConfigFilename)
	}
	return filepath.Join(home, ".ollama", serverConfigFilename)
}

func (s *Server) aliasStore() (*store, error) {
	s.aliasesOnce.Do(func() {
		s.aliases, s.aliasesErr = createStore(serverConfigPath())
	})

	return s.aliases, s.aliasesErr
}

func (s *Server) resolveAlias(name model.Name) (model.Name, bool, error) {
	store, err := s.aliasStore()
	if err != nil {
		return name, false, err
	}

	if store == nil {
		return name, false, nil
	}

	return store.ResolveName(name)
}
