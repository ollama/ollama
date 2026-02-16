//go:build mlx

package mlxrunner

import (
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

type CacheEntry struct {
	Caches  []cache.Cache
	Count   int
	Entries map[int32]*CacheEntry
}

func (s Runner) FindNearestCache(tokens []int32) ([]cache.Cache, []int32) {
	current := &CacheEntry{Entries: s.CacheEntries}
	index, cacheIndex := 0, -1
	for _, token := range tokens {
		if _, ok := current.Entries[token]; !ok {
			break
		}

		current = current.Entries[token]
		if len(current.Caches) > 0 {
			cacheIndex = index
		}

		index += 1
	}

	if cacheIndex == len(tokens)-1 {
		slog.Info("Cache hit", "type", "exact", "total", len(tokens), "cached", len(tokens), "left", len(tokens))
		return current.Caches, []int32{}
	} else if cacheIndex > 1 {
		slog.Info("Cache hit", "type", "partial", "total", len(tokens), "cached", cacheIndex+1, "left", len(tokens[cacheIndex+1:]))
		return current.Caches, tokens[cacheIndex+1:]
	} else if index > 0 && cacheIndex < 0 {
		type stackItem struct {
			entry  *CacheEntry
			tokens []int32
		}

		var best, item stackItem
		stack := []stackItem{{entry: current, tokens: []int32{}}}
		for len(stack) > 0 {
			item, stack = stack[len(stack)-1], stack[:len(stack)-1]
			if len(item.entry.Caches) > 0 {
				if len(best.tokens) == 0 || len(item.tokens) < len(best.tokens) {
					best = item
				}
			} else {
				for token, entry := range item.entry.Entries {
					stack = append(stack, stackItem{
						entry:  entry,
						tokens: append(item.tokens, token),
					})
				}
			}
		}

		prefix := min(len(tokens)-1, index)
		caches := make([]cache.Cache, len(best.entry.Caches))
		trim := len(best.tokens)+1
		for i := range caches {
			caches[i] = best.entry.Caches[i].Clone()
			caches[i].Trim(trim)
		}

		slog.Info("Cache hit", "type", "prefix", "total", len(tokens), "cached", prefix, "left", len(tokens[prefix:]), "trimmed", trim)
		return caches, tokens[prefix:]
	}

	slog.Info("Cache miss", "left", len(tokens))
	return nil, tokens
}

func (s *Runner) InsertCache(tokens []int32, caches []cache.Cache) {
	current := &CacheEntry{Entries: s.CacheEntries}
	for _, token := range tokens {
		if _, ok := current.Entries[token]; !ok {
			current.Entries[token] = &CacheEntry{
				Entries: make(map[int32]*CacheEntry),
			}
		}

		current = current.Entries[token]
	}

	if len(current.Caches) > 0 {
		current.Count += 1
	} else {
		current.Caches = caches
	}
}
