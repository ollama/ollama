package syncmap

import (
	"maps"
	"sync"
)

// SyncMap is a simple, generic thread-safe map implementation.
type SyncMap[K comparable, V any] struct {
	mu sync.RWMutex
	m  map[K]V
}

func NewSyncMap[K comparable, V any]() *SyncMap[K, V] {
	return &SyncMap[K, V]{
		m: make(map[K]V),
	}
}

func (s *SyncMap[K, V]) Load(key K) (V, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.m[key]
	return val, ok
}

func (s *SyncMap[K, V]) Store(key K, value V) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = value
}

func (s *SyncMap[K, V]) Items() map[K]V {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// shallow copy map items
	return maps.Clone(s.m)
}
