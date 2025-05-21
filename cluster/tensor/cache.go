package tensor

import (
	"container/list"
	"fmt"
	"sync"
	"time"
)

// CacheKey uniquely identifies a tensor in the cache
type CacheKey struct {
	// ModelID identifies which model this tensor belongs to
	ModelID string
	
	// PartitionID identifies the partition
	PartitionID string
	
	// TensorID identifies the specific tensor
	TensorID string
}

// String returns a string representation of the cache key
func (k CacheKey) String() string {
	return fmt.Sprintf("%s/%s/%s", k.ModelID, k.PartitionID, k.TensorID)
}

// CacheEntry represents a tensor stored in the cache
type CacheEntry struct {
	// Key uniquely identifies this entry
	Key CacheKey
	
	// Data is the actual tensor data
	Data []byte
	
	// CompressionType indicates how the tensor is compressed
	CompressionType CompressionType
	
	// Size is the memory footprint of this entry in bytes
	Size uint64
	
	// LastAccess is when this entry was last accessed
	LastAccess time.Time
	
	// ExpireAt is when this entry should expire (zero for no expiration)
	ExpireAt time.Time
	
	// HitCount tracks how many times this entry was accessed
	HitCount int
}

// CacheOptions configures the tensor cache
type CacheOptions struct {
	// MaxEntries limits the number of cached tensors
	MaxEntries int
	
	// MaxMemory limits the total memory usage in bytes
	MaxMemory uint64
	
	// DefaultTTL is the default time-to-live for entries
	DefaultTTL time.Duration
	
	// EvictionPolicy determines how entries are removed when full
	EvictionPolicy string
	
	// EnableStats enables tracking of cache statistics
	EnableStats bool
}

// DefaultCacheOptions provides sensible defaults
var DefaultCacheOptions = CacheOptions{
	MaxEntries:     1000,
	MaxMemory:      1024 * 1024 * 1024, // 1 GB
	DefaultTTL:     10 * time.Minute,
	EvictionPolicy: "lru",
	EnableStats:    true,
}

// CacheStats tracks cache performance metrics
type CacheStats struct {
	// Hits is the number of cache hits
	Hits int
	
	// Misses is the number of cache misses
	Misses int
	
	// Evictions is the number of entries evicted
	Evictions int
	
	// Expirations is the number of entries expired
	Expirations int
	
	// CurrentEntries is the current number of entries
	CurrentEntries int
	
	// CurrentMemoryBytes is the current memory usage in bytes
	CurrentMemoryBytes uint64
}

// TensorCache caches tensors to avoid redundant transfers
type TensorCache struct {
	// options stores the cache configuration
	options CacheOptions
	
	// entries maps keys to cache entries
	entries map[CacheKey]*list.Element
	
	// lruList maintains entries ordered by access time
	lruList *list.List
	
	// stats tracks cache performance
	stats CacheStats
	
	// currentMemory tracks the total memory usage
	currentMemory uint64
	
	// mu protects the cache from concurrent access
	mu sync.RWMutex
	
	// compressor is used to compress/decompress cached tensors
	compressor *Compressor
}

// NewTensorCache creates a new tensor cache
func NewTensorCache(options CacheOptions) *TensorCache {
	cache := &TensorCache{
		options:     options,
		entries:     make(map[CacheKey]*list.Element),
		lruList:     list.New(),
		compressor:  NewCompressor(DefaultCompressionConfig()),
	}
	
	// Start a background goroutine for periodic cleanup
	go cache.cleanupLoop()
	
	return cache
}
// cleanupLoop periodically checks for and removes expired entries
func (c *TensorCache) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		c.removeExpiredEntries()
	}
}

// Get retrieves a tensor from the cache
func (c *TensorCache) Get(key CacheKey) ([]byte, CompressionType, bool) {
	c.mu.RLock()
	elem, found := c.entries[key]
	
	if !found {
		c.mu.RUnlock()
		if c.options.EnableStats {
			c.mu.Lock()
			c.stats.Misses++
			c.mu.Unlock()
		}
		return nil, CompressionNone, false
	}
	
	// Move to front of LRU list under write lock
	c.mu.RUnlock()
	c.mu.Lock()
	
	// Check if entry still exists (could have been removed between locks)
	elem, found = c.entries[key]
	if !found {
		c.mu.Unlock()
		if c.options.EnableStats {
			c.mu.Lock()
			c.stats.Misses++
			c.mu.Unlock()
		}
		return nil, CompressionNone, false
	}
	
	entry := elem.Value.(*CacheEntry)
	
	// Check if entry has expired
	if !entry.ExpireAt.IsZero() && entry.ExpireAt.Before(time.Now()) {
		// Remove expired entry
		c.removeEntry(elem)
		c.mu.Unlock()
		
		if c.options.EnableStats {
			c.mu.Lock()
			c.stats.Misses++
			c.stats.Expirations++
			c.mu.Unlock()
		}
		
		return nil, CompressionNone, false
	}
	
	// Update last access time and move to front
	entry.LastAccess = time.Now()
	entry.HitCount++
	c.lruList.MoveToFront(elem)
	
	data := entry.Data
	compressionType := entry.CompressionType
	
	c.mu.Unlock()
	
	if c.options.EnableStats {
		c.mu.Lock()
		c.stats.Hits++
		c.mu.Unlock()
	}
	
	return data, compressionType, true
}

// Put adds or updates a tensor in the cache
func (c *TensorCache) Put(key CacheKey, data []byte, compressionType CompressionType, ttl time.Duration) {
	if len(data) == 0 {
		return
	}
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// If entry already exists, update it
	if elem, found := c.entries[key]; found {
		entry := elem.Value.(*CacheEntry)
		
		// Update memory tracking
		c.currentMemory -= entry.Size
		
		// Update entry
		entry.Data = data
		entry.Size = uint64(len(data))
		entry.LastAccess = time.Now()
		entry.CompressionType = compressionType
		
		if ttl > 0 {
			entry.ExpireAt = time.Now().Add(ttl)
		} else if c.options.DefaultTTL > 0 {
			entry.ExpireAt = time.Now().Add(c.options.DefaultTTL)
		}
		
		// Move to front of LRU list
		c.lruList.MoveToFront(elem)
		
		// Update memory tracking
		c.currentMemory += entry.Size
		
		// Update stats
		if c.options.EnableStats {
			c.stats.CurrentMemoryBytes = c.currentMemory
		}
		
		return
	}
	
	// Create new entry
	entry := &CacheEntry{
		Key:             key,
		Data:            data,
		Size:            uint64(len(data)),
		LastAccess:      time.Now(),
		CompressionType: compressionType,
		HitCount:        0,
	}
	
	if ttl > 0 {
		entry.ExpireAt = time.Now().Add(ttl)
	} else if c.options.DefaultTTL > 0 {
		entry.ExpireAt = time.Now().Add(c.options.DefaultTTL)
	}
	
	// Add to LRU list and map
	elem := c.lruList.PushFront(entry)
	c.entries[key] = elem
	
	// Update memory tracking
	c.currentMemory += entry.Size
	
	// Update stats
	if c.options.EnableStats {
		c.stats.CurrentEntries = len(c.entries)
		c.stats.CurrentMemoryBytes = c.currentMemory
	}
	
	// Check if we need to evict entries
	c.evictIfNeeded()
}

// Remove removes an entry from the cache
func (c *TensorCache) Remove(key CacheKey) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	elem, found := c.entries[key]
	if !found {
		return false
	}
	
	c.removeEntry(elem)
	return true
}

// removeEntry removes a list element from the cache
func (c *TensorCache) removeEntry(elem *list.Element) {
	entry := elem.Value.(*CacheEntry)
	
	delete(c.entries, entry.Key)
	c.lruList.Remove(elem)
	c.currentMemory -= entry.Size
	
	// Update stats
	if c.options.EnableStats {
		c.stats.CurrentEntries = len(c.entries)
		c.stats.CurrentMemoryBytes = c.currentMemory
	}
}

// removeExpiredEntries removes all expired entries from the cache
func (c *TensorCache) removeExpiredEntries() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	now := time.Now()
	var nextElem *list.Element
	
	for elem := c.lruList.Back(); elem != nil; elem = nextElem {
		nextElem = elem.Prev()
		entry := elem.Value.(*CacheEntry)
		
		if !entry.ExpireAt.IsZero() && entry.ExpireAt.Before(now) {
			c.removeEntry(elem)
			
			// Update stats
			if c.options.EnableStats {
				c.stats.Expirations++
			}
		}
	}
}

// evictIfNeeded removes entries if cache exceeds size limits
func (c *TensorCache) evictIfNeeded() {
	// Check if we're under limits
	if (c.options.MaxEntries <= 0 || len(c.entries) <= c.options.MaxEntries) &&
	   (c.options.MaxMemory <= 0 || c.currentMemory <= c.options.MaxMemory) {
		return
	}
	
	// Evict entries until we're under limits
	for c.lruList.Len() > 0 &&
	   ((c.options.MaxEntries > 0 && len(c.entries) > c.options.MaxEntries) ||
		(c.options.MaxMemory > 0 && c.currentMemory > c.options.MaxMemory)) {
		
		// Remove entry from back of list (least recently used)
		elem := c.lruList.Back()
		if elem == nil {
			break
		}
		
		c.removeEntry(elem)
		
		// Update eviction stats
		if c.options.EnableStats {
			c.stats.Evictions++
		}
	}
}

// Clear empties the cache
func (c *TensorCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.entries = make(map[CacheKey]*list.Element)
	c.lruList.Init()
	c.currentMemory = 0
	
	// Reset stats
	if c.options.EnableStats {
		c.stats.CurrentEntries = 0
		c.stats.CurrentMemoryBytes = 0
	}
}

// GetStats returns cache statistics
func (c *TensorCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return c.stats
}

// GetOptions returns the current cache options
func (c *TensorCache) GetOptions() CacheOptions {
	return c.options
}

// SetOptions updates the cache options
func (c *TensorCache) SetOptions(options CacheOptions) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.options = options
	
	// Check if we need to evict entries based on new limits
	c.evictIfNeeded()
}