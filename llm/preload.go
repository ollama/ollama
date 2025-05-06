package llm

import (
	"log/slog"
	"runtime"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// ModelPreloader provides functionality to preload model memory pages
// and keep them warm to prevent page faults during inference
type ModelPreloader struct {
	// Protects access to the preload state
	mu sync.Mutex

	// Flag indicating if preloading is active
	active bool

	// Stop signal for the background preloader
	stop chan struct{}

	// Size of the model in bytes
	modelSize uint64

	// Last time the model was accessed
	lastAccess time.Time

	// Model memory buffer - we keep a reference to prevent GC
	modelBuffer []byte

	// Name of the model being preloaded
	modelName string
}

// NewModelPreloader creates a new model preloader
func NewModelPreloader(modelPath string, modelSize uint64) *ModelPreloader {
	return &ModelPreloader{
		modelSize: modelSize,
		stop:      make(chan struct{}),
		lastAccess: time.Now(),
		modelName: modelPath,
	}
}

// StartPreloading begins the memory preloading process
// It reads through the model memory to ensure pages are in RAM
func (p *ModelPreloader) StartPreloading(buffer []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.active {
		// Already preloading
		return
	}

	p.active = true
	p.modelBuffer = buffer
	p.lastAccess = time.Now()

	slog.Info("starting model memory preloading", 
		"model", p.modelName,
		"size", format.HumanBytes2(p.modelSize))

	// Start background goroutine to maintain model in memory
	go p.preloadRoutine()
}

// StopPreloading stops the preloading process
func (p *ModelPreloader) StopPreloading() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.active {
		return
	}

	close(p.stop)
	p.active = false
	p.modelBuffer = nil

	slog.Info("stopped model memory preloading", "model", p.modelName)
}

// UpdateLastAccess updates the timestamp when the model was last accessed
func (p *ModelPreloader) UpdateLastAccess() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastAccess = time.Now()
}

// preloadRoutine is the background routine that keeps model pages in memory
func (p *ModelPreloader) preloadRoutine() {
	// Touch all memory pages initially
	p.touchAllPages()

	// Keep touching pages periodically to prevent them from being swapped out
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Use the system-wide keep alive timeout setting
	inactivityTimeout := envconfig.KeepAlive()
	
	for {
		select {
		case <-p.stop:
			return
		case <-ticker.C:
			p.mu.Lock()
			lastAccess := p.lastAccess
			p.mu.Unlock()

			// If the model hasn't been accessed for a while, stop preloading
			if time.Since(lastAccess) > inactivityTimeout {
				slog.Info("model inactive, stopping preloading", 
					"model", p.modelName, 
					"inactivity", time.Since(lastAccess))
				p.StopPreloading()
				return
			}

			// Touch pages to keep them in memory
			p.touchPages()
		}
	}
}

// touchAllPages reads through all model memory pages to bring them into RAM
func (p *ModelPreloader) touchAllPages() {
	p.mu.Lock()
	if !p.active || p.modelBuffer == nil {
		p.mu.Unlock()
		return
	}
	buffer := p.modelBuffer
	p.mu.Unlock()

	// System page size, typically 4KB
	pageSize := 4096
	sum := byte(0)

	// Process model in chunks to avoid excessive CPU usage
	chunkSize := 64 * 1024 * 1024 // 64MB chunks
	numChunks := (len(buffer) + chunkSize - 1) / chunkSize
	
	slog.Debug("touching all model memory pages", 
		"model", p.modelName,
		"chunks", numChunks, 
		"total_size", format.HumanBytes2(uint64(len(buffer))))

	start := time.Now()
	
	for chunkIdx := 0; chunkIdx < numChunks; chunkIdx++ {
		startOffset := chunkIdx * chunkSize
		endOffset := min(startOffset+chunkSize, len(buffer))
		
		// Touch each page in the chunk
		for offset := startOffset; offset < endOffset; offset += pageSize {
			// Just read one byte per page to touch it
			sum ^= buffer[offset]
		}
		
		// Allow other goroutines to run
		if chunkIdx%16 == 0 {
			runtime.Gosched()
		}
	}
	
	// Use sum to prevent compiler optimization removing the reads
	if sum != 0 {
		_ = sum
	}
	
	slog.Info("completed initial model memory preloading", 
		"model", p.modelName,
		"duration", time.Since(start), 
		"size", format.HumanBytes2(uint64(len(buffer))))
}

// touchPages touches a subset of pages to keep them warm
func (p *ModelPreloader) touchPages() {
	p.mu.Lock()
	if !p.active || p.modelBuffer == nil {
		p.mu.Unlock()
		return
	}
	buffer := p.modelBuffer
	p.mu.Unlock()

	// System page size
	pageSize := 4096
	
	// Touch every Nth page (sparse touching to reduce overhead)
	touchInterval := 16 * pageSize
	sum := byte(0)
	
	for offset := 0; offset < len(buffer); offset += touchInterval {
		sum ^= buffer[offset]
	}
	
	// Use sum to prevent compiler optimization removing the reads
	if sum != 0 {
		_ = sum
	}
}
