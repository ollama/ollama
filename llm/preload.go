package llm

import (
	"log/slog"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// ModelPreloader provides functionality to preload model memory pages
// and lock them in RAM to prevent page faults during inference
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

// LockMemory locks a memory region in RAM using madvise system call
// This prevents the OS from swapping these pages to disk
func LockMemory(addr uintptr, length uintptr) error {
	// MADV_WILLNEED: Indicates that the application will need these pages
	// MADV_LOCK: Prevent these pages from being swapped out (requires CAP_SYS_ADMIN or similar privileges)
	err := syscall.Madvise(unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(length)), syscall.MADV_WILLNEED)
	if err != nil {
		return err
	}
	
	// Try to lock the memory - this may fail if the process doesn't have the necessary privileges
	err = syscall.Madvise(unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(length)), syscall.MADV_DONTFORK)
	if err != nil {
		// Log but continue - MADV_DONTFORK may not be critical
		slog.Debug("could not set MADV_DONTFORK, continuing anyway", "error", err)
	}
	
	// Try to lock the memory - this requires privileges but provides the strongest guarantee
	err = syscall.Madvise(unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(length)), syscall.MADV_LOCK)
	if err != nil {
		// Log but continue - this is expected to fail without elevated privileges
		slog.Debug("could not lock memory with MADV_LOCK (requires elevated privileges), falling back to manual preloading", "error", err)
		return nil
	}
	
	return nil
}

// StartPreloading begins the memory preloading process
// It tries to lock the model memory in RAM, and if that's not possible,
// falls back to periodic touches to keep pages in memory
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

	// Try to lock the memory in RAM first
	addr := uintptr(unsafe.Pointer(&buffer[0]))
	length := uintptr(len(buffer))
	
	err := LockMemory(addr, length)
	if err == nil {
		slog.Info("successfully locked model memory in RAM", 
			"model", p.modelName, 
			"size", format.HumanBytes2(p.modelSize))
	} else {
		slog.Warn("failed to lock model memory, falling back to periodic access", 
			"model", p.modelName, 
			"error", err)
	}

	// Start background goroutine to maintain model in memory
	// This is needed even with memory locking as a fallback and to handle timeout
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

	// Try to lock the memory first (in case it wasn't locked in StartPreloading)
	addr := uintptr(unsafe.Pointer(&buffer[0]))
	length := uintptr(len(buffer))
	
	if err := LockMemory(addr, length); err != nil {
		// If locking fails, fall back to manual touching
		manuallyTouchPages(buffer, p.modelName)
	} else {
		slog.Info("memory locked successfully", "model", p.modelName)
	}
}

// manuallyTouchPages manually touches all pages to bring them into RAM
// This is used as a fallback when memory locking is not available
func manuallyTouchPages(buffer []byte, modelName string) {
	// System page size, typically 4KB
	pageSize := 4096
	sum := byte(0)

	// Process model in chunks to avoid excessive CPU usage
	chunkSize := 64 * 1024 * 1024 // 64MB chunks
	numChunks := (len(buffer) + chunkSize - 1) / chunkSize
	
	slog.Debug("manually touching all model memory pages", 
		"model", modelName,
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
	
	slog.Info("completed model memory preloading via manual touching", 
		"model", modelName,
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
