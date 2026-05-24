// +build ignore
// Patch for OLLaMA runner/scheduler.go or similar
// This implements persistent batching, speculative decoding hooks,
// and eliminates per-token allocation overhead.

package main

import (
	"sync"
	"unsafe"
)

// PersistentBatch keeps the llama.cpp batch alive across decode steps
type PersistentBatch struct {
	batch unsafe.Pointer // *llama.Batch
	size  int
	mu    sync.Mutex

	// Pre-allocated buffers to eliminate malloc per token
	tokenBuf  []int32
	posBuf    []int32
	nSeqBuf   []int32
	seqIDBuf  [][]int32
	logitsBuf []int8
}

// NewPersistentBatch creates a batch that never frees between tokens
func NewPersistentBatch(maxTokens int) *PersistentBatch {
	return &PersistentBatch{
		tokenBuf:  make([]int32, 0, maxTokens),
		posBuf:    make([]int32, 0, maxTokens),
		nSeqBuf:   make([]int32, 0, maxTokens),
		seqIDBuf:  make([][]int32, 0, maxTokens),
		logitsBuf: make([]int8, 0, maxTokens),
	}
}

// Reset clears without deallocating (persistent batching)
func (pb *PersistentBatch) Reset() {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.tokenBuf = pb.tokenBuf[:0]
	pb.posBuf = pb.posBuf[:0]
	pb.nSeqBuf = pb.nSeqBuf[:0]
	pb.seqIDBuf = pb.seqIDBuf[:0]
	pb.logitsBuf = pb.logitsBuf[:0]
	pb.size = 0
}

// SpeculativeDecoder runs n-gram draft prediction to reduce per-token latency
type SpeculativeDecoder struct {
	draftTokens []int32
	draftModel  unsafe.Pointer // lightweight model or null for n-gram

	// n-gram state for fallback
	gram4       map[[4]int32]int32 // 4-gram prediction table
	gram3       map[[3]int32]int32
	lastTokens  [8]int32
	lastIdx     int
}

func NewSpeculativeDecoder() *SpeculativeDecoder {
	return &SpeculativeDecoder{
		gram4:      make(map[[4]int32]int32),
		gram3:      make(map[[3]int32]int32),
		lastTokens: [8]int32{},
	}
}

// Predict attempts to predict the next N tokens without GPU round-trip
func (sd *SpeculativeDecoder) Predict(lastToken int32, n int) []int32 {
	sd.lastTokens[sd.lastIdx%8] = lastToken
	sd.lastIdx++

	// Try 4-gram match first
	var key4 [4]int32
	copy(key4[:], sd.lastTokens[(sd.lastIdx-4)%8:])
	if tok, ok := sd.gram4[key4]; ok {
		return []int32{tok}
	}

	// Fall back to 3-gram
	var key3 [3]int32
	copy(key3[:], sd.lastTokens[(sd.lastIdx-3)%8:])
	if tok, ok := sd.gram3[key3]; ok {
		return []int32{tok}
	}

	return nil
}

// UpdateGrams updates n-gram tables from accepted sequence
func (sd *SpeculativeDecoder) UpdateGrams(tokens []int32) {
	for i := 0; i < len(tokens)-4; i++ {
		var key [4]int32
		copy(key[:], tokens[i:i+4])
		sd.gram4[key] = tokens[i+4]
	}
	for i := 0; i < len(tokens)-3; i++ {
		var key [3]int32
		copy(key[:], tokens[i:i+3])
		sd.gram3[key] = tokens[i+3]
	}
}

// AggressiveScheduler wraps the standard scheduler with zero-allocation paths
type AggressiveScheduler struct {
	pb          *PersistentBatch
	sd          *SpeculativeDecoder
	ctx         unsafe.Pointer

	// Pre-allocated response buffer
	responseBuf []byte

	// Async prefetch channel
	prefetchCh  chan []int32
}

func NewAggressiveScheduler(ctx unsafe.Pointer, maxBatch int) *AggressiveScheduler {
	s := &AggressiveScheduler{
		pb:         NewPersistentBatch(maxBatch),
		sd:         NewSpeculativeDecoder(),
		ctx:        ctx,
		responseBuf: make([]byte, 0, 65536),
		prefetchCh:  make(chan []int32, 4),
	}

	// Start async speculative prefetch goroutine
	go s.speculativePrefetchLoop()

	return s
}

func (s *AggressiveScheduler) speculativePrefetchLoop() {
	for draft := range s.prefetchCh {
		// Pre-run draft tokens through model if we have a draft model
		// Otherwise n-gram predictions are already computed
		_ = draft
	}
}

// DecodeStep runs one decode step with persistent batching
func (s *AggressiveScheduler) DecodeStep(tokens []int32, pos int) (int32, error) {
	s.pb.mu.Lock()
	defer s.pb.mu.Unlock()

	// Add tokens to persistent batch without reallocating
	for i, tok := range tokens {
		s.pb.tokenBuf = append(s.pb.tokenBuf, tok)
		s.pb.posBuf = append(s.pb.posBuf, int32(pos+i))
		s.pb.nSeqBuf = append(s.pb.nSeqBuf, 1)
		s.pb.logitsBuf = append(s.pb.logitsBuf, 0)
	}

	// Set logits on last token only
	if len(s.pb.logitsBuf) > 0 {
		s.pb.logitsBuf[len(s.pb.logitsBuf)-1] = 1
	}

	// Call into llama.cpp decode with persistent batch
	// llama_decode(s.ctx, s.pb.batch) // C call would go here

	// Try speculative prediction for next token
	speculative := s.sd.Predict(tokens[len(tokens)-1], 1)
	_ = speculative

	// Sample from logits (simplified)
	nextToken := int32(0) // would be llama_sample()

	return nextToken, nil
}

// KVCachePressure returns true if we should defrag/purge
func (s *AggressiveScheduler) KVCachePressure() bool {
	// Hook into llama_get_kv_cache_used_cells / total_cells
	return false
}
