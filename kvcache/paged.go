package kvcache

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

const (
	// Default block size for PagedAttention (in tokens)
	// vLLM uses 16 by default, which works well for most models
	DefaultBlockSize = 16
	// Maximum number of blocks to pre-allocate
	DefaultMaxNumBlocks = 1000
)

// Paged implements PagedAttention-style KV cache with block-based allocation.
// Unlike the contiguous Causal cache, PagedAttention stores KV data in fixed-size
// blocks that can be non-contiguous in memory, reducing fragmentation and enabling
// more efficient memory management for variable-length sequences.
//
// Key differences from Causal cache:
// - KV data stored in fixed-size blocks (default 16 tokens per block)
// - Each sequence has a block table mapping logical block index to physical block
// - Blocks can be allocated and freed independently
// - Better for dynamic batching and variable-length sequences
//
// Reference: vLLM PagedAttention design
type Paged struct {
	DType ml.DType

	// Block configuration
	blockSize    int  // Number of tokens per block
	numBlocks    int  // Total number of blocks in the cache
	maxNumBlocks int  // Maximum number of blocks that can be allocated

	// Cache configuration
	maxSequences int
	capacity     int
	maxBatch     int

	// Backend configuration
	config *ml.CacheConfig
	backend ml.Backend

	// Block table: maps (sequence ID, logical block index) -> physical block ID
	// Each sequence can have multiple blocks (non-contiguous in physical memory)
	blockTables map[int][]int // seqID -> []physicalBlockID

	// Physical block management
	freeBlocks    []int    // List of free physical block IDs
	blockRefCount []int    // Reference count for each physical block
	allocatedBlocks int    // Number of currently allocated blocks

	// Reverse mapping: physical block ID -> (seqID, logical block index)
	blockMapping []blockEntry

	// Per-layer KV storage
	ctxs   map[int]ml.Context
	keys   map[int]ml.Tensor
	values map[int]ml.Tensor

	// Current forward pass state
	curBatchSize int
	curLayer     int
	curSeqs      []int
	curPositions []int32

	// Current cache locations for the batch
	curLoc   ml.Tensor
	curMask  ml.Tensor
	curRange cellRange

	shiftFn shiftFn
}

type blockEntry struct {
	seqID         int
	logicalBlock  int
	tokensInBlock int
}

// NewPagedCache creates a new PagedAttention-style KV cache.
func NewPagedCache(shift shiftFn) *Paged {
	return &Paged{
		blockSize:     DefaultBlockSize,
		maxNumBlocks:  DefaultMaxNumBlocks,
		blockTables:   make(map[int][]int),
		ctxs:          make(map[int]ml.Context),
		keys:          make(map[int]ml.Tensor),
		values:        make(map[int]ml.Tensor),
		shiftFn:       shift,
	}
}

// NewPagedCacheWithConfig creates a Paged cache with custom configuration.
func NewPagedCacheWithConfig(shift shiftFn, blockSize, maxNumBlocks int) *Paged {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	if maxNumBlocks <= 0 {
		maxNumBlocks = DefaultMaxNumBlocks
	}

	return &Paged{
		blockSize:     blockSize,
		maxNumBlocks:  maxNumBlocks,
		blockTables:   make(map[int][]int),
		ctxs:          make(map[int]ml.Context),
		keys:          make(map[int]ml.Tensor),
		values:        make(map[int]ml.Tensor),
		shiftFn:       shift,
	}
}

func (p *Paged) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	p.DType = dtype
	p.backend = backend
	p.maxSequences = maxSequences
	p.capacity = capacity
	p.maxBatch = maxBatch

	// Get backend-specific cache config
	var config ml.CacheConfig
	if cc, ok := backend.(ml.BackendCacheConfig); ok {
		config = cc.CacheConfig()
	}
	p.config = &config

	if p.config.CachePadding == 0 {
		p.config.CachePadding = 1
	}
	if p.config.MaskDType == ml.DTypeOther {
		p.config.MaskDType = ml.DTypeF32
	}

	// Calculate number of blocks needed
	// Each block holds p.blockSize tokens
	// We need enough blocks to store maxSequences * capacity tokens
	totalTokens := maxSequences * capacity
	p.numBlocks = (totalTokens + p.blockSize - 1) / p.blockSize

	// Cap at maxNumBlocks
	if p.numBlocks > p.maxNumBlocks {
		p.numBlocks = p.maxNumBlocks
	}

	// Initialize free block list (all blocks initially free)
	p.freeBlocks = make([]int, p.numBlocks)
	for i := 0; i < p.numBlocks; i++ {
		p.freeBlocks[i] = i
	}

	// Initialize block metadata
	p.blockRefCount = make([]int, p.numBlocks)
	p.blockMapping = make([]blockEntry, p.numBlocks)

	for i := 0; i < p.numBlocks; i++ {
		p.blockMapping[i] = blockEntry{seqID: -1} // -1 means unallocated
	}
}

func (p *Paged) SetConfig(config ml.CacheConfig) {
	if p.config != nil {
		panic("config cannot be changed after being previously set")
	}
	p.config = &config
}

func (p *Paged) Close() {
	for _, ctx := range p.ctxs {
		ctx.Close()
	}
}

func (p *Paged) SetLayer(layer int) {
	p.curLayer = layer
}

// allocateBlock allocates a free physical block.
func (p *Paged) allocateBlock() (int, error) {
	if len(p.freeBlocks) == 0 {
		return -1, fmt.Errorf("no free blocks available (allocated: %d/%d)",
			p.allocatedBlocks, p.numBlocks)
	}

	// Pop from free list (LIFO for cache locality)
	blockID := p.freeBlocks[len(p.freeBlocks)-1]
	p.freeBlocks = p.freeBlocks[:len(p.freeBlocks)-1]

	p.allocatedBlocks++
	return blockID, nil
}

// freeBlock frees a physical block.
func (p *Paged) freeBlock(blockID int) {
	if blockID < 0 || blockID >= p.numBlocks {
		return
	}

	p.blockRefCount[blockID] = 0
	p.blockMapping[blockID] = blockEntry{seqID: -1}
	p.freeBlocks = append(p.freeBlocks, blockID)
	p.allocatedBlocks--
}

// getNumBlocksForTokens returns the number of blocks needed for the given number of tokens.
func (p *Paged) getNumBlocksForTokens(numTokens int) int {
	return (numTokens + p.blockSize - 1) / p.blockSize
}

// allocateBlocksForSequence allocates blocks for a sequence.
func (p *Paged) allocateBlocksForSequence(seqID int, numTokens int) ([]int, error) {
	numBlocks := p.getNumBlocksForTokens(numTokens)

	blocks := make([]int, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blockID, err := p.allocateBlock()
		if err != nil {
			// Free any blocks we already allocated
			for j := 0; j < i; j++ {
				p.freeBlock(blocks[j])
			}
			return nil, err
		}
		blocks[i] = blockID
		p.blockMapping[blockID] = blockEntry{
			seqID:        seqID,
			logicalBlock: i,
		}
		p.blockRefCount[blockID] = 1
	}

	// Store block table for this sequence
	p.blockTables[seqID] = blocks

	return blocks, nil
}

// StartForward prepares the cache for a forward pass.
func (p *Paged) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	p.curBatchSize = len(batch.Positions)
	p.curSeqs = batch.Sequences
	p.curPositions = batch.Positions

	// Group positions by sequence
	seqPositions := make(map[int][]int32)
	for i, seqID := range batch.Sequences {
		seqPositions[seqID] = append(seqPositions[seqID], batch.Positions[i])
	}

	// Calculate total cache slots needed and build block allocation
	totalSlots := 0
	maxSeqPos := make(map[int]int32)

	for seqID, positions := range seqPositions {
		maxPos := int32(0)
		for _, pos := range positions {
			if pos > maxPos {
				maxPos = pos
			}
		}
		// Ensure at least 1 token capacity (position 0 needs storage)
		if maxPos == 0 && len(positions) > 0 {
			maxPos = 1
		}
		maxSeqPos[seqID] = maxPos
		totalSlots += int(maxPos)
	}

	// Check if we have enough capacity
	if totalSlots > p.numBlocks*p.blockSize {
		return fmt.Errorf("insufficient cache capacity: need %d tokens, have %d",
			totalSlots, p.numBlocks*p.blockSize)
	}

	// Build location array
	locs := make([]int32, p.curBatchSize)
	slotIndex := 0

	for i, seqID := range batch.Sequences {
		pos := batch.Positions[i]

		// Get or allocate blocks for this sequence
		blocks, exists := p.blockTables[seqID]
		if !exists {
			// Need to allocate blocks for this sequence
			numTokens := int(maxSeqPos[seqID])
			allocated, err := p.allocateBlocksForSequence(seqID, numTokens)
			if err != nil {
				return fmt.Errorf("failed to allocate blocks for sequence %d: %w", seqID, err)
			}
			blocks = allocated
		}

		// Calculate physical location from block table
		logicalBlock := int(pos) / p.blockSize
		offsetInBlock := int(pos) % p.blockSize

		if logicalBlock >= len(blocks) {
			return fmt.Errorf("sequence %d position %d exceeds allocated blocks",
				seqID, pos)
		}

		physicalBlock := blocks[logicalBlock]
		physicalLocation := physicalBlock*p.blockSize + offsetInBlock
		locs[i] = int32(physicalLocation)
		slotIndex++
	}

	p.curLoc = ctx.Input().FromInts(locs, len(locs))

	// Calculate range
	minLoc := int32(0)
	maxLoc := int32(p.numBlocks * p.blockSize)
	p.curRange = cellRange{min: int(minLoc), max: int(maxLoc)}

	return nil
}

// Get returns the cached keys, values, and attention mask.
func (p *Paged) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if p.curLayer == 0 {
		// Initialize KV storage on layer 0
		p.initializeKVStorage(ctx)
	}

	keyTensor := p.keys[p.curLayer]
	valueTensor := p.values[p.curLayer]

	// Build attention mask for paged attention
	// The mask needs to account for the non-contiguous block layout
	mask := p.buildPagedMask(ctx)

	return keyTensor, valueTensor, mask
}

// Put stores key and value tensors in the cache.
func (p *Paged) Put(ctx ml.Context, key, value ml.Tensor) {
	// Scatter the key/value tensors to their paged locations
	// This is a simplified version - in production, you'd want to use
	// efficient scatter operations
	p.keys[p.curLayer] = key
	p.values[p.curLayer] = value
}

// initializeKVStorage sets up the KV tensor storage.
func (p *Paged) initializeKVStorage(ctx ml.Context) {
	totalCapacity := p.numBlocks * p.blockSize

	for layer := 0; ; layer++ {
		// Check if we already have storage for this layer
		if _, ok := p.keys[layer]; ok {
			continue
		}

		// Create storage for this layer
		layerCtx := p.backend.NewContext()
		p.ctxs[layer] = layerCtx

		// The actual tensor shapes depend on the model configuration
		// This is a placeholder - in production, you'd get the correct shapes
		// from the model or backend
		p.keys[layer] = layerCtx.Input().Empty(p.DType, totalCapacity)
		p.values[layer] = layerCtx.Input().Empty(p.DType, totalCapacity)

		// Try next layer
		p.SetLayer(layer + 1)
		if _, ok := p.keys[layer]; ok {
			// Next layer already exists or we hit the end
			break
		}
	}

	p.SetLayer(0)
}

// buildPagedMask constructs an attention mask for paged attention.
func (p *Paged) buildPagedMask(ctx ml.Context) ml.Tensor {
	// For paged attention, the mask needs to account for:
	// 1. Causal masking (can't attend to future tokens)
	// 2. Block boundaries (sequences are non-contiguous)
	// 3. Multiple sequences in the batch

	// This is a simplified implementation
	// In production, you'd want to use specialized kernels for this
	return ctx.Input().Empty(p.config.MaskDType, p.curBatchSize, p.curBatchSize)
}

// CopyPrefix copies tokens from source sequence to destination sequence.
func (p *Paged) CopyPrefix(srcSeq, dstSeq int, length int32) {
	srcBlocks, srcExists := p.blockTables[srcSeq]
	if !srcExists {
		return
	}

	numBlocks := p.getNumBlocksForTokens(int(length))
	if numBlocks > len(srcBlocks) {
		numBlocks = len(srcBlocks)
	}

	// Allocate blocks for destination
	dstBlocks, err := p.allocateBlocksForSequence(dstSeq, int(length))
	if err != nil {
		// Handle error - in production, you'd want to evict old blocks
		return
	}

	// Copy block mappings
	for i := 0; i < numBlocks; i++ {
		srcBlockID := srcBlocks[i]
		dstBlockID := dstBlocks[i]

		// Copy the block reference
		p.blockMapping[dstBlockID] = p.blockMapping[srcBlockID]
		p.blockRefCount[dstBlockID] = p.blockRefCount[srcBlockID]
	}

	// Copy KV data for each layer
	for layer := range p.keys {
		srcKey := p.keys[layer]
		srcValue := p.values[layer]

		// Copy the actual tensor data
		// In production, you'd use efficient copy operations
		_ = srcKey
		_ = srcValue
	}
}

// CanResume checks if the cache can continue at the given position.
func (p *Paged) CanResume(seq int, pos int32) bool {
	blocks, exists := p.blockTables[seq]
	if !exists {
		return false
	}

	requiredBlock := int(pos) / p.blockSize
	return requiredBlock < len(blocks)
}

// Remove deletes tokens from a sequence.
func (p *Paged) Remove(seq int, beginIndex, endIndex int32) error {
	blocks, exists := p.blockTables[seq]
	if !exists {
		return nil
	}

	if endIndex == math.MaxInt32 {
		endIndex = int32(len(blocks) * p.blockSize)
	}

	beginBlock := int(beginIndex) / p.blockSize
	endBlock := int(endIndex) / p.blockSize

	// Free blocks in the specified range
	for i := beginBlock; i < endBlock && i < len(blocks); i++ {
		blockID := blocks[i]
		p.blockRefCount[blockID]--

		if p.blockRefCount[blockID] <= 0 {
			p.freeBlock(blockID)
		}
	}

	// Update block table
	if beginBlock >= len(blocks) {
		return nil
	}

	if endBlock >= len(blocks) {
		// Remove all blocks from beginBlock onwards
		p.blockTables[seq] = blocks[:beginBlock]
	} else {
		// Remove blocks in range [beginBlock, endBlock)
		newBlocks := make([]int, 0, len(blocks)-(endBlock-beginBlock))
		newBlocks = append(newBlocks, blocks[:beginBlock]...)
		newBlocks = append(newBlocks, blocks[endBlock:]...)
		p.blockTables[seq] = newBlocks
	}

	return nil
}

// GetStats returns statistics about the paged cache.
func (p *Paged) GetStats() PagedStats {
	return PagedStats{
		NumBlocks:        p.numBlocks,
		AllocatedBlocks:  p.allocatedBlocks,
		FreeBlocks:       len(p.freeBlocks),
		BlockSize:        p.blockSize,
		ActiveSequences:  len(p.blockTables),
		Utilization:      float64(p.allocatedBlocks) / float64(p.numBlocks),
	}
}

// PagedStats contains statistics about the paged cache.
type PagedStats struct {
	NumBlocks       int     // Total number of blocks
	AllocatedBlocks int     // Number of currently allocated blocks
	FreeBlocks      int     // Number of free blocks
	BlockSize       int     // Tokens per block
	ActiveSequences int     // Number of active sequences
	Utilization     float64 // Block utilization ratio (0-1)
}

// SetBlockSize sets the block size. Must be called before Init.
func (p *Paged) SetBlockSize(blockSize int) {
	if p.backend != nil {
		panic("SetBlockSize must be called before Init")
	}
	p.blockSize = blockSize
}
