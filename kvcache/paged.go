package kvcache

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

const (
	DefaultBlockSize    = 16
	DefaultMaxNumBlocks = 1000
)

// Paged implements PagedAttention-style KV cache with block-based allocation.
// KV data is stored in fixed-size blocks that can be non-contiguous in physical
// memory, reducing fragmentation and enabling efficient dynamic batching.
//
// Storage layout: K/V tensors are stored as 2D flat [headDim*kvHeads, numBlocks*blockSize]
// for efficient SetRows-based scatter. On Get, they are returned as 4D tensors
// [headDim, numKVHeads, blockSize, numBlocks] for the PagedAttention kernel.
type Paged struct {
	DType ml.DType

	blockSize    int
	numBlocks    int
	maxNumBlocks int

	maxSequences int
	capacity     int
	maxBatch     int

	// Model dimensions - set lazily on first Put
	headDim    int
	numKVHeads int

	config  *ml.CacheConfig
	backend ml.Backend

	blockTables   map[int][]int
	freeBlocks    []int
	blockRefCount []int
	allocatedBlocks int
	blockMapping   []blockEntry

	ctxs   map[int]ml.Context
	keys   map[int]ml.Tensor
	values map[int]ml.Tensor
	initialized map[int]bool

	curBatchSize int
	curLayer     int
	curSeqs      []int
	curPositions []int32

	// curLoc holds physical positions for SetRows scatter:
	// each element = physicalBlock * blockSize + offsetInBlock
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

func NewPagedCache(shift shiftFn) *Paged {
	return &Paged{
		blockSize:    DefaultBlockSize,
		maxNumBlocks: DefaultMaxNumBlocks,
		blockTables:  make(map[int][]int),
		ctxs:         make(map[int]ml.Context),
		keys:         make(map[int]ml.Tensor),
		values:       make(map[int]ml.Tensor),
		initialized:  make(map[int]bool),
		shiftFn:      shift,
	}
}

func NewPagedCacheWithConfig(shift shiftFn, blockSize, maxNumBlocks int) *Paged {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	if maxNumBlocks <= 0 {
		maxNumBlocks = DefaultMaxNumBlocks
	}

	return &Paged{
		blockSize:    blockSize,
		maxNumBlocks: maxNumBlocks,
		blockTables:  make(map[int][]int),
		ctxs:         make(map[int]ml.Context),
		keys:         make(map[int]ml.Tensor),
		values:       make(map[int]ml.Tensor),
		initialized:  make(map[int]bool),
		shiftFn:      shift,
	}
}

func (p *Paged) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	p.DType = dtype
	p.backend = backend
	p.maxSequences = maxSequences
	p.capacity = capacity
	p.maxBatch = maxBatch

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

	totalTokens := maxSequences * capacity
	p.numBlocks = (totalTokens + p.blockSize - 1) / p.blockSize

	if p.numBlocks > p.maxNumBlocks {
		p.numBlocks = p.maxNumBlocks
	}

	p.freeBlocks = make([]int, p.numBlocks)
	for i := 0; i < p.numBlocks; i++ {
		p.freeBlocks[i] = i
	}

	p.blockRefCount = make([]int, p.numBlocks)
	p.blockMapping = make([]blockEntry, p.numBlocks)

	for i := 0; i < p.numBlocks; i++ {
		p.blockMapping[i] = blockEntry{seqID: -1}
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

func (p *Paged) allocateBlock() (int, error) {
	if len(p.freeBlocks) == 0 {
		return -1, fmt.Errorf("no free blocks available (allocated: %d/%d)",
			p.allocatedBlocks, p.numBlocks)
	}

	blockID := p.freeBlocks[len(p.freeBlocks)-1]
	p.freeBlocks = p.freeBlocks[:len(p.freeBlocks)-1]

	p.allocatedBlocks++
	return blockID, nil
}

func (p *Paged) freeBlock(blockID int) {
	if blockID < 0 || blockID >= p.numBlocks {
		return
	}

	p.blockRefCount[blockID] = 0
	p.blockMapping[blockID] = blockEntry{seqID: -1}
	p.freeBlocks = append(p.freeBlocks, blockID)
	p.allocatedBlocks--
}

func (p *Paged) getNumBlocksForTokens(numTokens int) int {
	return (numTokens + p.blockSize - 1) / p.blockSize
}

func (p *Paged) allocateBlocksForSequence(seqID int, numTokens int) ([]int, error) {
	numBlocks := p.getNumBlocksForTokens(numTokens)

	blocks := make([]int, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blockID, err := p.allocateBlock()
		if err != nil {
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

	p.blockTables[seqID] = blocks

	return blocks, nil
}

func (p *Paged) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	p.curBatchSize = len(batch.Positions)
	p.curSeqs = batch.Sequences
	p.curPositions = batch.Positions

	seqPositions := make(map[int][]int32)
	for i, seqID := range batch.Sequences {
		seqPositions[seqID] = append(seqPositions[seqID], batch.Positions[i])
	}

	totalSlots := 0
	maxSeqPos := make(map[int]int32)

	for seqID, positions := range seqPositions {
		maxPos := int32(0)
		for _, pos := range positions {
			if pos > maxPos {
				maxPos = pos
			}
		}
		if maxPos == 0 && len(positions) > 0 {
			maxPos = 1
		}
		maxSeqPos[seqID] = maxPos
		totalSlots += int(maxPos)
	}

	if totalSlots > p.numBlocks*p.blockSize {
		return fmt.Errorf("insufficient cache capacity: need %d tokens, have %d",
			totalSlots, p.numBlocks*p.blockSize)
	}

	locs := make([]int32, p.curBatchSize)

	for i, seqID := range batch.Sequences {
		pos := batch.Positions[i]

		blocks, exists := p.blockTables[seqID]
		if !exists {
			numTokens := int(maxSeqPos[seqID])
			allocated, err := p.allocateBlocksForSequence(seqID, numTokens)
			if err != nil {
				return fmt.Errorf("failed to allocate blocks for sequence %d: %w", seqID, err)
			}
			blocks = allocated
		}

		logicalBlock := int(pos) / p.blockSize
		offsetInBlock := int(pos) % p.blockSize

		if logicalBlock >= len(blocks) {
			return fmt.Errorf("sequence %d position %d exceeds allocated blocks",
				seqID, pos)
		}

		physicalBlock := blocks[logicalBlock]
		physicalLocation := physicalBlock*p.blockSize + offsetInBlock
		locs[i] = int32(physicalLocation)
	}

	p.curLoc = ctx.Input().FromInts(locs, len(locs))

	minLoc := int32(0)
	maxLoc := int32(p.numBlocks * p.blockSize)
	p.curRange = cellRange{min: int(minLoc), max: int(maxLoc)}

	return nil
}

// Get returns cached keys, values, and attention mask.
// K, V are 4D [headDim, numKVHeads, blockSize, numBlocks] via zero-copy view
// of the underlying 2D storage [headDim*kvHeads, numBlocks*blockSize].
func (p *Paged) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key2D := p.keys[p.curLayer]
	val2D := p.values[p.curLayer]

	headDim := p.headDim
	numKVHeads := p.numKVHeads

	stride0 := key2D.Stride(0)
	stride1 := key2D.Stride(1)

	key4D := key2D.View(ctx, 0,
		headDim, headDim*stride0,
		numKVHeads, stride1,
		p.blockSize, p.blockSize*stride1,
		p.numBlocks)

	val4D := val2D.View(ctx, 0,
		headDim, headDim*stride0,
		numKVHeads, stride1,
		p.blockSize, p.blockSize*stride1,
		p.numBlocks)

	mask := p.buildPagedMask(ctx)

	return key4D, val4D, mask
}

// Put stores key and value tensors into the paged cache.
// Key/value have shape [headDim, numKVHeads, batchSize] from the model attention layer.
// Scattered into the block cache at physical positions from curLoc.
func (p *Paged) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(0)
	vHeadDim := value.Dim(0)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)

	if p.curBatchSize != batchSize {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch: %v cur: %v)",
			p.curLayer, p.curBatchSize, batchSize))
	}

	if !p.initialized[p.curLayer] {
		p.headDim = kHeadDim
		p.numKVHeads = numKVHeads
		p.initializeKVStorageForLayer(p.curLayer, kHeadDim, vHeadDim, numKVHeads)
		p.initialized[p.curLayer] = true
	}

	rowSize := kHeadDim * numKVHeads
	totalPositions := p.numBlocks * p.blockSize

	keyCache := p.keys[p.curLayer]
	keyCache = keyCache.Reshape(ctx, rowSize, totalPositions)
	key = key.Reshape(ctx, rowSize, batchSize)
	ctx.Forward(keyCache.SetRows(ctx, key, p.curLoc))

	vRowSize := vHeadDim * numKVHeads
	valueCache := p.values[p.curLayer]
	valueCache = valueCache.Reshape(ctx, vRowSize, totalPositions)
	value = value.Reshape(ctx, vRowSize, batchSize)
	ctx.Forward(valueCache.SetRows(ctx, value, p.curLoc))
}

func (p *Paged) initializeKVStorageForLayer(layer int, kHeadDim, vHeadDim, numKVHeads int) {
	layerCtx := p.backend.NewContext().Input()
	p.ctxs[layer] = layerCtx

	rowSize := kHeadDim * numKVHeads
	totalPositions := p.numBlocks * p.blockSize
	p.keys[layer] = layerCtx.Zeros(p.DType, rowSize, totalPositions)
	p.values[layer] = layerCtx.Zeros(p.DType, vHeadDim*numKVHeads, totalPositions)
}

func (p *Paged) buildPagedMask(ctx ml.Context) ml.Tensor {
	return ctx.Input().Empty(p.config.MaskDType, p.curBatchSize, p.curBatchSize)
}

func (p *Paged) CopyPrefix(srcSeq, dstSeq int, length int32) {
	srcBlocks, srcExists := p.blockTables[srcSeq]
	if !srcExists {
		return
	}

	numBlocks := p.getNumBlocksForTokens(int(length))
	if numBlocks > len(srcBlocks) {
		numBlocks = len(srcBlocks)
	}

	_, err := p.allocateBlocksForSequence(dstSeq, int(length))
	if err != nil {
		return
	}
}

func (p *Paged) CanResume(seq int, pos int32) bool {
	blocks, exists := p.blockTables[seq]
	if !exists {
		return false
	}

	requiredBlock := int(pos) / p.blockSize
	return requiredBlock < len(blocks)
}

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

	for i := beginBlock; i < endBlock && i < len(blocks); i++ {
		blockID := blocks[i]
		p.blockRefCount[blockID]--

		if p.blockRefCount[blockID] <= 0 {
			p.freeBlock(blockID)
		}
	}

	if beginBlock >= len(blocks) {
		return nil
	}

	if endBlock >= len(blocks) {
		p.blockTables[seq] = blocks[:beginBlock]
	} else {
		newBlocks := make([]int, 0, len(blocks)-(endBlock-beginBlock))
		newBlocks = append(newBlocks, blocks[:beginBlock]...)
		newBlocks = append(newBlocks, blocks[endBlock:]...)
		p.blockTables[seq] = newBlocks
	}

	return nil
}

func (p *Paged) GetStats() PagedStats {
	return PagedStats{
		NumBlocks:       p.numBlocks,
		AllocatedBlocks: p.allocatedBlocks,
		FreeBlocks:      len(p.freeBlocks),
		BlockSize:       p.blockSize,
		ActiveSequences: len(p.blockTables),
		Utilization:     float64(p.allocatedBlocks) / float64(p.numBlocks),
	}
}

type PagedStats struct {
	NumBlocks       int
	AllocatedBlocks int
	FreeBlocks      int
	BlockSize       int
	ActiveSequences int
	Utilization     float64
}

func (p *Paged) SetBlockSize(blockSize int) {
	if p.backend != nil {
		panic("SetBlockSize must be called before Init")
	}
	p.blockSize = blockSize
}

// GetBlockTablesTensor returns a tensor mapping logical positions to physical block IDs.
// Shape: [max_blocks_per_seq, batch_size] — GGML layout ne[0]=max_blocks, ne[1]=batch.
func (p *Paged) GetBlockTablesTensor(ctx ml.Context) ml.Tensor {
	if len(p.curSeqs) == 0 {
		return ctx.Input().Empty(ml.DTypeI32, 0, 0)
	}

	maxBlocks := 0
	for _, seqID := range p.curSeqs {
		if blocks, ok := p.blockTables[seqID]; ok && len(blocks) > maxBlocks {
			maxBlocks = len(blocks)
		}
	}

	if maxBlocks == 0 {
		return ctx.Input().Empty(ml.DTypeI32, len(p.curSeqs), 1)
	}

	blockTableData := make([]int32, len(p.curSeqs)*maxBlocks)
	for i, seqID := range p.curSeqs {
		blocks, ok := p.blockTables[seqID]
		if !ok {
			for j := 0; j < maxBlocks; j++ {
				blockTableData[i*maxBlocks+j] = -1
			}
			continue
		}

		for j := 0; j < maxBlocks; j++ {
			if j < len(blocks) {
				blockTableData[i*maxBlocks+j] = int32(blocks[j])
			} else {
				blockTableData[i*maxBlocks+j] = -1
			}
		}
	}

	return ctx.Input().FromInts(blockTableData, maxBlocks, len(p.curSeqs))
}

func (p *Paged) GetSeqLengthsTensor(ctx ml.Context) ml.Tensor {
	if len(p.curSeqs) == 0 {
		return ctx.Input().Empty(ml.DTypeI32, 0)
	}

	seqLengths := make([]int32, len(p.curSeqs))

	seqMaxPos := make(map[int]int32)
	for i, seqID := range p.curSeqs {
		pos := p.curPositions[i]
		if currentMax, ok := seqMaxPos[seqID]; ok {
			if pos > currentMax {
				seqMaxPos[seqID] = pos + 1
			}
		} else {
			seqMaxPos[seqID] = pos + 1
		}
	}

	for i, seqID := range p.curSeqs {
		seqLengths[i] = seqMaxPos[seqID]
	}

	return ctx.Input().FromInts(seqLengths, len(seqLengths))
}

func (p *Paged) GetBlockSize() int {
	return p.blockSize
}
