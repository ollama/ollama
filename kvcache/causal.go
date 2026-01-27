package kvcache

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

type shiftFn func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error)

// Causal cache stores K and V tensors according to their position in the
// sequence. Returns the history and a mask for attending to past tokens
//
// The tensors are of shape embed dim, kv heads, batch size
// The mask is of shape history size, batch size
type Causal struct {
	DType ml.DType

	// swaWindowSize is the number of tokens that will be included in the mask
	// during attention operations. swaMemorySize is the number of tokens that
	// will be retained in memory for partial prefix caching. Set to math.MaxInt32
	// for unlimited or if sliding window attention is not being used.
	swaWindowSize int32
	swaMemorySize int32

	chunkSize int32

	opts CausalOptions

	// maxBatch is the largest batch that we might receive
	maxBatch int

	// config controls mostly backend-specific optimizations
	config *ml.CacheConfig

	// ** current forward pass **

	// size of the current batch
	curBatchSize int

	// locations for data storage for this batch
	curLoc ml.Tensor

	// mask of the cache as used by this batch
	curMask ml.Tensor

	// the active layer for Get and Put
	curLayer int

	// locations in the cache that are needed for this batch
	curCellRange cellRange

	// curSequences is the sequences corresponding to this pass's entries in the cache
	curSequences []int

	// curPositions is the positions corresponding to this pass's entries in the cache
	curPositions []int32

	// ** cache metadata **

	// for each possible location in the cache, stores the position and set of sequences
	// that reference the data there
	cells []cacheCell

	// maps from sequence to the range of locations where it is stored in the cache
	cellRanges map[int]cellRange

	// ** cache data storage **

	shiftFn      shiftFn
	backend      ml.Backend
	ctxs         map[int]ml.Context
	keys, values map[int]ml.Tensor
}

type cacheCell struct {
	pos       int32
	sequences []int
}

type cellRange struct {
	min int
	max int
}

func NewCausalCache(shift shiftFn) *Causal {
	return &Causal{
		shiftFn: shift,
		ctxs:    make(map[int]ml.Context),
		keys:    make(map[int]ml.Tensor),
		values:  make(map[int]ml.Tensor),
	}
}

func NewSWACache(windowSize int32, shift shiftFn) *Causal {
	return &Causal{
		swaWindowSize: windowSize,
		shiftFn:       shift,
		ctxs:          make(map[int]ml.Context),
		keys:          make(map[int]ml.Tensor),
		values:        make(map[int]ml.Tensor),
	}
}

func NewSWAMemCache(windowSize int32, memorySize int32, shift shiftFn) *Causal {
	return &Causal{
		swaWindowSize: windowSize,
		swaMemorySize: memorySize,
		shiftFn:       shift,
		ctxs:          make(map[int]ml.Context),
		keys:          make(map[int]ml.Tensor),
		values:        make(map[int]ml.Tensor),
	}
}

func NewChunkedAttentionCache(chunkSize int32, shift shiftFn) *Causal {
	return &Causal{
		chunkSize: chunkSize,
		shiftFn:   shift,
		ctxs:      make(map[int]ml.Context),
		keys:      make(map[int]ml.Tensor),
		values:    make(map[int]ml.Tensor),
	}
}

func (c *Causal) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	if c.config == nil {
		var config ml.CacheConfig
		if cc, ok := backend.(ml.BackendCacheConfig); ok {
			config = cc.CacheConfig()
		}
		c.config = &config
	}

	if c.config.CachePadding == 0 {
		c.config.CachePadding = 1
	}

	if c.config.MaskDType == ml.DTypeOther {
		c.config.MaskDType = ml.DTypeF32
	}

	if c.swaWindowSize == 0 {
		c.swaWindowSize = math.MaxInt32
	}
	if c.swaMemorySize == 0 {
		c.swaMemorySize = c.swaWindowSize
	}
	// We will allocate space in the cache for the stop token, which won't be part of a follow on
	// sequence, so allocate an extra token of storage to ensure that we can jump back without
	// causing a cache break. As an optimization, only do this when we have parallel sequences
	// because the extra token will live in the batch buffer and won't get overwritten if we
	// only have a single sequence.
	if c.swaMemorySize != math.MaxInt32 && maxSequences > 1 {
		c.swaMemorySize = max(c.swaMemorySize, c.swaWindowSize+1)
	}
	if int(c.swaMemorySize) >= capacity {
		c.swaMemorySize = math.MaxInt32
	}

	if c.swaMemorySize < c.swaWindowSize {
		panic(fmt.Errorf("sliding window memory (%v) must be at least as large as the window (%v)", c.swaMemorySize, c.swaWindowSize))
	}

	var cacheSize int
	if c.swaMemorySize == math.MaxInt32 {
		cacheSize = maxSequences * capacity
	} else {
		cacheSize = (maxSequences * int(c.swaMemorySize)) + maxBatch
	}
	cacheSize = roundUp(cacheSize, c.config.CachePadding)
	c.cells = make([]cacheCell, cacheSize)

	c.DType = dtype
	c.cellRanges = make(map[int]cellRange)
	c.backend = backend
	c.maxBatch = maxBatch
}

func (c *Causal) SetConfig(config ml.CacheConfig) {
	if c.config != nil {
		panic("config cannot be changed after being previously set, either by the model or backend")
	}

	c.config = &config
}

func (c *Causal) Close() {
	for _, ctx := range c.ctxs {
		ctx.Close()
	}
}

func (c *Causal) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.curBatchSize = len(batch.Positions)
	c.curSequences = batch.Sequences
	c.curPositions = batch.Positions
	c.opts.Except = nil

	var locs []int32
	if !reserve {
		c.updateSlidingWindow()

		var err error
		locs, err = c.findLocs()
		if err != nil {
			return err
		}

		for i, pos := range batch.Positions {
			seq := batch.Sequences[i]
			loc := int(locs[i])

			c.cells[loc] = cacheCell{pos: pos, sequences: []int{seq}}

			seqRange, ok := c.cellRanges[seq]
			if !ok {
				seqRange = newRange()
			}

			seqRange.min = min(seqRange.min, loc)
			c.curCellRange.min = min(c.curCellRange.min, loc)

			seqRange.max = max(seqRange.max, loc)
			c.curCellRange.max = max(c.curCellRange.max, loc)

			c.cellRanges[seq] = seqRange
		}
	} else {
		// If we are reserving memory, don't update any of the cache metadata but set the size
		// to the worst case.
		locs = make([]int32, c.curBatchSize)
		for i := range locs {
			locs[i] = int32(i)
		}
		c.curCellRange.min = 0
		c.curCellRange.max = len(c.cells) - 1
	}

	c.curLoc = ctx.Input().FromInts(locs, len(locs))
	c.curMask = c.buildMask(ctx)

	return nil
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

// Returns a slice of locations where each token in the batch should be stored
func (c *Causal) findLocs() ([]int32, error) {
	loc := make([]int32, 0, c.curBatchSize)

	for i := range c.cells {
		if len(c.cells[i].sequences) == 0 {
			loc = append(loc, int32(i))
			if len(loc) >= c.curBatchSize {
				return loc, nil
			}
		}
	}

	return nil, fmt.Errorf("%w (cache: %v batch: %v)", ErrKvCacheFull, len(c.cells), c.curBatchSize)
}

func (c *Causal) updateSlidingWindow() {
	c.curCellRange = newRange()

	if c.swaMemorySize == math.MaxInt32 {
		for _, seq := range c.curSequences {
			if seqRange, ok := c.cellRanges[seq]; ok {
				c.curCellRange.min = min(c.curCellRange.min, seqRange.min)
				c.curCellRange.max = max(c.curCellRange.max, seqRange.max)
			}
		}

		return
	}

	type lowestPosition struct {
		pos      int32
		curBatch bool
	}

	// create a map of unique sequences to the lowest position in that sequence
	lowestPos := make(map[int]lowestPosition)
	for i := range c.curPositions {
		seq := c.curSequences[i]

		lowest, ok := lowestPos[seq]
		if !ok {
			lowest = lowestPosition{pos: c.curPositions[i], curBatch: true}
		} else if c.curPositions[i] < lowest.pos {
			lowest.pos = c.curPositions[i]
		}

		lowestPos[seq] = lowest
	}

	// for any sequences are not part of this batch, clean up any tokens
	// that are no longer needed after the processing of the previous
	// batch
	for seq, seqRange := range c.cellRanges {
		if _, ok := lowestPos[seq]; !ok {
			var last int32
			for i := seqRange.min; i <= seqRange.max; i++ {
				if slices.Contains(c.cells[i].sequences, seq) {
					last = max(last, c.cells[i].pos)
				}
			}

			lowestPos[seq] = lowestPosition{pos: last + 1, curBatch: false}
		}
	}

	// delete any entries that are beyond the window of the oldest position in the sequence
	for seq, lowest := range lowestPos {
		oldRange, ok := c.cellRanges[seq]
		if !ok {
			continue
		}

		newRange := newRange()

		for i := oldRange.min; i <= oldRange.max; i++ {
			if slices.Contains(c.cells[i].sequences, seq) {
				if c.cells[i].pos < lowest.pos-c.swaMemorySize {
					c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == seq })
				} else {
					newRange.min = min(newRange.min, i)
					newRange.max = max(newRange.max, i)
				}
				if lowest.curBatch && c.cells[i].pos >= lowest.pos-c.swaWindowSize {
					c.curCellRange.min = min(c.curCellRange.min, i)
					c.curCellRange.max = max(c.curCellRange.max, i)
				}
			}
		}

		c.cellRanges[seq] = newRange
	}
}

func roundDown(length, pad int) int {
	return (length / pad) * pad
}

func roundUp(length, pad int) int {
	return ((length + pad - 1) / pad) * pad
}

// Builds a mask of history x batch indicating whether for each token in the batch the
// token in the history should apply. This is based on both the sequence and causality (the
// position of the history is not ahead of the token in the batch).
func (c *Causal) buildMask(ctx ml.Context) ml.Tensor {
	c.curCellRange.min = roundDown(c.curCellRange.min, c.config.CachePadding)
	c.curCellRange.max = roundUp(c.curCellRange.max+1, c.config.CachePadding) - 1

	length := c.curCellRange.max - c.curCellRange.min + 1

	mask := make([]float32, c.curBatchSize*length)

	for i := range c.curBatchSize {
		enabled := !slices.Contains(c.opts.Except, i)
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			if !slices.Contains(c.cells[j].sequences, c.curSequences[i]) ||
				(enabled && c.cells[j].pos > c.curPositions[i]) ||
				c.chunkSize > 0 && c.cells[j].pos < c.curPositions[i]-c.curPositions[i]%c.chunkSize ||
				c.cells[j].pos < c.curPositions[i]-c.swaWindowSize {
				mask[i*length+(j-c.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	maskTensor := ctx.Input().FromFloats(mask, length, c.curBatchSize)

	if c.config.MaskDType != ml.DTypeF32 {
		maskTensor = maskTensor.Cast(ctx, c.config.MaskDType)
	}

	return maskTensor
}

func (c *Causal) SetLayer(layer int) {
	c.curLayer = layer
}

type CausalOptions struct {
	// Enabled controls whether the causal mask is generated for a particular index in a batch
	Except []int
}

// SetCausal disables causal mask generation for a particular range of indicies in
// the current batch for subsequent calls to Get. The state resets for the next forward pass.
func (c *Causal) SetCausal(ctx ml.Context, opts CausalOptions) {
	if !slices.Equal(c.opts.Except, opts.Except) {
		c.opts = opts
		if ctx != nil {
			c.curMask = c.buildMask(ctx)
		}
	}
}

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	kHeadDim := key.Dim(0)
	numKVHeads := key.Dim(1)
	rowSize := key.Stride(2)
	cachedSize := c.curMask.Dim(0)

	key = key.View(ctx, rowSize*c.curCellRange.min,
		kHeadDim, key.Stride(1),
		numKVHeads, key.Stride(2),
		cachedSize,
	)

	if c.config.PermutedV {
		vHeadDim := value.Dim(1)
		elemSize := value.Stride(0)

		value = value.View(ctx, elemSize*c.curCellRange.min,
			cachedSize, value.Stride(1),
			vHeadDim, value.Stride(2),
			numKVHeads,
		)
	} else {
		vHeadDim := value.Dim(0)
		rowSize := value.Stride(2)

		value = value.View(ctx, rowSize*c.curCellRange.min,
			vHeadDim, value.Stride(1),
			numKVHeads, value.Stride(2),
			cachedSize,
		)
	}

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(0)
	vHeadDim := value.Dim(0)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)

	if c.curBatchSize != batchSize {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, batchSize))
	}

	if _, ok := c.ctxs[c.curLayer]; !ok {
		c.ctxs[c.curLayer] = c.backend.NewContextSize(2).Layer(c.curLayer)
	}

	if _, ok := c.keys[c.curLayer]; !ok {
		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, kHeadDim, numKVHeads, len(c.cells))
	}

	if _, ok := c.values[c.curLayer]; !ok {
		if c.config.PermutedV {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, len(c.cells), vHeadDim, numKVHeads)
		} else {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, vHeadDim, numKVHeads, len(c.cells))
		}
	}

	key = key.Reshape(ctx, kHeadDim*numKVHeads, batchSize)
	keyCache := c.keys[c.curLayer]
	keyCache = keyCache.Reshape(ctx, kHeadDim*numKVHeads, len(c.cells))
	ctx.Forward(keyCache.SetRows(ctx, key, c.curLoc))

	if c.config.PermutedV {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, 1, batchSize)
		value = value.Permute(ctx, 2, 0, 1, 3)

		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, 1, len(c.cells), vHeadDim*numKVHeads)

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	} else {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, batchSize)
		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, vHeadDim*numKVHeads, len(c.cells))

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	}
}

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, len int32) {
	seqRange := newRange()

	for i := range c.cells {
		// Remove the contents of dstSeq so that we only have the copied prefix, metadata will be reset at the end
		if slices.Contains(c.cells[i].sequences, dstSeq) {
			c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == dstSeq })
		}

		if slices.Contains(c.cells[i].sequences, srcSeq) && c.cells[i].pos < len {
			c.cells[i].sequences = append(c.cells[i].sequences, dstSeq)
			if i < seqRange.min {
				seqRange.min = i
			}
			if i > seqRange.max {
				seqRange.max = i
			}
		}
	}

	c.cellRanges[dstSeq] = seqRange
}

func (c *Causal) CanResume(seq int, pos int32) bool {
	if c.swaMemorySize == math.MaxInt32 {
		return true
	}

	seqRange, ok := c.cellRanges[seq]
	if !ok {
		return false
	}

	// for sliding window, check that the window of the new sequence is contained in
	// the window of what we are storing
	var first int32 = math.MaxInt32
	var last int32 = -1
	for i := seqRange.min; i <= seqRange.max; i++ {
		if slices.Contains(c.cells[i].sequences, seq) {
			first = min(first, c.cells[i].pos)
			last = max(last, c.cells[i].pos)
		}
	}

	if last == -1 {
		return false
	}

	posWindowStart := max(0, pos-c.swaWindowSize)
	return posWindowStart >= first && pos <= last+1
}

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	if c.shiftFn == nil {
		return ErrNotSupported
	}

	seqRange := c.cellRanges[seq]

	for start := seqRange.min; start <= seqRange.max; start += c.maxBatch {
		size := min(seqRange.max-start+1, c.maxBatch)
		offsets := make([]int32, size)

		var batchFirst, batchLast int

		batchFirst = -1
		for i := range offsets {
			cell := c.cells[start+i]

			if slices.Contains(cell.sequences, seq) && cell.pos >= beginIndex {
				offsets[i] = offset
				if batchFirst < 0 {
					batchFirst = i
				}
				batchLast = i
			}
		}

		if batchFirst < 0 {
			continue
		}

		offsets = offsets[batchFirst : batchLast+1]

		ctx := c.backend.NewContext()
		kShift := ctx.Input().FromInts(offsets, len(offsets))

		for i, key := range c.keys {
			if key == nil {
				continue
			}

			kHeadDim := key.Dim(0)
			numKVHeads := key.Dim(1)
			rowSize := key.Stride(2)

			key = key.View(ctx, rowSize*(start+batchFirst),
				kHeadDim, key.Stride(1),
				numKVHeads, key.Stride(2),
				len(offsets),
			)

			roped, err := c.shiftFn(ctx, i, key, kShift)
			if err != nil {
				ctx.Close()
				return err
			}

			ctx.Forward(roped.Copy(ctx, key))
		}

		ctx.Compute()
		ctx.Close()
	}

	return nil
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
	// TODO(jessegross): We should check to see if removing the middle of the sequence will
	// cause the sliding window to encompass tokens that we no longer have. If so, then we
	// should return an error, which will trigger the runner to evaluate the full history and
	// rebuild the window. However, if we have multimodal inputs in our history, this reuse
	// results in use after free, so we don't do it for now.

	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range c.cells {
		if slices.Contains(c.cells[i].sequences, seq) {
			if c.cells[i].pos >= beginIndex && c.cells[i].pos < endIndex {
				c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == seq })
			} else {
				if c.cells[i].pos >= endIndex {
					if slices.ContainsFunc(c.cells[i].sequences, func(s int) bool { return s != seq }) {
						return errors.New("shifting cells shared by multiple sequences not supported")
					}

					c.cells[i].pos += offset
				}
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}
	}

	if seqRange == newRange() {
		delete(c.cellRanges, seq)
		return nil
	}

	c.cellRanges[seq] = seqRange

	if endIndex != math.MaxInt32 {
		err := c.shift(seq, endIndex+offset, offset)
		if err != nil {
			return err
		}
	}

	return nil
}
