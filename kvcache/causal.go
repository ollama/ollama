package kvcache

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
)

type shiftFn func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error)

// Causal cache stores K and V tensors according to their position in the
// sequence. Returns the history and a mask for attending to past tokens
//
// The tensors are of shape embed dim, kv heads, batch size
// The mask is of shape history size, batch size
type Causal struct {
	DType      ml.DType
	Capacity   int32
	windowSize int32

	// config controls mostly backend-specific optimizations
	config *ml.CacheConfig

	// ** current forward pass **

	// the active layer for Get and Put
	curLayer int

	// starting location for data storage for this batch
	curLoc int

	// size of the current batch
	curBatchSize int

	// mask of the cache as used by this batch
	curMask ml.Tensor

	// locations in the cache that are needed for this batch
	curCellRange cellRange

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
		windowSize: math.MaxInt32,
		shiftFn:    shift,
		ctxs:       make(map[int]ml.Context),
		keys:       make(map[int]ml.Tensor),
		values:     make(map[int]ml.Tensor),
	}
}

func NewSWACache(windowSize int32, shift shiftFn) *Causal {
	return &Causal{
		windowSize: windowSize,
		shiftFn:    shift,
		ctxs:       make(map[int]ml.Context),
		keys:       make(map[int]ml.Tensor),
		values:     make(map[int]ml.Tensor),
	}
}

func (c *Causal) Init(backend ml.Backend, dtype ml.DType, capacity int32) {
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

	if c.config.MaskBatchPadding == 0 {
		c.config.MaskBatchPadding = 1
	}

	if c.config.MaskDType == ml.DTypeOther {
		c.config.MaskDType = ml.DTypeF32
	}

	c.DType = dtype
	c.Capacity = int32(roundUp(int(capacity), c.config.CachePadding))
	c.cells = make([]cacheCell, c.Capacity)
	c.cellRanges = make(map[int]cellRange)
	c.backend = backend
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

func (c *Causal) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	c.curBatchSize = len(positions)

	var err error
	c.curLoc, err = c.findStartLoc()
	if errors.Is(err, ErrKvCacheFull) {
		c.defrag()
		c.curLoc, err = c.findStartLoc()
	}
	if err != nil {
		return err
	}

	c.curCellRange = newRange()
	for i, pos := range positions {
		seq := seqs[i]

		c.cells[c.curLoc+i] = cacheCell{pos: pos, sequences: []int{seq}}

		seqRange, ok := c.cellRanges[seq]
		if !ok {
			seqRange = newRange()
		}

		if c.curLoc+i > seqRange.max {
			seqRange.max = c.curLoc + i
		}
		if seqRange.max > c.curCellRange.max {
			c.curCellRange.max = seqRange.max
		}

		if c.curLoc+i < seqRange.min {
			seqRange.min = c.curLoc + i
		}
		if seqRange.min < c.curCellRange.min {
			c.curCellRange.min = seqRange.min
		}
		c.cellRanges[seq] = seqRange
	}

	c.curMask, err = c.buildMask(ctx, positions, seqs)

	return err
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

// Find the first contiguous block of at least curBatchSize
func (c *Causal) findStartLoc() (int, error) {
	var start, count int
	for i := range c.cells {
		if len(c.cells[i].sequences) == 0 {
			count++
			if count >= c.curBatchSize {
				return start, nil
			}
		} else {
			start = i + 1
			count = 0
		}
	}

	return 0, fmt.Errorf("%w (length: %v)", ErrKvCacheFull, c.Capacity)
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
func (c *Causal) buildMask(ctx ml.Context, positions []int32, seqs []int) (ml.Tensor, error) {
	// Align and pad the two dimensions as required by the backend
	batchSize := roundUp(c.curBatchSize, c.config.MaskBatchPadding)

	c.curCellRange.min = roundDown(c.curCellRange.min, c.config.CachePadding)
	c.curCellRange.max = roundUp(c.curCellRange.max+1, c.config.CachePadding) - 1

	length := c.curCellRange.max - c.curCellRange.min + 1
	mask := make([]float32, batchSize*length)

	for i := range c.curBatchSize {
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			if !slices.Contains(c.cells[j].sequences, seqs[i]) || c.cells[j].pos > positions[i] ||
				c.cells[j].pos < positions[i]-c.windowSize {
				mask[i*length+(j-c.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	// Mask out any padding tokens we added. For padding that we added to the cache history, this
	// has already been masked out because the sequence doesn't match.
	for i := c.curBatchSize * length; i < len(mask); i++ {
		mask[i] = float32(math.Inf(-1))
	}

	maskTensor, err := ctx.Input().FromFloatSlice(mask, length, batchSize)
	if err != nil {
		return nil, err
	}

	if c.config.MaskDType != ml.DTypeF32 {
		out := ctx.Input().Empty(c.config.MaskDType, maskTensor.Shape()...)
		ctx.Forward(maskTensor.Copy(ctx, out))
		maskTensor = out
	}

	return maskTensor, nil
}

func (c *Causal) moveCells(ctx ml.Context, src, dst, len int) {
	for i, key := range c.keys {
		if key == nil {
			continue
		}

		kHeadDim := key.Dim(0)
		numKVHeads := key.Dim(1)
		rowSize := key.Stride(2)

		kSrcView := key.View(ctx, rowSize*src, kHeadDim*numKVHeads*len)
		kDstView := key.View(ctx, rowSize*dst, kHeadDim*numKVHeads*len)

		value := c.values[i]
		var vSrcView, vDstView ml.Tensor
		if c.config.PermutedV {
			vHeadDim := value.Dim(1)
			elemSize := value.Stride(0)

			vSrcView = value.View(ctx, elemSize*src, len, int(c.Capacity)*elemSize, vHeadDim*numKVHeads)
			vDstView = value.View(ctx, elemSize*dst, len, int(c.Capacity)*elemSize, vHeadDim*numKVHeads)
		} else {
			vHeadDim := value.Dim(0)
			rowSize := value.Stride(2)

			vSrcView = value.View(ctx, rowSize*src, vHeadDim*numKVHeads*len)
			vDstView = value.View(ctx, rowSize*dst, vHeadDim*numKVHeads*len)
		}

		ctx.Forward(
			kSrcView.Copy(ctx, kDstView),
			vSrcView.Copy(ctx, vDstView),
		)
	}
}

func (c *Causal) defrag() {
	slog.Debug("defragmenting kv cache")

	// Defrag strategy:
	// - Search for empty holes at the beginning of the cache,
	//   filling them with active data starting at the end
	// - If there are contiguous elements that need to be moved,
	//   combine them into a single operation by holding new moves
	//   until we see that the next one is non-contiguous
	// - Fill up the context with the maximum number of operations it
	//   can hold then compute that and continue with a new context
	//
	// We could try to optimize placement by grouping blocks from
	// the same sequences together but most likely the next forward
	// pass will disrupt this anyways, so the real world benefit
	// seems limited as this time.

	ctx := c.backend.NewContext()

	// For every move, 6 tensors are required per layer (2 views and a
	// copy for each of k and v).
	layers := 0
	for _, key := range c.keys {
		if key == nil {
			continue
		}
		layers++
	}

	maxMoves := ctx.MaxGraphNodes() / (6 * layers)
	moves := 0

	var pendingSrc, pendingDst, pendingLen int
	src := len(c.cells) - 1

	for dst := 0; dst < src; dst++ {
		if len(c.cells[dst].sequences) == 0 {
			for ; src > dst; src-- {
				if len(c.cells[src].sequences) != 0 {
					c.cells[dst] = c.cells[src]
					c.cells[src] = cacheCell{}

					if pendingLen > 0 {
						if src == pendingSrc-pendingLen && dst == pendingDst+pendingLen {
							pendingSrc = src
							pendingLen++
							break
						} else {
							c.moveCells(ctx, pendingSrc, pendingDst, pendingLen)
							moves++
						}
					}

					pendingSrc = src
					pendingDst = dst
					pendingLen = 1

					break
				}
			}
		}

		if moves >= maxMoves {
			ctx.Compute()
			ctx.Close()
			ctx = c.backend.NewContext()

			moves = 0
		}
	}

	if pendingLen > 0 {
		c.moveCells(ctx, pendingSrc, pendingDst, pendingLen)
		moves++
	}

	if moves > 0 {
		ctx.Compute()
	}
	ctx.Close()

	// Reset range metadata
	for seq := range c.cellRanges {
		seqRange := newRange()

		for i, cell := range c.cells {
			if slices.Contains(cell.sequences, seq) {
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}

		c.cellRanges[seq] = seqRange
	}
}

func (c *Causal) SetLayer(layer int) {
	c.curLayer = layer
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
		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, kHeadDim, numKVHeads, int(c.Capacity))
	}

	if _, ok := c.values[c.curLayer]; !ok {
		if c.config.PermutedV {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, int(c.Capacity), vHeadDim, numKVHeads)
		} else {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, vHeadDim, numKVHeads, int(c.Capacity))
		}
	}

	rowSize := c.keys[c.curLayer].Stride(2)
	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, rowSize*c.curLoc, kHeadDim*numKVHeads*batchSize)))

	if c.config.PermutedV {
		elemSize := c.values[c.curLayer].Stride(0)

		value = value.Permute(ctx, 1, 2, 0, 3)
		ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, elemSize*c.curLoc, batchSize, int(c.Capacity)*elemSize, vHeadDim*numKVHeads)))
	} else {
		rowSize := c.values[c.curLayer].Stride(2)

		ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, rowSize*c.curLoc, vHeadDim*numKVHeads*batchSize)))
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

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	if c.shiftFn == nil {
		return ErrNotSupported
	}

	ctx := c.backend.NewContext()
	defer ctx.Close()

	seqRange := c.cellRanges[seq]
	size := seqRange.max - seqRange.min + 1

	offsets := make([]int32, size)
	for i := range offsets {
		cell := c.cells[seqRange.min+i]

		if slices.Contains(cell.sequences, seq) && cell.pos >= beginIndex {
			offsets[i] = offset
		}
	}

	kShift, err := ctx.FromIntSlice(offsets, len(offsets))
	if err != nil {
		return err
	}

	for i, key := range c.keys {
		if key == nil {
			continue
		}

		kHeadDim := key.Dim(0)
		numKVHeads := key.Dim(1)
		rowSize := key.Stride(2)

		key = key.View(ctx, rowSize*seqRange.min,
			kHeadDim, key.Stride(1),
			numKVHeads, key.Stride(2),
			size,
		)

		roped, err := c.shiftFn(ctx, i, key, kShift)
		if err != nil {
			return err
		}

		ctx.Forward(roped.Copy(ctx, key))
	}

	ctx.Compute()

	return nil
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
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
						// TODO(jessegross): Need to be careful about data shared between sequences
						return errors.New("shifting on cells shared by multiple sequences not yet implemented")
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
