package kvcache

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// Accelerated implements a free block pool using a bitmap for efficient
// free cell tracking, reducing overhead compared to linear search
type Accelerated struct {
	DType ml.DType

	swaWindowSize int32
	swaMemorySize int32
	chunkSize     int32

	opts CausalOptions
	maxBatch int
	config *ml.CacheConfig

	// current forward pass
	curBatchSize int
	curLoc       ml.Tensor
	curMask      ml.Tensor
	curLayer     int
	curCellRange cellRange
	curSequences []int
	curPositions []int32

	// cache metadata
	cells     []cacheCell
	cellRanges map[int]cellRange

	// ** ACCELERATION: Free Block Pool **
	// Bitmap of free cells for efficient lookup
	// Each bit represents whether a cell is free (1) or occupied (0)
	freeBitmap []uint64
	// Index of the first known free cell (hint for faster search)
	firstFreeHint int
	// Total number of free cells
	freeCount int

	shiftFn shiftFn
	backend ml.Backend
	ctxs    map[int]ml.Context
	keys, values map[int]ml.Tensor
}

const (
	bitsPerWord = 64
)

// NewAcceleratedCache creates a new cache with free block pool optimization
func NewAcceleratedCache(shift shiftFn) *Accelerated {
	return &Accelerated{
		shiftFn: shift,
		ctxs:    make(map[int]ml.Context),
		keys:    make(map[int]ml.Tensor),
		values:  make(map[int]ml.Tensor),
	}
}

func (a *Accelerated) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	if a.config == nil {
		var config ml.CacheConfig
		if cc, ok := backend.(ml.BackendCacheConfig); ok {
			config = cc.CacheConfig()
		}
		a.config = &config
	}

	if a.config.CachePadding == 0 {
		a.config.CachePadding = 1
	}
	if a.config.MaskDType == ml.DTypeOther {
		a.config.MaskDType = ml.DTypeF32
	}

	if a.swaWindowSize == 0 {
		a.swaWindowSize = math.MaxInt32
	}
	if a.swaMemorySize == 0 {
		a.swaMemorySize = a.swaWindowSize
	}
	if a.swaMemorySize != math.MaxInt32 && maxSequences > 1 {
		a.swaMemorySize = max(a.swaMemorySize, a.swaWindowSize+1)
	}
	if int(a.swaMemorySize) >= capacity {
		a.swaMemorySize = math.MaxInt32
	}

	if a.swaMemorySize < a.swaWindowSize {
		panic(fmt.Errorf("sliding window memory (%v) must be at least as large as the window (%v)", a.swaMemorySize, a.swaWindowSize))
	}

	var cacheSize int
	if a.swaMemorySize == math.MaxInt32 {
		cacheSize = maxSequences * capacity
	} else {
		cacheSize = (maxSequences * int(a.swaMemorySize)) + maxBatch
	}
	cacheSize = roundUp(cacheSize, a.config.CachePadding)
	a.cells = make([]cacheCell, cacheSize)

	a.DType = dtype
	a.cellRanges = make(map[int]cellRange)
	a.backend = backend
	a.maxBatch = maxBatch

	// Initialize free block pool - all cells are initially free
	a.initFreePool(cacheSize)
}

// initFreePool initializes the free block pool bitmap
func (a *Accelerated) initFreePool(size int) {
	bitmapWords := (size + bitsPerWord - 1) / bitsPerWord
	a.freeBitmap = make([]uint64, bitmapWords)

	// Set all bits to 1 (all free)
	for i := range a.freeBitmap {
		a.freeBitmap[i] = ^uint64(0)
	}

	// Mark the last word's unused bits as 0 (occupied)
	remainingBits := size % bitsPerWord
	if remainingBits != 0 {
		a.freeBitmap[len(a.freeBitmap)-1] = (1 << remainingBits) - 1
	}

	a.firstFreeHint = 0
	a.freeCount = size
}

// isSet checks if a bit is set (cell is free) in the bitmap
func (a *Accelerated) isSet(index int) bool {
	word := index / bitsPerWord
	bit := uint(index % bitsPerWord)
	return (a.freeBitmap[word] & (1 << bit)) != 0
}

// setBit sets a bit to 1 (mark as free)
func (a *Accelerated) setBit(index int) {
	word := index / bitsPerWord
	bit := uint(index % bitsPerWord)
	a.freeBitmap[word] |= (1 << bit)
	a.freeCount++
	if index < a.firstFreeHint {
		a.firstFreeHint = index
	}
}

// clearBit clears a bit to 0 (mark as occupied)
func (a *Accelerated) clearBit(index int) {
	word := index / bitsPerWord
	bit := uint(index % bitsPerWord)
	a.freeBitmap[word] &^= (1 << bit)
	a.freeCount--

	// Update hint if we just occupied the first free cell
	if index == a.firstFreeHint {
		a.findNextFreeHint()
	}
}

// findNextFreeHint updates the firstFreeHint to the next free cell
func (a *Accelerated) findNextFreeHint() {
	// Search from current hint
	for i := a.firstFreeHint; i < len(a.cells); i++ {
		if a.isSet(i) {
			a.firstFreeHint = i
			return
		}
	}
	// No free cells found
	a.firstFreeHint = len(a.cells)
}

// findLocsAccelerated finds free locations using the bitmap for efficient lookup.
// Uses a hint (firstFreeHint) to reduce scan distance, with wraparound if needed.
func (a *Accelerated) findLocsAccelerated() ([]int32, error) {
	if a.freeCount < a.curBatchSize {
		return nil, fmt.Errorf("%w (cache: %v batch: %v free: %v)", ErrKvCacheFull, len(a.cells), a.curBatchSize, a.freeCount)
	}

	locs := make([]int32, 0, a.curBatchSize)
	found := 0

	// Start from hint and scan forward
	for i := a.firstFreeHint; i < len(a.cells) && found < a.curBatchSize; i++ {
		word := i / bitsPerWord
		bit := uint(i % bitsPerWord)

		// Check if this cell is free
		if a.freeBitmap[word]&(1<<bit) != 0 {
			locs = append(locs, int32(i))
			found++
		}
	}

	if found < a.curBatchSize {
		// Wrap around and search from beginning
		for i := 0; i < a.firstFreeHint && found < a.curBatchSize; i++ {
			word := i / bitsPerWord
			bit := uint(i % bitsPerWord)

			if a.freeBitmap[word]&(1<<bit) != 0 {
				locs = append(locs, int32(i))
				found++
			}
		}
	}

	return locs, nil
}

func (a *Accelerated) SetConfig(config ml.CacheConfig) {
	if a.config != nil {
		panic("config cannot be changed after being previously set")
	}
	a.config = &config
}

func (a *Accelerated) Close() {
	for _, ctx := range a.ctxs {
		ctx.Close()
	}
}

func (a *Accelerated) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	a.curBatchSize = len(batch.Positions)
	a.curSequences = batch.Sequences
	a.curPositions = batch.Positions
	a.opts.Except = nil

	var locs []int32
	if !reserve {
		a.updateSlidingWindow()

		var err error
		locs, err = a.findLocsAccelerated() // Use accelerated version
		if err != nil {
			return err
		}

		for i, pos := range batch.Positions {
			seq := batch.Sequences[i]
			loc := int(locs[i])

			// Mark cell as occupied in bitmap
			a.clearBit(loc)

			a.cells[loc] = cacheCell{pos: pos, sequences: []int{seq}}

			seqRange, ok := a.cellRanges[seq]
			if !ok {
				seqRange = newRange()
			}

			seqRange.min = min(seqRange.min, loc)
			a.curCellRange.min = min(a.curCellRange.min, loc)

			seqRange.max = max(seqRange.max, loc)
			a.curCellRange.max = max(a.curCellRange.max, loc)

			a.cellRanges[seq] = seqRange
		}
	} else {
		locs = make([]int32, a.curBatchSize)
		for i := range locs {
			locs[i] = int32(i)
		}
		a.curCellRange.min = 0
		a.curCellRange.max = len(a.cells) - 1
	}

	a.curLoc = ctx.Input().FromInts(locs, len(locs))
	a.curMask = a.buildMask(ctx)

	return nil
}

func (a *Accelerated) updateSlidingWindow() {
	a.curCellRange = newRange()

	if a.swaMemorySize == math.MaxInt32 {
		for _, seq := range a.curSequences {
			if seqRange, ok := a.cellRanges[seq]; ok {
				a.curCellRange.min = min(a.curCellRange.min, seqRange.min)
				a.curCellRange.max = max(a.curCellRange.max, seqRange.max)
			}
		}
		return
	}

	type lowestPosition struct {
		pos      int32
		curBatch bool
	}

	lowestPos := make(map[int]lowestPosition)
	for i := range a.curPositions {
		seq := a.curSequences[i]
		lowest, ok := lowestPos[seq]
		if !ok {
			lowest = lowestPosition{pos: a.curPositions[i], curBatch: true}
		} else if a.curPositions[i] < lowest.pos {
			lowest.pos = a.curPositions[i]
		}
		lowestPos[seq] = lowest
	}

	for seq, seqRange := range a.cellRanges {
		if _, ok := lowestPos[seq]; !ok {
			var last int32
			for i := seqRange.min; i <= seqRange.max; i++ {
				if slices.Contains(a.cells[i].sequences, seq) {
					last = max(last, a.cells[i].pos)
				}
			}
			lowestPos[seq] = lowestPosition{pos: last + 1, curBatch: false}
		}
	}

	// Delete entries beyond window and mark as free in bitmap
	for seq, lowest := range lowestPos {
		oldRange, ok := a.cellRanges[seq]
		if !ok {
			continue
		}

		newRange := newRange()

		for i := oldRange.min; i <= oldRange.max; i++ {
			if slices.Contains(a.cells[i].sequences, seq) {
				if a.cells[i].pos < lowest.pos-a.swaMemorySize {
					a.cells[i].sequences = slices.DeleteFunc(a.cells[i].sequences, func(s int) bool { return s == seq })
					// Mark as free if no more sequences reference this cell
					if len(a.cells[i].sequences) == 0 {
						a.setBit(i)
					}
				} else {
					newRange.min = min(newRange.min, i)
					newRange.max = max(newRange.max, i)
				}
				if lowest.curBatch && a.cells[i].pos >= lowest.pos-a.swaWindowSize {
					a.curCellRange.min = min(a.curCellRange.min, i)
					a.curCellRange.max = max(a.curCellRange.max, i)
				}
			}
		}

		a.cellRanges[seq] = newRange
	}
}

func (a *Accelerated) buildMask(ctx ml.Context) ml.Tensor {
	a.curCellRange.min = roundDown(a.curCellRange.min, a.config.CachePadding)
	a.curCellRange.max = roundUp(a.curCellRange.max+1, a.config.CachePadding) - 1

	length := a.curCellRange.max - a.curCellRange.min + 1

	mask := make([]float32, a.curBatchSize*length)

	for i := range a.curBatchSize {
		enabled := !slices.Contains(a.opts.Except, i)
		for j := a.curCellRange.min; j <= a.curCellRange.max; j++ {
			if !slices.Contains(a.cells[j].sequences, a.curSequences[i]) ||
				(enabled && a.cells[j].pos > a.curPositions[i]) ||
				a.chunkSize > 0 && a.cells[j].pos < a.curPositions[i]-a.curPositions[i]%a.chunkSize ||
				a.cells[j].pos < a.curPositions[i]-a.swaWindowSize {
				mask[i*length+(j-a.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	maskTensor := ctx.Input().FromFloats(mask, length, a.curBatchSize)

	if a.config.MaskDType != ml.DTypeF32 {
		maskTensor = maskTensor.Cast(ctx, a.config.MaskDType)
	}

	return maskTensor
}

func (a *Accelerated) SetLayer(layer int) {
	a.curLayer = layer
}

func (a *Accelerated) SetCausal(ctx ml.Context, opts CausalOptions) {
	if !slices.Equal(a.opts.Except, opts.Except) {
		a.opts = opts
		if ctx != nil {
			a.curMask = a.buildMask(ctx)
		}
	}
}

func (a *Accelerated) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := a.keys[a.curLayer]
	value := a.values[a.curLayer]

	kHeadDim := key.Dim(0)
	numKVHeads := key.Dim(1)
	rowSize := key.Stride(2)
	cachedSize := a.curMask.Dim(0)

	key = key.View(ctx, rowSize*a.curCellRange.min,
		kHeadDim, key.Stride(1),
		numKVHeads, key.Stride(2),
		cachedSize,
	)

	if a.config.PermutedV {
		vHeadDim := value.Dim(1)
		elemSize := value.Stride(0)

		value = value.View(ctx, elemSize*a.curCellRange.min,
			cachedSize, value.Stride(1),
			vHeadDim, value.Stride(2),
			numKVHeads,
		)
	} else {
		vHeadDim := value.Dim(0)
		rowSize := value.Stride(2)

		value = value.View(ctx, rowSize*a.curCellRange.min,
			vHeadDim, value.Stride(1),
			numKVHeads, value.Stride(2),
			cachedSize,
		)
	}

	return key, value, a.curMask
}

func (a *Accelerated) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(0)
	vHeadDim := value.Dim(0)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)

	if a.curBatchSize != batchSize {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", a.curLayer, a.curBatchSize, batchSize))
	}

	if _, ok := a.ctxs[a.curLayer]; !ok {
		a.ctxs[a.curLayer] = a.backend.NewContextSize(2).Layer(a.curLayer)
	}

	if _, ok := a.keys[a.curLayer]; !ok {
		a.keys[a.curLayer] = a.ctxs[a.curLayer].Zeros(a.DType, kHeadDim, numKVHeads, len(a.cells))
	}

	if _, ok := a.values[a.curLayer]; !ok {
		if a.config.PermutedV {
			a.values[a.curLayer] = a.ctxs[a.curLayer].Zeros(a.DType, len(a.cells), vHeadDim, numKVHeads)
		} else {
			a.values[a.curLayer] = a.ctxs[a.curLayer].Zeros(a.DType, vHeadDim, numKVHeads, len(a.cells))
		}
	}

	key = key.Reshape(ctx, kHeadDim*numKVHeads, batchSize)
	keyCache := a.keys[a.curLayer]
	keyCache = keyCache.Reshape(ctx, kHeadDim*numKVHeads, len(a.cells))
	ctx.Forward(keyCache.SetRows(ctx, key, a.curLoc))

	if a.config.PermutedV {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, 1, batchSize)
		value = value.Permute(ctx, 2, 0, 1, 3)

		valueCache := a.values[a.curLayer]
		valueCache = valueCache.Reshape(ctx, 1, len(a.cells), vHeadDim*numKVHeads)

		ctx.Forward(valueCache.SetRows(ctx, value, a.curLoc))
	} else {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, batchSize)
		valueCache := a.values[a.curLayer]
		valueCache = valueCache.Reshape(ctx, vHeadDim*numKVHeads, len(a.cells))

		ctx.Forward(valueCache.SetRows(ctx, value, a.curLoc))
	}
}

func (a *Accelerated) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	seqRange := newRange()

	for i := range a.cells {
		if slices.Contains(a.cells[i].sequences, dstSeq) {
			// If removing dstSeq makes cell empty, mark as free
			a.cells[i].sequences = slices.DeleteFunc(a.cells[i].sequences, func(s int) bool { return s == dstSeq })
			if len(a.cells[i].sequences) == 0 {
				a.setBit(i)
			}
		}

		if slices.Contains(a.cells[i].sequences, srcSeq) && a.cells[i].pos < prefixLen {
			wasEmpty := len(a.cells[i].sequences) == 0
			a.cells[i].sequences = append(a.cells[i].sequences, dstSeq)
			if wasEmpty {
				a.clearBit(i)
			}
			if i < seqRange.min {
				seqRange.min = i
			}
			if i > seqRange.max {
				seqRange.max = i
			}
		}
	}

	a.cellRanges[dstSeq] = seqRange
}

func (a *Accelerated) CanResume(seq int, pos int32) bool {
	if a.swaMemorySize == math.MaxInt32 {
		return true
	}

	seqRange, ok := a.cellRanges[seq]
	if !ok {
		return false
	}

	var first int32 = math.MaxInt32
	var last int32 = -1
	for i := seqRange.min; i <= seqRange.max; i++ {
		if slices.Contains(a.cells[i].sequences, seq) {
			first = min(first, a.cells[i].pos)
			last = max(last, a.cells[i].pos)
		}
	}

	if last == -1 {
		return false
	}

	posWindowStart := max(0, pos-a.swaWindowSize)
	return posWindowStart >= first && pos <= last+1
}

func (a *Accelerated) shift(seq int, beginIndex, offset int32) error {
	if a.shiftFn == nil {
		return ErrNotSupported
	}

	seqRange := a.cellRanges[seq]

	for start := seqRange.min; start <= seqRange.max; start += a.maxBatch {
		size := min(seqRange.max-start+1, a.maxBatch)
		offsets := make([]int32, size)

		var batchFirst, batchLast int

		batchFirst = -1
		for i := range offsets {
			cell := a.cells[start+i]

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

		ctx := a.backend.NewContext()
		kShift := ctx.Input().FromInts(offsets, len(offsets))

		for i, key := range a.keys {
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

			roped, err := a.shiftFn(ctx, i, key, kShift)
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

func (a *Accelerated) Remove(seq int, beginIndex, endIndex int32) error {
	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range a.cells {
		if slices.Contains(a.cells[i].sequences, seq) {
			if a.cells[i].pos >= beginIndex && a.cells[i].pos < endIndex {
				a.cells[i].sequences = slices.DeleteFunc(a.cells[i].sequences, func(s int) bool { return s == seq })
				// Mark as free if no more sequences reference this cell
				if len(a.cells[i].sequences) == 0 {
					a.setBit(i)
				}
			} else {
				if a.cells[i].pos >= endIndex {
					if slices.ContainsFunc(a.cells[i].sequences, func(s int) bool { return s != seq }) {
						return errors.New("shifting cells shared by multiple sequences not supported")
					}
					a.cells[i].pos += offset
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
		delete(a.cellRanges, seq)
		return nil
	}

	a.cellRanges[seq] = seqRange

	if endIndex != math.MaxInt32 {
		err := a.shift(seq, endIndex+offset, offset)
		if err != nil {
			return err
		}
	}

	return nil
}
