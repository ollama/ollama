package causal

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// TODO(jessegross): This needs to have unit tests

type Causal struct {
	DType    ml.DType
	Capacity int32

	// ** current forward pass **

	// the active layer for Get and Put
	curLayer int

	// starting location for data storage for this batch
	curLoc int64

	// size of the current batch
	curBatchSize int64

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

	model        model.Model
	cacheCtx     ml.Context
	keys, values []ml.Tensor
	needsPermute bool
	seqDim       int
	keysShape    []int64
	valuesShape  []int64
	keysStride   []int64
	valuesStride []int64
}

type cacheCell struct {
	pos       int32
	sequences []int
}

type cellRange struct {
	min int64
	max int64
}

func NewCausalCache(model model.Model, dtype ml.DType, capacity int32) cache.Cache {
	return &Causal{
		Capacity:   capacity,
		DType:      dtype,
		cells:      make([]cacheCell, capacity),
		cellRanges: make(map[int]cellRange),
		model:      model,
		cacheCtx:   model.Backend().NewContext(),
	}
}

func (c *Causal) Close() {
	c.cacheCtx.Close()
}

func (c *Causal) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	if len(positions) != len(seqs) {
		return fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(positions), len(seqs))
	}

	c.curBatchSize = int64(len(positions))

	if c.curBatchSize < 1 {
		return errors.New("batch size cannot be less than 1")
	}

	var err error
	c.curLoc, err = c.findStartLoc()
	if errors.Is(err, cache.ErrKvCacheFull) {
		c.defrag()
		c.curLoc, err = c.findStartLoc()
	}
	if err != nil {
		return err
	}

	c.curCellRange = newRange()
	for i, pos := range positions {
		seq := seqs[i]

		c.cells[int(c.curLoc)+i] = cacheCell{pos: pos, sequences: []int{seq}}

		seqRange, ok := c.cellRanges[seq]
		if !ok {
			seqRange = newRange()
		}

		if c.curLoc+int64(i) > seqRange.max {
			seqRange.max = c.curLoc + int64(i)
		}
		if seqRange.max > c.curCellRange.max {
			c.curCellRange.max = seqRange.max
		}

		if c.curLoc+int64(i) < seqRange.min {
			seqRange.min = c.curLoc + int64(i)
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
func (c *Causal) findStartLoc() (int64, error) {
	// TODO(jessegross): We could be a little more efficient by saving the last location
	// so we don't need to rescan the cache for every token

	var start, count int64
	for i := range c.cells {
		if len(c.cells[i].sequences) == 0 {
			count++
			if count >= c.curBatchSize {
				return start, nil
			}
		} else {
			start = int64(i + 1)
			count = 0
		}
	}

	return 0, fmt.Errorf("%w (length: %v)", cache.ErrKvCacheFull, c.Capacity)
}

// Builds a mask of history x batch indicating whether for each token in the batch the
// token in the history should apply. This is based on both the sequence and causality (the
// position of the history is not ahead of the token in the batch).
func (c *Causal) buildMask(ctx ml.Context, positions []int32, seqs []int) (ml.Tensor, error) {
	// TODO(jessegross): This does not do padding, which is required for flash attention
	len := c.curCellRange.max - c.curCellRange.min + 1
	mask := make([]float32, int(c.curBatchSize*len))

	for i := range c.curBatchSize {
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			if !slices.Contains(c.cells[j].sequences, seqs[i]) || c.cells[j].pos > positions[i] {
				mask[int(i*len+(j-c.curCellRange.min))] = float32(math.Inf(-1))
			}
		}
	}

	// TODO the batch concept here and in the K/V aren't consistent so this is likely a bug
	// slog.Info("Generating Mask", "curBatchSize", c.curBatchSize, "len", len, "mask", mask)
	return ctx.FromFloatSlice(mask, 1, int(len), int(c.curBatchSize))
}

func moveCell(ctx ml.Context, objs []ml.Tensor, src, dst, len int64) {
	for _, obj := range objs {
		srcView := obj.View(ctx, obj.Stride(2)*src, []int64{obj.Dim(0) * obj.Dim(1) * len}, nil)
		dstView := obj.View(ctx, obj.Stride(2)*dst, []int64{obj.Dim(0) * obj.Dim(1) * len}, nil)

		ctx.Forward(srcView.Copy(ctx, dstView))
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

	ctx := c.model.Backend().NewContext()

	// For every move, 6 tensors are required per layer (2 views and a
	// copy for each of k and v).
	maxMoves := ctx.MaxTensors() / (6 * len(c.keys))
	moves := 0

	var pendingSrc, pendingDst, pendingLen int64
	src := int64(len(c.cells) - 1)

	for dst := int64(0); dst < src; dst++ {
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
							moveCell(ctx, c.keys, pendingSrc, pendingDst, pendingLen)
							moveCell(ctx, c.values, pendingSrc, pendingDst, pendingLen)
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
			ctx.Compute(nil)
			ctx.Close()
			ctx = c.model.Backend().NewContext()

			moves = 0
		}
	}

	if pendingLen > 0 {
		moveCell(ctx, c.keys, pendingSrc, pendingDst, pendingLen)
		moveCell(ctx, c.values, pendingSrc, pendingDst, pendingLen)
		moves++
	}

	if moves > 0 {
		ctx.Compute(nil)
	}
	ctx.Close()

	// Reset range metadata
	for seq := range c.cellRanges {
		seqRange := newRange()

		for i, cell := range c.cells {
			if slices.Contains(cell.sequences, seq) {
				if int64(i) < seqRange.min {
					seqRange.min = int64(i)
				}
				if int64(i) > seqRange.max {
					seqRange.max = int64(i)
				}
			}
		}

		c.cellRanges[seq] = seqRange
	}
}

func (c *Causal) SetLayer(layer int) {
	if layer >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, layer-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, layer-len(c.values)+1)...)
	}

	c.curLayer = layer
}

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]
	kSeqStride := c.keysStride[c.seqDim]
	vSeqStride := c.valuesStride[c.seqDim]
	kShape := append([]int64{}, c.keysShape...)
	kShape[c.seqDim] = c.curMask.Dim(1)
	kStride := c.keysStride
	vShape := append([]int64{}, c.valuesShape...)
	vStride := c.valuesStride
	vShape[c.seqDim] = c.curMask.Dim(1)

	key = key.View(ctx, kSeqStride*c.curCellRange.min, kShape, kStride)
	value = value.View(ctx, vSeqStride*c.curCellRange.min, vShape, vStride)

	if c.needsPermute {
		key = key.Permute(ctx, 0, 2, 1, 3)
		value = value.Permute(ctx, 0, 2, 1, 3)
	}

	// TODO figure out mask...
	// HACK!  Something's not always lined up right with the mask
	if c.curMask.Dim(2) == 1 {
		return key, value, c.curMask.Permute(ctx, 0, 1, 3, 2)
	}

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor, seqDim int) {
	if seqDim == 2 {
		// TODO - underlying logic doesn't work correctly if the tensor is
		// Batch, kvheads, seq, embed
		// so we flip kvHeads and sequence
		seqDim = 1
		key = key.Permute(ctx, 0, 2, 1, 3)
		value = value.Permute(ctx, 0, 2, 1, 3)
		c.needsPermute = true
	}
	if c.curBatchSize != key.Dim(seqDim) {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, key.Dim(seqDim)))
	}

	nDims := len(key.Shape())
	if seqDim < 0 || seqDim > 3 {
		panic("invalid sequence dimension")
	}

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		if key.Dim(seqDim) == 1 {
			// TODO need to find a solution
			panic("initial sequence of 1 yields invalid stride info")
		}
		// Record stride information on initial request with sequence > 1
		kStride := []int64{}
		vStride := []int64{}
		for d := range nDims - 1 {
			kStride = append(kStride, key.Stride(d))
			vStride = append(vStride, value.Stride(d))
		}
		c.keysStride = kStride
		c.valuesStride = vStride
		c.seqDim = seqDim
		c.keysShape = key.Shape()
		c.valuesShape = value.Shape()
		c.keysShape[seqDim] = -1
		c.valuesShape[seqDim] = -1
	}

	kSize := int64(1)
	vSize := int64(1)
	for d := range nDims {
		if d == seqDim {
			continue
		}
		kSize *= key.Dim(d)
		vSize *= value.Dim(d)
	}

	kSeqStride := c.keysStride[seqDim]
	vSeqStride := c.valuesStride[seqDim]

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(c.DType, kSize, int64(c.Capacity))
		c.values[c.curLayer] = c.cacheCtx.Zeros(c.DType, vSize, int64(c.Capacity))
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, kSeqStride*c.curLoc, []int64{kSize * key.Dim(seqDim)}, nil)))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, vSeqStride*c.curLoc, []int64{vSize * value.Dim(seqDim)}, nil)))
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
			if int64(i) < seqRange.min {
				seqRange.min = int64(i)
			}
			if int64(i) > seqRange.max {
				seqRange.max = int64(i)
			}
		}
	}

	c.cellRanges[dstSeq] = seqRange
}

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	modelShift, ok := c.model.(model.ModelWithShift)
	if !ok {
		return cache.ErrNotSupported
	}

	ctx := c.model.Backend().NewContext()
	defer ctx.Close()

	seqRange := c.cellRanges[seq]
	size := seqRange.max - seqRange.min + 1

	offsets := make([]int32, size)
	for i := range offsets {
		cell := c.cells[int(seqRange.min)+i]

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

		kShape := append([]int64{}, c.keysShape...)
		kShape[c.seqDim] = c.curMask.Dim(1)
		kSeqStride := c.keysStride[c.seqDim]

		key = key.View(ctx, kSeqStride*seqRange.min, kShape, []int64{kSeqStride, size})

		// TODO(jessegross): dequantize once we support data types other than F32 for the cache

		roped, err := modelShift.Shift(ctx, i, key, kShift)
		if err != nil {
			return err
		}

		ctx.Forward(roped.Copy(ctx, key))
	}

	ctx.Compute(nil)

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
				if int64(i) < seqRange.min {
					seqRange.min = int64(i)
				}
				if int64(i) > seqRange.max {
					seqRange.max = int64(i)
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
