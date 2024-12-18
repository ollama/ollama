package cache

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
)

var ErrNotSupported = errors.New("model does not support operation")

type Cache interface {
	// ** used by model implementations **

	// Returns an instance of the cache for layer 'i'
	Sub(i int) Cache

	// Returns the history of key and value tensors plus a mask
	//
	// The tensors are of shape embed dim, kv heads, batch size
	// The mask is of shape history size, batch size
	Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor)

	// Stores a batch of key and value in the cache
	//
	// The tensors must be of shape embed dim, kv heads, batch size
	Put(ctx ml.Context, key, value ml.Tensor)

	// ** cache management **

	// Closes the cache and frees resources associated with it
	Close()

	// Called before the start of the model's forward pass. For each
	// token in the coming batch, there must be a corresponding entry
	// in positions and seqs.
	StartForward(ctx ml.Context, positions []int32, seqs []int) error

	// Copies tokens in the range [0, len) from srcSeq to dstSeq
	CopyPrefix(srcSeq, dstSeq int, len int32)

	// Removes tokens in the range [beginIndex, endIndex) from seq. Set
	// endIndex to math.MaxInt32 to remove everything starting at beginIndex
	Remove(seq int, beginIndex, endIndex int32) error
}

type Causal struct {
	DType    ml.DType
	Capacity int32

	// current forward pass
	curLayer     int
	curLoc       int
	curBatchSize int
	curMask      ml.Tensor
	curCellRange cellRange

	// metadata
	cells      []cacheCell
	cellRanges map[int]cellRange

	// cache data storage
	backend      ml.Backend
	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

type seqCell struct {
	seq int
	pos int32
}

type cacheCell struct {
	sequences []seqCell
}

type cellRange struct {
	min int
	max int
}

func (cell cacheCell) findSeq(seq int) *seqCell {
	for i := range cell.sequences {
		if cell.sequences[i].seq == seq {
			return &cell.sequences[i]
		}
	}
	return nil
}

func NewCausalCache(backend ml.Backend, dtype ml.DType, capacity int32) Cache {
	return &Causal{
		Capacity:   capacity,
		DType:      dtype,
		cells:      make([]cacheCell, capacity),
		cellRanges: make(map[int]cellRange),
		backend:    backend,
		cacheCtx:   backend.NewContext(),
	}
}

func (c *Causal) Close() {
	c.cacheCtx.Close()
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

func (c *Causal) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	if len(positions) != len(seqs) {
		return fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(positions), len(seqs))
	}

	c.curBatchSize = len(positions)

	if c.curBatchSize < 1 {
		return errors.New("batch size cannot be less than 1")
	}

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

		c.cells[c.curLoc+i] = cacheCell{sequences: []seqCell{{seq: seq, pos: pos}}}

		ranges, ok := c.cellRanges[seq]
		if !ok {
			ranges = newRange()
		}

		if c.curLoc+i > ranges.max {
			ranges.max = c.curLoc + i
		}
		if ranges.max > c.curCellRange.max {
			c.curCellRange.max = ranges.max
		}

		if c.curLoc+i < ranges.min {
			ranges.min = c.curLoc + i
		}
		if ranges.min < c.curCellRange.min {
			c.curCellRange.min = ranges.min
		}
		c.cellRanges[seq] = ranges
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

func (c *Causal) buildMask(ctx ml.Context, positions []int32, seqs []int) (ml.Tensor, error) {
	// TODO(jessegross): This makes a number of simplifications such as no padding,
	// which could be an issue for CUDA graphs and/or flash attention
	len := c.curCellRange.max - c.curCellRange.min + 1
	mask := make([]float32, c.curBatchSize*len)

	for i := range c.curBatchSize {
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			cellSeq := c.cells[j].findSeq(seqs[i])
			if cellSeq == nil || cellSeq.pos > positions[i] {
				mask[i*len+(j-c.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	return ctx.FromFloatSlice(mask, len, c.curBatchSize)
}

func moveCell(ctx ml.Context, objs []ml.Tensor, src, dst, len int) {
	for _, obj := range objs {
		srcView := obj.View(ctx, int(obj.Stride(2))*src, int(obj.Dim(0)*obj.Dim(1))*len)
		dstView := obj.View(ctx, int(obj.Stride(2))*dst, int(obj.Dim(0)*obj.Dim(1))*len)

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
	//   until we see the next one is non-contiguous
	// - Fill up the context with the maximum number of operations it
	//   can hold then compute that and continue with a new context
	//
	// We could try to optimize placement by grouping blocks from
	// the same sequences together but most likely the next forward
	// pass will disrupt this anyways, so the real world benefit
	// seems limited as this time.

	ctx := c.backend.NewContext()

	// For every move, 6 tensors are required per layer (2 views and a
	// copy for each of k and v). For efficiency, we try to group
	// multiple contiguous blocks into a single move. However, if we
	// exceed the maximum number of tensors then we need to compute
	// what we have and start a new batch.
	maxMoves := ctx.MaxTensors() / (6 * len(c.keys))
	moves := 0

	var pendingSrc, pendingDst, pendingLen int

	for dst := range c.cells {
		if len(c.cells[dst].sequences) == 0 {
			for src := len(c.cells) - 1; src > dst; src-- {
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
			ctx = c.backend.NewContext()

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

	for seq := range c.cellRanges {
		seqRange := newRange()

		for i, cell := range c.cells {
			if cell.findSeq(seq) != nil {
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

func (c *Causal) Sub(i int) Cache {
	if i >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, i-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, i-len(c.values)+1)...)
	}

	c.curLayer = i

	return c
}

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	key = key.View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(key.Dim(0)), int(key.Stride(1)),
		int(key.Dim(1)), int(key.Stride(2)),
		int(c.curMask.Dim(0)),
	)

	value = value.View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(value.Dim(0)), int(value.Stride(1)),
		int(value.Dim(1)), int(value.Stride(2)),
		int(c.curMask.Dim(0)),
	)

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
	if c.curBatchSize != int(key.Dim(2)) {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, int(key.Dim(2))))
	}

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(c.DType, key.Dim(0), key.Dim(1), int64(c.Capacity))
		c.values[c.curLayer] = c.cacheCtx.Zeros(c.DType, value.Dim(0), value.Dim(1), int64(c.Capacity))
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, int(key.Stride(2))*c.curLoc, int(key.Dim(0)*key.Dim(1)*key.Dim(2)))))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, int(value.Stride(2))*c.curLoc, int(value.Dim(0)*value.Dim(1)*value.Dim(2)))))
}

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, len int32) {
	seqRange := newRange()

	for i := range c.cells {
		srcCellSeq := c.cells[i].findSeq(srcSeq)
		dstCellSeq := c.cells[i].findSeq(dstSeq)

		if dstCellSeq != nil {
			c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s seqCell) bool { return s.seq == dstSeq })
		}

		if srcCellSeq != nil && srcCellSeq.pos < len {
			c.cells[i].sequences = append(c.cells[i].sequences, seqCell{seq: dstSeq, pos: srcCellSeq.pos})
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
	panic("Shift not yet implemented")
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range c.cells {
		cellSeq := c.cells[i].findSeq(seq)
		if cellSeq != nil {
			if cellSeq.pos >= beginIndex && cellSeq.pos < endIndex {
				c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s seqCell) bool { return s.seq == seq })
			} else {
				if cellSeq.pos >= endIndex {
					cellSeq.pos += offset
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

	if endIndex != math.MaxInt32 {
		err := c.shift(seq, endIndex, offset)
		if err != nil {
			return err
		}
	}

	c.cellRanges[seq] = seqRange

	return nil
}
