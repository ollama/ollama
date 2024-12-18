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
	// used by model implementations
	Sub(i int) Cache
	Put(ctx ml.Context, key, value ml.Tensor) (ml.Tensor, ml.Tensor, ml.Tensor)

	// cache management
	Close()

	StartForward(ctx ml.Context, seqs []int) error

	CopyPrefix(srcSeq, dstSeq int, len int)
	Remove(seq int, beginIndex, endIndex int) error
}

type Causal struct {
	DType    ml.DType
	Capacity int

	// current forward pass
	curLayer     int
	curPos       int
	curBatchSize int
	curMask      ml.Tensor
	curCellRange cellRange

	// metadata
	cells      []cacheCell
	seqNextPos map[int]int
	cellRanges map[int]cellRange

	// cache data storage
	backend      ml.Backend
	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

type seqCell struct {
	seq int
	pos int
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

func NewCausalCache(backend ml.Backend, capacity int, dtype ml.DType) Cache {
	return &Causal{
		Capacity:   capacity,
		DType:      dtype,
		cells:      make([]cacheCell, capacity),
		seqNextPos: make(map[int]int),
		cellRanges: make(map[int]cellRange),
		backend:    backend,
		// TODO(jessegross): This context is not sized appropriately
		cacheCtx: backend.NewContext(),
	}
}

func (c *Causal) Close() {
	c.cacheCtx.Close()
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

func (c *Causal) StartForward(ctx ml.Context, seqs []int) error {
	c.curBatchSize = len(seqs)

	var err error
	c.curPos, err = c.findStartPos()
	if errors.Is(err, ErrKvCacheFull) {
		c.defrag()
		c.curPos, err = c.findStartPos()
	}
	if err != nil {
		return err
	}

	// TODO(jessegross): There should be a better way to do this
	origSeq := make(map[int]int)
	for k, v := range c.seqNextPos {
		origSeq[k] = v
	}

	c.curCellRange = newRange()
	for i, seq := range seqs {
		c.cells[c.curPos+i] = cacheCell{sequences: []seqCell{{seq: seq, pos: c.seqNextPos[seq]}}}
		c.seqNextPos[seq]++

		ranges := c.cellRanges[seq]
		if c.curPos+i > ranges.max {
			ranges.max = c.curPos + i
		}
		if ranges.max > c.curCellRange.max {
			c.curCellRange.max = ranges.max
		}

		if c.curPos+i < ranges.min {
			ranges.min = c.curPos + i
		}
		if ranges.min < c.curCellRange.min {
			c.curCellRange.min = ranges.min
		}
		c.cellRanges[seq] = ranges
	}

	c.curMask, err = c.buildMask(ctx, origSeq, seqs)

	return err
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

func (c *Causal) findStartPos() (int, error) {
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

func (c *Causal) buildMask(ctx ml.Context, origSeq map[int]int, seqs []int) (ml.Tensor, error) {
	// TODO(jessegross): This makes a number of simplifications such as no padding
	len := c.curCellRange.max - c.curCellRange.min
	mask := make([]float32, c.curBatchSize*len)

	for i := range c.curBatchSize {
		for j := c.curCellRange.min; j < c.curCellRange.max; j++ {
			cellSeq := c.cells[j].findSeq(seqs[i])
			if cellSeq == nil || cellSeq.pos > origSeq[seqs[i]]+i {
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

	// TODO(jessegross):
	// - Need to size the context and compute maxMoves correctly
	// - Just compacts, doesn't optimize placement
	maxMoves := 8192 / (6 * len(c.keys))

	ctx := c.backend.NewContext()
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

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.curBatchSize != int(key.Dim(2)) {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, int(key.Dim(2))))
	}

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(c.DType, key.Dim(0), key.Dim(1), int64(c.Capacity))
		c.values[c.curLayer] = c.cacheCtx.Zeros(c.DType, value.Dim(0), value.Dim(1), int64(c.Capacity))
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, int(key.Stride(2))*c.curPos, int(key.Dim(0)*key.Dim(1)*key.Dim(2)))))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, int(value.Stride(2))*c.curPos, int(value.Dim(0)*value.Dim(1)*value.Dim(2)))))

	len := c.curCellRange.max - c.curCellRange.min

	key = c.keys[c.curLayer].View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(key.Dim(0)), int(key.Stride(1)),
		int(key.Dim(1)), int(key.Stride(2)),
		len,
	)

	value = c.values[c.curLayer].View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(value.Dim(0)), int(value.Stride(1)),
		int(value.Dim(1)), int(value.Stride(2)),
		len,
	)

	return key, value, c.curMask
}

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, len int) {
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
	c.seqNextPos[dstSeq] = len
}

func (c *Causal) shift(seq int, beginIndex, endIndex, offset int) error {
	panic("Shift not yet implemented")
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int) error {
	endIndex = min(endIndex, c.seqNextPos[seq])
	offset := beginIndex - endIndex

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

	if endIndex != c.seqNextPos[seq] {
		err := c.shift(seq, endIndex, c.seqNextPos[seq], offset)
		if err != nil {
			return err
		}
	}

	c.cellRanges[seq] = seqRange
	c.seqNextPos[seq] += offset

	return nil
}
