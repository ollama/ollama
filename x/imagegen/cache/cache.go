//go:build mlx

package cache

import "github.com/ollama/ollama/x/imagegen/mlx"

type Cache interface {
	Update(k, v *mlx.Array, seqLen int) (*mlx.Array, *mlx.Array)
	Offset() int
	Len() int
	State() []*mlx.Array
}

type KVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

func (c *KVCache) Update(k, v *mlx.Array, seqLen int) (*mlx.Array, *mlx.Array) {
	prev := c.offset
	shape := k.Shape()
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	// Grow buffer if needed
	if c.keys == nil || (prev+seqLen) > int(c.keys.Shape()[2]) {
		nSteps := (c.step + seqLen - 1) / c.step
		newK := mlx.Zeros([]int32{B, H, int32(nSteps * c.step), Dk}, k.Dtype())
		newV := mlx.Zeros([]int32{B, H, int32(nSteps * c.step), Dv}, v.Dtype())

		if c.keys != nil {
			if prev%c.step != 0 {
				c.keys = mlx.Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, int32(prev), Dk})
				c.values = mlx.Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, int32(prev), Dv})
			}
			c.keys = mlx.Concatenate([]*mlx.Array{c.keys, newK}, 2)
			c.values = mlx.Concatenate([]*mlx.Array{c.values, newV}, 2)
		} else {
			c.keys, c.values = newK, newV
		}
	}

	c.offset += seqLen
	c.keys = mlx.SliceUpdateInplace(c.keys, k, []int32{0, 0, int32(prev), 0}, []int32{B, H, int32(c.offset), Dk})
	c.values = mlx.SliceUpdateInplace(c.values, v, []int32{0, 0, int32(prev), 0}, []int32{B, H, int32(c.offset), Dv})

	return mlx.Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.offset), Dk}),
		mlx.Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, int32(c.offset), Dv})
}

func (c *KVCache) State() []*mlx.Array {
	if c.keys == nil {
		return nil
	}
	return []*mlx.Array{c.keys, c.values}
}

func (c *KVCache) Offset() int { return c.offset }
func (c *KVCache) Len() int    { return c.offset }

// RotatingKVCache implements sliding window attention with bounded memory
type RotatingKVCache struct {
	keys, values *mlx.Array
	offset       int
	maxSize      int
	step         int
	idx          int
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, step: 256}
}

func (c *RotatingKVCache) Update(k, v *mlx.Array, seqLen int) (*mlx.Array, *mlx.Array) {
	if seqLen > 1 {
		return c.updateConcat(k, v, seqLen)
	}
	return c.updateInPlace(k, v)
}

func (c *RotatingKVCache) updateInPlace(k, v *mlx.Array) (*mlx.Array, *mlx.Array) {
	shape := k.Shape()
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	// Grow buffer if not yet at max
	if c.keys == nil || (c.idx >= int(c.keys.Shape()[2]) && int(c.keys.Shape()[2]) < c.maxSize) {
		var cap int
		if c.keys != nil {
			cap = int(c.keys.Shape()[2])
		}
		newSize := min(c.step, c.maxSize-cap)
		newK := mlx.Zeros([]int32{B, H, int32(newSize), Dk}, k.Dtype())
		newV := mlx.Zeros([]int32{B, H, int32(newSize), Dv}, v.Dtype())
		if c.keys != nil {
			c.keys = mlx.Concatenate([]*mlx.Array{c.keys, newK}, 2)
			c.values = mlx.Concatenate([]*mlx.Array{c.values, newV}, 2)
		} else {
			c.keys, c.values = newK, newV
		}
	}

	// Rotate when hitting max
	if c.idx >= c.maxSize {
		c.idx = 0
	}

	c.keys = mlx.SliceUpdateInplace(c.keys, k, []int32{0, 0, int32(c.idx), 0}, []int32{B, H, int32(c.idx + 1), Dk})
	c.values = mlx.SliceUpdateInplace(c.values, v, []int32{0, 0, int32(c.idx), 0}, []int32{B, H, int32(c.idx + 1), Dv})

	c.offset++
	c.idx++

	validLen := int32(min(c.offset, c.maxSize))
	return mlx.Slice(c.keys, []int32{0, 0, 0, 0}, []int32{B, H, validLen, Dk}),
		mlx.Slice(c.values, []int32{0, 0, 0, 0}, []int32{B, H, validLen, Dv})
}

func (c *RotatingKVCache) updateConcat(k, v *mlx.Array, seqLen int) (*mlx.Array, *mlx.Array) {
	shape := k.Shape()
	B, H, Dk := shape[0], shape[1], shape[3]
	Dv := v.Shape()[3]

	if c.keys == nil {
		c.keys, c.values = k, v
	} else {
		c.keys = mlx.Concatenate([]*mlx.Array{c.keys, k}, 2)
		c.values = mlx.Concatenate([]*mlx.Array{c.values, v}, 2)
	}
	c.offset += seqLen

	// Trim to max_size to maintain sliding window
	cap := int(c.keys.Shape()[2])
	if trim := cap - c.maxSize; trim > 0 {
		c.keys = mlx.Slice(c.keys, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dk})
		c.values = mlx.Slice(c.values, []int32{0, 0, int32(trim), 0}, []int32{B, H, int32(cap), Dv})
	}

	c.idx = int(c.keys.Shape()[2])
	return c.keys, c.values
}

func (c *RotatingKVCache) State() []*mlx.Array {
	if c.keys == nil {
		return nil
	}
	return []*mlx.Array{c.keys, c.values}
}

func (c *RotatingKVCache) Offset() int { return c.offset }
func (c *RotatingKVCache) Len() int    { return min(c.offset, c.maxSize) }
