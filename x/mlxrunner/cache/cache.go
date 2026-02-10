//go:build mlx

package cache

import (
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Cache interface {
	Update(keys, values *mlx.Array) (newKeys, newValues *mlx.Array)
	State() (keys, values *mlx.Array)
	Trim(int) int
	Clone() Cache
	Offset() int
	Len() int
}

type KVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

func (c *KVCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	// Grow buffer if needed
	if c.keys == nil || (prev+L) > c.keys.Dim(2) {
		steps := (c.step + L - 1) / c.step
		newKeys := mlx.Zeros(keys.DType(), B, H, steps*c.step, Dk)
		newValues := mlx.Zeros(values.DType(), B, H, steps*c.step, Dv)

		if c.keys != nil {
			if prev%c.step != 0 {
				c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
				c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
			}
			c.keys.Set(c.keys.Concatenate(2, newKeys))
			c.values.Set(c.values.Concatenate(2, newValues))
		} else {
			c.keys, c.values = newKeys, newValues
		}
	}

	c.offset += L
	c.keys.Set(c.keys.SliceUpdate(keys, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(values, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))

	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice())
}

func (c *KVCache) State() (*mlx.Array, *mlx.Array) {
	if c.offset == c.keys.Dim(2) {
		return c.keys, c.values
	}
	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice())
}

func (c *KVCache) Trim(n int) int {
	n = min(c.offset, n)
	c.offset -= n
	return n
}

func (c *KVCache) Clone() Cache {
	return &KVCache{
		keys:   c.keys.Clone(),
		values: c.values.Clone(),
		offset: c.offset,
		step:   c.step,
	}
}

func (c *KVCache) Offset() int { return c.offset }
func (c *KVCache) Len() int    { return c.offset }

// RotatingKVCache implements sliding window attention with bounded memory
type RotatingKVCache struct {
	maxSize int
	idx     int

	*KVCache
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, KVCache: NewKVCache()}
}

func (c *RotatingKVCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	if keys.Dim(2) > 1 {
		return c.concat(keys, values)
	}
	return c.update(keys, values)
}

func (c *RotatingKVCache) concat(keys, values *mlx.Array) (newK *mlx.Array, newV *mlx.Array) {
	slog.Debug("(*RotatingKVCache).concat", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)
	if c.keys == nil {
		c.keys, c.values = keys, values
	} else {
		if c.idx < c.keys.Dim(2) {
			c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
			c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
		}

		// Trim to max_size to maintain sliding window
		if trim := c.idx - c.maxSize + 1; trim > 0 {
			c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.keys.Dim(2)), mlx.Slice()))
			c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.values.Dim(2)), mlx.Slice()))
		}

		c.keys.Set(c.keys.Concatenate(2, keys))
		c.values.Set(c.values.Concatenate(2, values))
		c.idx = c.keys.Dim(2)
	}

	c.offset += keys.Dim(2)
	c.idx = c.keys.Dim(2)
	return c.keys, c.values
}

func (c *RotatingKVCache) update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	slog.Debug("(*RotatingKVCache).update", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)
	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	// Grow buffer if not yet at max
	if c.keys == nil || (prev >= c.keys.Dim(2) && c.keys.Dim(2) < c.maxSize) {
		newSize := min(c.step, c.maxSize-prev)
		newKeys := mlx.Zeros(keys.DType(), B, H, newSize, Dk)
		newValues := mlx.Zeros(values.DType(), B, H, newSize, Dv)
		if c.keys != nil {
			c.keys.Set(c.keys.Concatenate(2, newKeys))
			c.values.Set(c.values.Concatenate(2, newValues))
		} else {
			c.keys, c.values = newKeys, newValues
		}
		c.idx = prev
	}

	// Trim to max_size to maintain sliding window
	if trim := c.keys.Dim(2) - c.maxSize; trim > 0 {
		c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.keys.Dim(2)), mlx.Slice()))
		c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.values.Dim(2)), mlx.Slice()))
		c.idx = c.maxSize
	}

	// Rotate when hitting max
	if c.idx >= c.maxSize {
		c.idx = 0
	}

	c.keys.Set(c.keys.SliceUpdate(keys, mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.idx+L), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(values, mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.idx+L), mlx.Slice()))

	c.offset += L
	c.idx += L

	validLen := min(c.offset, c.maxSize)
	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, validLen), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, validLen), mlx.Slice())
}

func (c *RotatingKVCache) State() (*mlx.Array, *mlx.Array) {
	if c.offset < c.keys.Dim(2) {
		return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
			c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice())
	}
	return c.keys, c.values
}

func (c *RotatingKVCache) Trim(n int) int {
	n = min(c.offset, n)
	c.offset -= n
	c.idx -= n
	return n
}

func (c *RotatingKVCache) Clone() Cache {
	return &RotatingKVCache{
		maxSize: c.maxSize,
		idx:     c.idx,
		KVCache: c.KVCache.Clone().(*KVCache),
	}
}

func (c *RotatingKVCache) Len() int { return min(c.offset, c.maxSize) }
