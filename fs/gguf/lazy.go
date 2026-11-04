package gguf

import (
	"encoding/binary"
	"fmt"
	"iter"
)

type lazy[T any] struct {
	count  uint64
	next   func() (T, bool)
	stop   func()
	values []T
	err    error

	// successFunc is called when all values have been successfully read.
	successFunc func() error
}

func newLazy[T any](f *File, fn func() (T, error)) (*lazy[T], error) {
	it := lazy[T]{}
	if f.Version == 1 {
		var count uint32
		if err := binary.Read(f.reader, f.byteOrder, &count); err != nil {
			return nil, err
		}
		it.count = uint64(count)
	} else if err := binary.Read(f.reader, f.byteOrder, &it.count); err != nil {
		return nil, err
	}
	if it.count > uint64(maxInt()) {
		return nil, fmt.Errorf("GGUF item count %d exceeds maximum %d", it.count, maxInt())
	}

	it.values = make([]T, 0)
	it.next, it.stop = iter.Pull(func(yield func(T) bool) {
		for i := range it.count {
			t, err := fn()
			if err != nil {
				it.err = fmt.Errorf("error reading GGUF item %d: %w", i, err)
				return
			}

			it.values = append(it.values, t)
			if !yield(t) {
				break
			}
		}

		if it.successFunc != nil {
			if err := it.successFunc(); err != nil {
				it.err = err
				return
			}
		}
	})

	return &it, nil
}

func (g *lazy[T]) All() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i := range g.count {
			n := int(i)
			if n < len(g.values) {
				if !yield(n, g.values[n]) {
					return
				}
			} else {
				t, ok := g.next()
				if !ok {
					return
				}

				if !yield(n, t) {
					return
				}
			}
		}

		// Resume once after the final yielded item so the producer can run its
		// completion callback and publish any state derived from the section.
		g.rest()
	}
}

func (g *lazy[T]) rest() (collected bool) {
	for {
		_, ok := g.next()
		collected = collected || ok
		if !ok {
			break
		}
	}

	return collected
}

func (g *lazy[T]) Err() error {
	return g.err
}
