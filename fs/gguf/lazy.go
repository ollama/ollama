package gguf

import (
	"encoding/binary"
	"iter"
	"log/slog"
)

type lazy[T any] struct {
	count  uint64
	next   func() (T, bool)
	stop   func()
	values []T

	// successFunc is called when all values have been successfully read.
	successFunc func() error
}

func newLazy[T any](f *File, fn func() (T, error)) (*lazy[T], error) {
	it := lazy[T]{}
	if err := binary.Read(f.reader, binary.LittleEndian, &it.count); err != nil {
		return nil, err
	}

	it.values = make([]T, 0)
	it.next, it.stop = iter.Pull(func(yield func(T) bool) {
		for i := range it.count {
			t, err := fn()
			if err != nil {
				slog.Error("error reading tensor", "index", i, "error", err)
				return
			}

			it.values = append(it.values, t)
			if !yield(t) {
				break
			}
		}

		if it.successFunc != nil {
			it.successFunc()
		}
	})

	return &it, nil
}

func (g *lazy[T]) Values() iter.Seq[T] {
	return func(yield func(T) bool) {
		for _, v := range g.All() {
			if !yield(v) {
				break
			}
		}
	}
}

func (g *lazy[T]) All() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i := range int(g.count) {
			if i < len(g.values) {
				if !yield(i, g.values[i]) {
					break
				}
			} else {
				t, ok := g.next()
				if !ok {
					break
				}

				if !yield(i, t) {
					break
				}
			}
		}
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
