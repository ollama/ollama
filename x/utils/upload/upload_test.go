package upload

import (
	"testing"

	"kr.dev/diff"
)

func TestChunks(t *testing.T) {
	const size = 101
	const chunkSize = 10
	var got []Chunk[int]
	var lastN int
	for n, c := range Chunks(size, chunkSize) {
		if n != lastN+1 {
			t.Errorf("n = %d; want %d", n, lastN+1)
		}
		got = append(got, c)
		lastN = n
	}

	want := []Chunk[int]{
		{0, 10},
		{10, 10},
		{20, 10},
		{30, 10},
		{40, 10},
		{50, 10},
		{60, 10},
		{70, 10},
		{80, 10},
		{90, 10},
		{100, 1},
	}

	diff.Test(t, t.Errorf, got, want)
}
