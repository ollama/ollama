package chunks

import (
	"slices"
	"testing"
)

func TestOf(t *testing.T) {
	cases := []struct {
		total     int64
		chunkSize int64
		want      []Chunk
	}{
		{0, 1, nil},
		{1, 1, []Chunk{{0, 0}}},
		{1, 2, []Chunk{{0, 0}}},
		{2, 1, []Chunk{{0, 0}, {1, 1}}},
		{10, 9, []Chunk{{0, 8}, {9, 9}}},
	}

	for _, tt := range cases {
		got := slices.Collect(Of(tt.total, tt.chunkSize))
		if !slices.Equal(got, tt.want) {
			t.Errorf("[%d/%d]: got %v; want %v", tt.total, tt.chunkSize, got, tt.want)
		}
	}
}

func TestSize(t *testing.T) {
	cases := []struct {
		c    Chunk
		want int64
	}{
		{Chunk{0, 0}, 1},
		{Chunk{0, 1}, 2},
		{Chunk{3, 4}, 2},
	}

	for _, tt := range cases {
		got := tt.c.Size()
		if got != tt.want {
			t.Errorf("%v: got %d; want %d", tt.c, got, tt.want)
		}
	}
}

func TestCount(t *testing.T) {
	cases := []struct {
		total     int64
		chunkSize int64
		want      int64
	}{
		{0, 1, 0},
		{1, 1, 1},
		{1, 2, 1},
		{2, 1, 2},
		{10, 9, 2},
	}
	for _, tt := range cases {
		got := Count(tt.total, tt.chunkSize)
		if got != tt.want {
			t.Errorf("[%d/%d]: got %d; want %d", tt.total, tt.chunkSize, got, tt.want)
		}
	}
}
