package chunks

import (
	"fmt"
	"iter"
	"strconv"
	"strings"
)

type Chunk struct {
	Start, End int64
}

func New(start, end int64) Chunk {
	return Chunk{start, end}
}

// ParseRange parses a string in the form "unit=range" where unit is a string
// and range is a string in the form "start-end". It returns the unit and the
// range as a Chunk.
func ParseRange(s string) (unit string, _ Chunk, _ error) {
	unit, r, _ := strings.Cut(s, "=")
	if r == "" {
		return unit, Chunk{}, nil
	}
	c, err := Parse(r)
	if err != nil {
		return "", Chunk{}, err
	}
	return unit, c, err
}

// Parse parses a string in the form "start-end" and returns the Chunk.
func Parse(s string) (Chunk, error) {
	startStr, endStr, _ := strings.Cut(s, "-")
	start, err := strconv.ParseInt(startStr, 10, 64)
	if err != nil {
		return Chunk{}, fmt.Errorf("invalid start: %v", err)
	}
	end, err := strconv.ParseInt(endStr, 10, 64)
	if err != nil {
		return Chunk{}, fmt.Errorf("invalid end: %v", err)
	}
	if start > end {
		return Chunk{}, fmt.Errorf("invalid range %d-%d: start > end", start, end)
	}
	return Chunk{start, end}, nil
}

// Of returns a sequence of contiguous Chunks of size chunkSize that cover
// the range [0, size), in order.
func Of(size, chunkSize int64) iter.Seq[Chunk] {
	return func(yield func(Chunk) bool) {
		for start := int64(0); start < size; start += chunkSize {
			end := min(start+chunkSize-1, size-1)
			if !yield(Chunk{start, end}) {
				break
			}
		}
	}
}

// Count returns the number of Chunks of size chunkSize needed to cover the
// range [0, size).
func Count(size, chunkSize int64) int64 {
	return (size + chunkSize - 1) / chunkSize
}

// Size returns end minus start plus one.
func (c Chunk) Size() int64 {
	return c.End - c.Start + 1
}

// String returns the string representation of the Chunk in the form
// "{start}-{end}".
func (c Chunk) String() string {
	return fmt.Sprintf("%d-%d", c.Start, c.End)
}
