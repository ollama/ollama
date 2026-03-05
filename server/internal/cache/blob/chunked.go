package blob

import (
	"crypto/sha256"
	"errors"
	"io"
	"os"
)

// Chunk represents a range of bytes in a blob.
type Chunk struct {
	Start int64
	End   int64
}

// Size returns end minus start plus one.
func (c Chunk) Size() int64 {
	return c.End - c.Start + 1
}

// Chunker writes to a blob in chunks.
// Its zero value is invalid. Use [DiskCache.Chunked] to create a new Chunker.
type Chunker struct {
	digest Digest
	size   int64
	f      *os.File // nil means pre-validated
}

// Chunked returns a new Chunker, ready for use storing a blob of the given
// size in chunks.
//
// Use [Chunker.Put] to write data to the blob at specific offsets.
func (c *DiskCache) Chunked(d Digest, size int64) (*Chunker, error) {
	name := c.GetFile(d)
	info, err := os.Stat(name)
	if err == nil && info.Size() == size {
		return &Chunker{}, nil
	}
	f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY, 0o666)
	if err != nil {
		return nil, err
	}
	return &Chunker{digest: d, size: size, f: f}, nil
}

// Put copies chunk.Size() bytes from r to the blob at the given offset,
// merging the data with the existing blob. It returns an error if any. As a
// special case, if r has less than chunk.Size() bytes, Put returns
// io.ErrUnexpectedEOF.
func (c *Chunker) Put(chunk Chunk, d Digest, r io.Reader) error {
	if c.f == nil {
		return nil
	}

	cw := &checkWriter{
		d:    d,
		size: chunk.Size(),
		h:    sha256.New(),
		f:    c.f,
		w:    io.NewOffsetWriter(c.f, chunk.Start),
	}

	_, err := io.CopyN(cw, r, chunk.Size())
	if err != nil && errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// Close closes the underlying file.
func (c *Chunker) Close() error {
	return c.f.Close()
}
