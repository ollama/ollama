package blob

import (
	"crypto/sha256"
	"errors"
	"io"
	"os"

	"github.com/ollama/ollama/server/internal/chunks"
)

type Chunk = chunks.Chunk // TODO: move chunks here?

// Chunker writes to a blob in chunks.
// Its zero value is invalid. Use [DiskCache.Chunked] to create a new Chunker.
type Chunker struct {
	digest Digest
	cache  *DiskCache
	size   int64
	f      *os.File // nil means pre-validated
}

// completed returns a single chunk that covers the entire size.
func completed(size int64) []Chunk {
	return []Chunk{{End: size - 1}}
}

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

// Put puts a chunk of data into the chunked file. The chunk must not overlap
// with any previously put chunks. The Digest is the digest of the data in the
// chunk, not the whole file.
//
// If the chunked file is complete, Put will return ErrFileComplete.
func (c *Chunker) Put(chunk Chunk, d Digest, r io.Reader) error {
	if c.f == nil {
		return nil
	}

	w := &checkWriter{
		d:      d,
		offset: chunk.Start,
		size:   chunk.Size(),
		h:      sha256.New(),
		f:      c.f,
	}

	_, err := io.CopyN(w, r, chunk.Size())
	if err != nil && errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// Close closes the underlying file.
func (c *Chunker) Close() error {
	return c.f.Close()
}

// // If the chunk is already in the cache, don't rewrite it.
// fname := c.cache.GetFile(d)
//
// // NOTE: Do not use chunk.String() here, it is not stable, and will
// // cause a cache miss if the implementation changes, causing all users
// // to redownload all chunks unnecessarily.
// actionData := fmt.Sprintf("v1 put %s chunk %d-%d: %s", d, chunk.Start, chunk.End, fname)
//
// action := DigestFromBytes(actionData)
// _, err := c.cache.Get(action)
// if err == nil {
// 	// The chunk is already in the blob, so we don't need to write
// 	// it.
// 	return nil
// }
//
