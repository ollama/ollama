package gguf

import (
	"bufio"
	"io"
)

type readSeeker struct {
	rs io.ReadSeeker
	br *bufio.Reader
}

func newReadSeeker(rs io.ReadSeeker, size int) *readSeeker {
	return &readSeeker{
		rs: rs,
		br: bufio.NewReaderSize(rs, size),
	}
}

func (b *readSeeker) Read(p []byte) (int, error) {
	return b.br.Read(p)
}

func (b *readSeeker) Seek(offset int64, whence int) (int64, error) {
	if whence == io.SeekCurrent {
		offset -= int64(b.br.Buffered())
	}
	n, err := b.rs.Seek(offset, whence)
	if err != nil {
		return 0, err
	}
	b.br.Reset(b.rs)
	return n, nil
}
