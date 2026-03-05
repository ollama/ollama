package bufioutil

import (
	"bufio"
	"io"
)

type BufferedSeeker struct {
	rs io.ReadSeeker
	br *bufio.Reader
}

func NewBufferedSeeker(rs io.ReadSeeker, size int) *BufferedSeeker {
	return &BufferedSeeker{
		rs: rs,
		br: bufio.NewReaderSize(rs, size),
	}
}

func (b *BufferedSeeker) Read(p []byte) (int, error) {
	return b.br.Read(p)
}

func (b *BufferedSeeker) Seek(offset int64, whence int) (int64, error) {
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
