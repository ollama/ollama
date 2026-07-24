package gguf

import (
	"bufio"
	"encoding/binary"
	"io"
)

type bufferedReader struct {
	offset int64
	rs     io.ReadSeeker
	*bufio.Reader
}

func newBufferedReader(rs io.ReadSeeker, size int) *bufferedReader {
	return &bufferedReader{
		Reader: bufio.NewReaderSize(rs, size),
		rs:     rs,
	}
}

func (rs *bufferedReader) Read(p []byte) (n int, err error) {
	n, err = rs.Reader.Read(p)
	rs.offset += int64(n)
	return n, err
}

func (rs *bufferedReader) ReadUint64() (uint64, error) {
	bts, err := rs.Reader.Peek(8)
	if err != nil {
		return 0, err
	}
	if _, err := rs.Reader.Discard(8); err != nil {
		return 0, err
	}
	rs.offset += 8
	return binary.LittleEndian.Uint64(bts), nil
}

func (rs *bufferedReader) Discard(n int64) error {
	if n <= 0 {
		return nil
	}

	if buffered := min(int64(rs.Reader.Buffered()), n); buffered > 0 {
		discarded, err := rs.Reader.Discard(int(buffered))
		rs.offset += int64(discarded)
		if err != nil {
			return err
		}
		n -= int64(discarded)
	}

	if n == 0 {
		return nil
	}

	if _, err := rs.rs.Seek(n, io.SeekCurrent); err != nil {
		return err
	}
	rs.Reader.Reset(rs.rs)
	rs.offset += n
	return nil
}
