package gguf

import (
	"bufio"
	"io"
)

type bufferedReader struct {
	offset int64
	*bufio.Reader
}

func newBufferedReader(rs io.ReadSeeker, size int) *bufferedReader {
	return &bufferedReader{
		Reader: bufio.NewReaderSize(rs, size),
	}
}

func (rs *bufferedReader) Read(p []byte) (n int, err error) {
	n, err = rs.Reader.Read(p)
	rs.offset += int64(n)
	return n, err
}

func (rs *bufferedReader) Discard(n int) (discarded int, err error) {
	discarded, err = rs.Reader.Discard(n)
	rs.offset += int64(discarded)
	return discarded, err
}
