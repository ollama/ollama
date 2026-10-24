//go:build !darwin && !linux

package qwen4_exp

import (
	"fmt"
	"io"
	"os"
)

type hostMapping struct {
	file *os.File
	size int64
}

func openHostMapping(path string) (*hostMapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &hostMapping{file: f, size: info.Size()}, nil
}

func (m *hostMapping) bytes(start, end int64) ([]byte, error) {
	if start < 0 || end < start || end > m.size {
		return nil, fmt.Errorf("mapping range [%d:%d] outside %d bytes", start, end, m.size)
	}
	data := make([]byte, end-start)
	if _, err := m.file.ReadAt(data, start); err != nil && err != io.EOF {
		return nil, err
	}
	return data, nil
}

func (m *hostMapping) close() error {
	return m.file.Close()
}
