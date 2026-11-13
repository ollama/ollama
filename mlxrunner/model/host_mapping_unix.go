//go:build darwin || linux

package model

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// HostMapping exposes bounded reads from a large tensor blob without copying
// untouched rows into the model's GPU allocation.
type HostMapping struct {
	data []byte
}

// OpenHostMapping maps a tensor blob for bounded, read-only row access.
func OpenHostMapping(path string) (*HostMapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() <= 0 || info.Size() > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("invalid mapping size %d", info.Size())
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(info.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	// PLE row IDs are hashes spread across the table. Disabling sequential
	// readahead keeps untouched pages reclaimable under memory pressure.
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return &HostMapping{data: data}, nil
}

func (m *HostMapping) Bytes(start, end int64) ([]byte, error) {
	if start < 0 || end < start || end > int64(len(m.data)) {
		return nil, fmt.Errorf("mapping range [%d:%d] outside %d bytes", start, end, len(m.data))
	}
	return m.data[start:end], nil
}

func (m *HostMapping) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	err := unix.Munmap(m.data)
	m.data = nil
	return err
}
