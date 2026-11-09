//go:build !darwin && !linux

package qwen4_exp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestHostMappingRejectsShortRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tensor")
	if err := os.WriteFile(path, []byte{1, 2, 3, 4}, 0o600); err != nil {
		t.Fatal(err)
	}
	m, err := openHostMapping(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.close()
	if err := os.Truncate(path, 2); err != nil {
		t.Fatal(err)
	}
	if _, err := m.bytes(0, 4); err == nil {
		t.Fatal("accepted a short read with zero-filled tensor data")
	}
}
