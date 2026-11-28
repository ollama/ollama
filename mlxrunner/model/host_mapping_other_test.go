//go:build !darwin && !linux

package model

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
	m, err := OpenHostMapping(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := os.Truncate(path, 2); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Bytes(0, 4); err == nil {
		t.Fatal("accepted a short read with zero-filled tensor data")
	}
}

func TestHostMappingCloseIsRepeatable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tensor")
	if err := os.WriteFile(path, []byte{1}, 0o600); err != nil {
		t.Fatal(err)
	}
	m, err := OpenHostMapping(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Bytes(0, 1); err == nil {
		t.Fatal("closed mapping remains readable")
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}
