package mlxrunner

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParallelFileReaderConcurrentSmallReads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	if err := os.WriteFile(path, []byte("ab"), 0o600); err != nil {
		t.Fatal(err)
	}

	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	released := false
	reader, err := newParallelFileReader(path, func(int64) {
		entered <- struct{}{}
		<-release
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if !released {
			close(release)
		}
		reader.Close()
	}()

	results := make(chan error, 2)
	for i := range 2 {
		go func() {
			buf := make([]byte, 1)
			_, err := reader.ReadAt(buf, int64(i))
			results <- err
		}()
	}
	for range 2 {
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("small reads were serialized")
		}
	}
	close(release)
	released = true
	for range 2 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
}

func TestParallelFileReaderReportsDuringRange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	size := int64(loadReaderProgressSize + 1)
	if err := file.Truncate(size); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	var reports []int64
	reader, err := newParallelFileReader(path, func(n int64) {
		reports = append(reports, n)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	buf := make([]byte, size)
	if n, err := reader.ReadAt(buf, 0); err != nil || int64(n) != size {
		t.Fatalf("ReadAt = (%d, %v), want (%d, nil)", n, err, size)
	}
	if len(reports) != 2 || reports[0] != loadReaderProgressSize || reports[1] != 1 {
		t.Fatalf("progress reports = %v, want [%d 1]", reports, loadReaderProgressSize)
	}
}
