package server

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"io"
	"testing"
)

func TestOrderedWriter_InOrder(t *testing.T) {
	var buf bytes.Buffer
	hasher := sha256.New()
	ow := newOrderedWriter(&buf, hasher)

	// Submit parts in order
	for i := 0; i < 5; i++ {
		data := []byte{byte(i), byte(i), byte(i)}
		if err := ow.Submit(i, data); err != nil {
			t.Fatalf("Submit(%d) failed: %v", i, err)
		}
	}

	// Verify output
	expected := []byte{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}
	if !bytes.Equal(buf.Bytes(), expected) {
		t.Errorf("got %v, want %v", buf.Bytes(), expected)
	}
}

func TestOrderedWriter_OutOfOrder(t *testing.T) {
	var buf bytes.Buffer
	hasher := sha256.New()
	ow := newOrderedWriter(&buf, hasher)

	// Submit parts out of order: 2, 4, 1, 0, 3
	order := []int{2, 4, 1, 0, 3}
	for _, i := range order {
		data := []byte{byte(i), byte(i), byte(i)}
		if err := ow.Submit(i, data); err != nil {
			t.Fatalf("Submit(%d) failed: %v", i, err)
		}
	}

	// Verify output is still in correct order
	expected := []byte{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}
	if !bytes.Equal(buf.Bytes(), expected) {
		t.Errorf("got %v, want %v", buf.Bytes(), expected)
	}
}

func TestOrderedWriter_Digest(t *testing.T) {
	var buf bytes.Buffer
	hasher := sha256.New()
	ow := newOrderedWriter(&buf, hasher)

	// Submit some data
	data := []byte("hello world")
	if err := ow.Submit(0, data); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Verify digest format and correctness
	got := ow.Digest()
	if len(got) != 71 { // "sha256:" + 64 hex chars
		t.Errorf("digest has wrong length: %d, got: %s", len(got), got)
	}
	if got[:7] != "sha256:" {
		t.Errorf("digest doesn't start with sha256: %s", got)
	}

	// Verify it matches expected hash
	expectedHash := sha256.Sum256(data)
	want := "sha256:" + fmt.Sprintf("%x", expectedHash[:])
	if got != want {
		t.Errorf("digest mismatch: got %s, want %s", got, want)
	}
}

func BenchmarkOrderedWriter_InOrder(b *testing.B) {
	// Benchmark throughput when parts arrive in order (best case)
	partSize := 64 * 1024 * 1024 // 64MB parts
	numParts := 4
	data := make([]byte, partSize)
	rand.Read(data)

	b.SetBytes(int64(partSize * numParts))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ow := newOrderedWriter(io.Discard, sha256.New())
		for p := 0; p < numParts; p++ {
			if err := ow.Submit(p, data); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkOrderedWriter_OutOfOrder(b *testing.B) {
	// Benchmark throughput when parts arrive out of order (worst case)
	partSize := 64 * 1024 * 1024 // 64MB parts
	numParts := 4
	data := make([]byte, partSize)
	rand.Read(data)

	// Reverse order: 3, 2, 1, 0
	order := make([]int, numParts)
	for i := 0; i < numParts; i++ {
		order[i] = numParts - 1 - i
	}

	b.SetBytes(int64(partSize * numParts))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ow := newOrderedWriter(io.Discard, sha256.New())
		for _, p := range order {
			if err := ow.Submit(p, data); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkHashThroughput(b *testing.B) {
	// Baseline: raw SHA256 throughput on this machine
	size := 256 * 1024 * 1024 // 256MB
	data := make([]byte, size)
	rand.Read(data)

	b.SetBytes(int64(size))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		h := sha256.New()
		h.Write(data)
		h.Sum(nil)
	}
}

func BenchmarkOrderedWriter_Memory(b *testing.B) {
	// Measure memory when buffering out-of-order parts
	partSize := 64 * 1024 * 1024 // 64MB parts
	numParts := 4
	data := make([]byte, partSize)
	rand.Read(data)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ow := newOrderedWriter(io.Discard, sha256.New())
		// Submit all except part 0 (forces buffering)
		for p := 1; p < numParts; p++ {
			if err := ow.Submit(p, data); err != nil {
				b.Fatal(err)
			}
		}
		// Submit part 0 to flush
		if err := ow.Submit(0, data); err != nil {
			b.Fatal(err)
		}
	}
}
