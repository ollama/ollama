package server

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"os"
	"sync"
	"testing"
)

func TestSpeedTracker_Median(t *testing.T) {
	s := &speedTracker{}

	// Less than 3 samples returns 0
	s.Record(100)
	s.Record(200)
	if got := s.Median(); got != 0 {
		t.Errorf("expected 0 with < 3 samples, got %f", got)
	}

	// With 3+ samples, returns median
	s.Record(300)
	// Samples: [100, 200, 300] -> median = 200
	if got := s.Median(); got != 200 {
		t.Errorf("expected median 200, got %f", got)
	}

	// Add more samples
	s.Record(50)
	s.Record(250)
	// Samples: [100, 200, 300, 50, 250] sorted = [50, 100, 200, 250, 300] -> median = 200
	if got := s.Median(); got != 200 {
		t.Errorf("expected median 200, got %f", got)
	}
}

func TestSpeedTracker_RollingWindow(t *testing.T) {
	s := &speedTracker{}

	// Add 105 samples (should keep only last 100)
	for i := 0; i < 105; i++ {
		s.Record(float64(i))
	}

	s.mu.Lock()
	if len(s.speeds) != 100 {
		t.Errorf("expected 100 samples, got %d", len(s.speeds))
	}
	// First sample should be 5 (0-4 were dropped)
	if s.speeds[0] != 5 {
		t.Errorf("expected first sample to be 5, got %f", s.speeds[0])
	}
	s.mu.Unlock()
}

func TestSpeedTracker_Concurrent(t *testing.T) {
	s := &speedTracker{}

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(v int) {
			defer wg.Done()
			s.Record(float64(v))
			s.Median() // concurrent read
		}(i)
	}
	wg.Wait()

	// Should not panic, and should have reasonable state
	s.mu.Lock()
	if len(s.speeds) == 0 || len(s.speeds) > 100 {
		t.Errorf("unexpected speeds length: %d", len(s.speeds))
	}
	s.mu.Unlock()
}

func TestStreamHasher_Sequential(t *testing.T) {
	// Create temp file
	f, err := os.CreateTemp("", "streamhasher_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	// Write test data
	data := []byte("hello world, this is a test of the stream hasher")
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}

	// Create parts
	parts := []*blobDownloadPart{
		{Offset: 0, Size: int64(len(data))},
	}

	sh := newStreamHasher(f, parts, int64(len(data)))

	// Mark complete and run
	sh.MarkComplete(0)

	done := make(chan struct{})
	go func() {
		sh.Run()
		close(done)
	}()
	<-done

	// Verify digest
	expected := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	if got := sh.Digest(); got != expected {
		t.Errorf("digest mismatch: got %s, want %s", got, expected)
	}

	if err := sh.Err(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStreamHasher_OutOfOrderCompletion(t *testing.T) {
	// Create temp file
	f, err := os.CreateTemp("", "streamhasher_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	// Write test data (3 parts of 10 bytes each)
	data := []byte("0123456789ABCDEFGHIJabcdefghij")
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}

	// Create 3 parts
	parts := []*blobDownloadPart{
		{N: 0, Offset: 0, Size: 10},
		{N: 1, Offset: 10, Size: 10},
		{N: 2, Offset: 20, Size: 10},
	}

	sh := newStreamHasher(f, parts, int64(len(data)))

	done := make(chan struct{})
	go func() {
		sh.Run()
		close(done)
	}()

	// Mark parts complete out of order: 2, 0, 1
	sh.MarkComplete(2)
	sh.MarkComplete(0) // This should trigger hashing of part 0
	sh.MarkComplete(1) // This should trigger hashing of parts 1 and 2

	<-done

	// Verify digest
	expected := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	if got := sh.Digest(); got != expected {
		t.Errorf("digest mismatch: got %s, want %s", got, expected)
	}
}

func TestStreamHasher_Stop(t *testing.T) {
	// Create temp file
	f, err := os.CreateTemp("", "streamhasher_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	parts := []*blobDownloadPart{
		{Offset: 0, Size: 100},
	}

	sh := newStreamHasher(f, parts, 100)

	done := make(chan struct{})
	go func() {
		sh.Run()
		close(done)
	}()

	// Stop without completing any parts
	sh.Stop()
	<-done

	// Should exit cleanly without error
	if err := sh.Err(); err != nil {
		t.Errorf("unexpected error after Stop: %v", err)
	}
}

func TestStreamHasher_HashedProgress(t *testing.T) {
	// Create temp file with known data
	f, err := os.CreateTemp("", "streamhasher_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	data := make([]byte, 1000)
	rand.Read(data)
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}

	parts := []*blobDownloadPart{
		{N: 0, Offset: 0, Size: 500},
		{N: 1, Offset: 500, Size: 500},
	}

	sh := newStreamHasher(f, parts, 1000)

	// Initially no progress
	if got := sh.Hashed(); got != 0 {
		t.Errorf("expected 0 hashed initially, got %d", got)
	}

	done := make(chan struct{})
	go func() {
		sh.Run()
		close(done)
	}()

	// Complete part 0
	sh.MarkComplete(0)

	// Give hasher time to process
	for i := 0; i < 100; i++ {
		if sh.Hashed() >= 500 {
			break
		}
	}

	// Complete part 1
	sh.MarkComplete(1)
	<-done

	if got := sh.Hashed(); got != 1000 {
		t.Errorf("expected 1000 hashed, got %d", got)
	}
}

func BenchmarkSpeedTracker_Record(b *testing.B) {
	s := &speedTracker{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Record(float64(i))
	}
}

func BenchmarkSpeedTracker_Median(b *testing.B) {
	s := &speedTracker{}
	// Pre-populate with 100 samples
	for i := 0; i < 100; i++ {
		s.Record(float64(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Median()
	}
}

func BenchmarkStreamHasher(b *testing.B) {
	// Create temp file with test data
	f, err := os.CreateTemp("", "streamhasher_bench")
	if err != nil {
		b.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	size := 64 * 1024 * 1024 // 64MB
	data := make([]byte, size)
	rand.Read(data)
	if _, err := f.Write(data); err != nil {
		b.Fatal(err)
	}

	parts := []*blobDownloadPart{
		{Offset: 0, Size: int64(size)},
	}

	b.SetBytes(int64(size))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		sh := newStreamHasher(f, parts, int64(size))
		sh.MarkComplete(0)

		done := make(chan struct{})
		go func() {
			sh.Run()
			close(done)
		}()
		<-done
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
