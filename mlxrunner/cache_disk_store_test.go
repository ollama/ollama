package mlxrunner

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Run in a subprocess so the OS, rather than our close path, releases ownership.
func TestCacheDiskStoreCrashHelper(t *testing.T) {
	root := os.Getenv("OLLAMA_TEST_DISK_STORE")
	if root == "" {
		return
	}
	s, err := openCacheDiskStore(t.Context(), root, "shared")
	if err != nil {
		t.Fatal(err)
	}
	s.limit, s.reserve = 1000, 0
	path := filepath.Join(s.dir, ".pending-crash")
	reservation, err := s.reserveWrite(context.Background(), path, 600)
	if err != nil {
		t.Fatal(err)
	}
	defer reservation.Close()
	if err := os.WriteFile(path, []byte("incomplete tensor"), 0o600); err != nil {
		t.Fatal(err)
	}
	fmt.Println("reserved")
	_, _ = bufio.NewReader(os.Stdin).ReadByte()
}

func TestCacheDiskStoreConcurrentStartupCleanup(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	const peers = 8
	var ready, finished sync.WaitGroup
	ready.Add(peers)
	finished.Add(peers)
	results := make(chan error, peers)
	for i := range peers {
		go func() {
			defer finished.Done()
			// Different build IDs race startup GC while every live model owns
			// a checkpoint. The namespace lock must protect all eight files.
			namespace := fmt.Sprintf("%064x-%064x", i, 99)
			s, err := openCacheDiskStore(t.Context(), root, namespace)
			if err != nil {
				ready.Done()
				results <- err
				return
			}
			defer s.close()
			path := filepath.Join(s.dir, "sealed.kv")
			err = os.WriteFile(path, []byte(namespace), 0o600)
			ready.Done()
			ready.Wait()
			if err == nil {
				err = s.purgeIncompatible(ctx)
			}
			if err == nil {
				var data []byte
				data, err = os.ReadFile(path)
				if err == nil && string(data) != namespace {
					err = fmt.Errorf("namespace %s checkpoint changed", namespace)
				}
			}
			results <- err
		}()
	}
	for range peers {
		if err := <-results; err != nil {
			t.Error(err)
		}
	}
	finished.Wait()
	current, err := openCacheDiskStore(t.Context(), root, fmt.Sprintf("%064x-%064x", peers, 99))
	if err != nil {
		t.Fatal(err)
	}
	defer current.close()
	if err := current.purgeIncompatible(ctx); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(root, "*", "*.kv"))
	if err != nil || len(files) != 0 {
		t.Fatalf("incompatible inactive checkpoints survived GC: %v %v", files, err)
	}
}

func TestCacheDiskStorePeerCrash(t *testing.T) {
	root := t.TempDir()
	peer, err := openCacheDiskStore(t.Context(), root, "shared")
	if err != nil {
		t.Fatal(err)
	}
	defer peer.close()
	peer.limit, peer.reserve = 1000, 0
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestCacheDiskStoreCrashHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_DISK_STORE="+root)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer stdin.Close()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || line != "reserved\n" {
		t.Fatalf("child reservation: %q %v", line, err)
	}
	path := filepath.Join(peer.dir, ".pending-survivor")
	if _, err := peer.reserveWrite(ctx, path, 600); err == nil {
		t.Fatal("evicted live peer reservation")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()
	reservation, err := peer.reserveWrite(ctx, path, 600)
	if err != nil {
		t.Fatalf("dead peer blocks surviving runner: %v", err)
	}
	defer reservation.Close()
	for _, name := range []string{".pending-crash", ".pending-crash.reservation"} {
		if _, err := os.Stat(filepath.Join(peer.dir, name)); !os.IsNotExist(err) {
			t.Fatalf("abandoned file survived: %s: %v", name, err)
		}
	}
}

func TestCacheDiskStoreReservationsAndActiveProtection(t *testing.T) {
	root := t.TempDir()
	a, err := openCacheDiskStore(t.Context(), root, "a")
	if err != nil {
		t.Fatal(err)
	}
	defer a.close()
	b, err := openCacheDiskStore(t.Context(), root, "b")
	if err != nil {
		t.Fatal(err)
	}
	defer b.close()
	a.limit, b.limit = 1000, 1000
	a.reserve, b.reserve = 0, 0
	ctx := context.Background()
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := a.reserveWrite(cancelled, filepath.Join(a.dir, ".pending-cancelled"), 600); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled reservation: %v", err)
	}
	pending, err := a.reserveWrite(ctx, filepath.Join(a.dir, ".pending-a"), 600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.reserveWrite(ctx, filepath.Join(b.dir, ".pending-b"), 600); err == nil {
		t.Fatal("concurrent reservations exceeded budget")
	}
	if err := pending.Close(); err != nil {
		t.Fatal(err)
	}
	saved := filepath.Join(a.dir, "saved.kv")
	if err := os.WriteFile(saved, make([]byte, 600), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := b.reserveWrite(ctx, filepath.Join(b.dir, ".pending-b"), 600); err == nil {
		t.Fatal("evicted active peer checkpoint")
	}
	if _, err := os.Stat(saved); err != nil {
		t.Fatal("active peer file removed")
	}
	a.close()
	reservation, err := b.reserveWrite(ctx, filepath.Join(b.dir, ".pending-b"), 600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(saved); !os.IsNotExist(err) {
		t.Fatalf("inactive checkpoint was not evicted: %v", err)
	}
	reservation.Close()
}

func TestCacheDiskStoreCrashCleanupAndBuildChange(t *testing.T) {
	root := t.TempDir()
	oldNamespace := strings.Repeat("a", 64) + "-" + strings.Repeat("b", 64)
	old, err := openCacheDiskStore(t.Context(), root, oldNamespace)
	if err != nil {
		t.Fatal(err)
	}
	defer old.close()
	staging := filepath.Join(old.dir, ".pending-interrupted")
	saved := staging + ".kv"
	for _, path := range []string{staging, staging + ".reservation", saved} {
		if err := os.WriteFile(path, []byte("checkpoint"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// Reopening while a peer is alive must preserve its staging files.
	peer, err := openCacheDiskStore(t.Context(), root, oldNamespace)
	if err != nil {
		t.Fatal(err)
	}
	peer.close()
	if _, err := os.Stat(staging); err != nil {
		t.Fatalf("removed live staging file: %v", err)
	}
	old.close()
	old, err = openCacheDiskStore(t.Context(), root, oldNamespace)
	if err != nil {
		t.Fatal(err)
	}
	defer old.close()
	for _, path := range []string{staging, staging + ".reservation"} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("abandoned staging file survived: %s: %v", path, err)
		}
	}
	current, err := openCacheDiskStore(t.Context(), root, strings.Repeat("c", 64)+"-"+strings.Repeat("b", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer current.close()
	if err := current.purgeIncompatible(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(saved); err != nil {
		t.Fatalf("purged active older runner: %v", err)
	}
	old.close()
	if err := current.purgeIncompatible(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(saved); !os.IsNotExist(err) {
		t.Fatalf("incompatible cache survived unload: %v", err)
	}
}

func TestCacheDiskStoreConcurrentReservations(t *testing.T) {
	root := t.TempDir()
	const peers = 8
	stores := make([]*cacheDiskStore, peers)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	started := make(chan error, peers)
	start := make(chan struct{})
	for i := range stores {
		go func() {
			<-start
			s, err := openCacheDiskStore(ctx, root, "same-model")
			if err == nil {
				s.limit, s.reserve = 1000, 0
				stores[i] = s
			}
			started <- err
		}()
	}
	close(start)
	for range peers {
		if err := <-started; err != nil {
			t.Error(err)
		}
	}
	for _, s := range stores {
		if s != nil {
			defer s.close()
		}
	}
	if t.Failed() {
		return
	}
	results := make(chan *cacheReservation, peers)
	for i, s := range stores {
		go func() {
			r, _ := s.reserveWrite(ctx, filepath.Join(s.dir, fmt.Sprintf(".pending-%d", i)), 600)
			results <- r
		}()
	}
	var admitted []*cacheReservation
	for range peers {
		if r := <-results; r != nil {
			admitted = append(admitted, r)
			defer r.Close()
		}
	}
	if len(admitted) != 1 {
		t.Fatalf("admitted %d writes that collectively exceed the shared quota", len(admitted))
	}
}

func TestCacheDiskStoreStartupWaitsForAllocation(t *testing.T) {
	root := t.TempDir()
	s, err := openCacheDiskStore(t.Context(), root, "same-model")
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	guard, err := s.lock(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer guard.Close()
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	peer, err := openCacheDiskStore(ctx, root, "same-model")
	if peer != nil {
		peer.close()
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("startup bypassed namespace coordination: %v", err)
	}
	guard.Close()
	peer, err = openCacheDiskStore(t.Context(), root, "same-model")
	if err != nil {
		t.Fatalf("startup after releasing allocation: %v", err)
	}
	peer.close()
}

func TestCacheDiskStoreRejectsEscapingPaths(t *testing.T) {
	root := t.TempDir()
	for _, namespace := range []string{"", ".", "..", "../outside", `..\outside`, "/absolute"} {
		if s, err := openCacheDiskStore(t.Context(), root, namespace); err == nil {
			s.close()
			t.Fatalf("accepted namespace %q", namespace)
		}
	}
	s, err := openCacheDiskStore(t.Context(), root, "model")
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	for _, path := range []string{filepath.Join(root, ".pending-outside"), filepath.Join(s.dir, "saved.kv"), filepath.Join(s.dir, ".pending-1.reservation")} {
		if r, err := s.reserveWrite(context.Background(), path, 1); err == nil {
			r.Close()
			t.Fatalf("accepted staging path %q", path)
		}
	}
}
