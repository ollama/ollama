package mlxrunner

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// cacheDiskStore coordinates reservations and eviction across runners. A shared
// namespace lease protects active files; cleanup acquires an exclusive lease.
// OS locks are released on process death, including on Windows.
type cacheDiskStore struct {
	root    string
	dir     string
	lease   *os.File
	limit   int64
	reserve int64
}

func openCacheDiskStore(ctx context.Context, root, namespace string) (*cacheDiskStore, error) {
	if namespace == "" || namespace == "." || namespace == ".." || strings.ContainsAny(namespace, `/\`) {
		return nil, fmt.Errorf("invalid cache namespace")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dir := filepath.Join(root, namespace)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	total, _, err := cacheDiskSpace(root)
	if err != nil {
		return nil, err
	}
	// All models share at most 10% of the filesystem, leaving at least 5%
	// (and never less than 1 GiB) free for other uses. Per-model admission can
	// be stricter; it must never multiply this shared allowance.
	s := &cacheDiskStore{root: root, dir: dir, limit: total / cacheDiskShare, reserve: max(cacheMinimumFree, total/cacheFreeShare)}
	// Serialize cleanup and the exclusive-to-shared transition with other
	// openers and namespace GC. Windows cannot downgrade the lease atomically.
	guard, err := s.lock(ctx)
	if err != nil {
		return nil, err
	}
	defer guard.Close()
	lease, err := os.OpenFile(filepath.Join(dir, ".active"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	// Clear abandoned staging files before taking a shared lease. A live peer
	// prevents the exclusive lock, so its pending work is never removed.
	if locked, err := lockCacheFile(lease, true); err != nil {
		lease.Close()
		return nil, err
	} else if locked {
		entries, _ := os.ReadDir(dir)
		for _, entry := range entries {
			if strings.HasPrefix(entry.Name(), ".pending-") && !strings.HasSuffix(entry.Name(), ".kv") {
				_ = os.Remove(filepath.Join(dir, entry.Name()))
			}
		}
	}
	// Reopen rather than converting locks: LockFileEx cannot atomically downgrade.
	lease.Close()
	lease, err = os.OpenFile(filepath.Join(dir, ".active"), os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if locked, err := lockCacheFile(lease, false); err != nil || !locked {
		lease.Close()
		return nil, fmt.Errorf("cache namespace busy: %v", err)
	}
	s.lease = lease
	return s, nil
}

func (s *cacheDiskStore) close() {
	if s.lease != nil {
		s.lease.Close()
		s.lease = nil
	}
}

func (s *cacheDiskStore) lock(ctx context.Context) (*os.File, error) {
	f, err := os.OpenFile(filepath.Join(s.root, ".allocation"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	ticker := time.NewTicker(checkpointLockInterval)
	defer ticker.Stop()
	for {
		if err := ctx.Err(); err != nil {
			f.Close()
			return nil, err
		}
		ok, err := lockCacheFile(f, true)
		if err != nil {
			f.Close()
			return nil, err
		}
		if ok {
			if err := ctx.Err(); err != nil {
				f.Close()
				return nil, err
			}
			return f, nil
		}
		select {
		case <-ctx.Done():
			f.Close()
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

// reserveWrite admits the whole checkpoint before any large transfer. Separate
// reservation files account for pending bytes even while their tensor files are
// still small. Holding the global lock makes admission atomic across runners.
func (s *cacheDiskStore) reserveWrite(ctx context.Context, path string, bytes int64) (*cacheReservation, error) {
	name, pending := strings.CutPrefix(filepath.Base(path), ".pending-")
	if filepath.Dir(path) != s.dir || !pending || name == "" || strings.ContainsAny(name, `./\`) {
		return nil, fmt.Errorf("invalid cache staging path")
	}
	guard, err := s.lock(ctx)
	if err != nil {
		return nil, err
	}
	defer guard.Close()
	if bytes <= 0 || bytes > s.limit {
		return nil, fmt.Errorf("KV checkpoint exceeds disk budget")
	}
	type victim struct {
		path string
		size int64
		used time.Time
	}
	var victims []victim
	var leases []*os.File
	defer func() {
		for _, f := range leases {
			f.Close()
		}
	}()
	dirs, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	var used, reserved int64
	for _, dir := range dirs {
		if !dir.IsDir() {
			continue
		}
		base := filepath.Join(s.root, dir.Name())
		var removable bool
		if base != s.dir {
			lease, err := os.OpenFile(filepath.Join(base, ".active"), os.O_CREATE|os.O_RDWR, 0o600)
			if err == nil {
				if ok, _ := lockCacheFile(lease, true); ok {
					removable = true
					leases = append(leases, lease)
				} else {
					lease.Close()
				}
			}
		}
		entries, err := os.ReadDir(base)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
				continue
			}
			name := filepath.Join(base, entry.Name())
			if strings.HasSuffix(name, ".reservation") {
				n, err := reclaimCacheReservation(name)
				if err != nil {
					return nil, err
				}
				if n < 0 || n > math.MaxInt64-reserved {
					return nil, fmt.Errorf("invalid KV reservation size")
				}
				reserved += n
			} else if strings.HasSuffix(name, ".kv") {
				info, err := entry.Info()
				if os.IsNotExist(err) {
					continue // the owning runner evicted this file
				}
				if err != nil {
					return nil, err
				}
				if info.Size() < 0 || info.Size() > math.MaxInt64-used {
					return nil, fmt.Errorf("invalid KV storage size")
				}
				used += info.Size()
				if removable {
					victims = append(victims, victim{name, info.Size(), info.ModTime()})
				}
			} else if removable && strings.HasPrefix(entry.Name(), ".pending-") {
				_ = os.Remove(name)
			}
		}
	}
	_, free, err := cacheDiskSpace(s.root)
	if err != nil {
		return nil, err
	}
	if bytes > math.MaxInt64-reserved {
		return nil, fmt.Errorf("KV reservations exceed disk budget")
	}
	required := reserved + bytes
	fits := func() bool {
		return used <= s.limit-required && free >= s.reserve && required <= free-s.reserve
	}
	slices.SortFunc(victims, func(a, b victim) int { return a.used.Compare(b.used) })
	for _, v := range victims {
		if fits() {
			break
		}
		if err := os.Remove(v.path); err == nil {
			used -= v.size
			// Sparse files and unrelated disk users make logical bytes removed
			// an unreliable estimate of physical space reclaimed.
			_, free, err = cacheDiskSpace(s.root)
			if err != nil {
				return nil, err
			}
		}
	}
	// Re-read free space: sparse allocation and unrelated disk users mean logical
	// bytes removed are not a reliable estimate of physical bytes reclaimed.
	_, free, err = cacheDiskSpace(s.root)
	if err != nil {
		return nil, err
	}
	if !fits() {
		return nil, fmt.Errorf("insufficient KV disk budget or free-space reserve")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path+".reservation", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	r := &cacheReservation{f}
	// Byte zero carries the ownership lock. Windows enforces byte-range locks
	// on reads too, so the byte count is stored outside that locked range.
	var data [9]byte
	binary.LittleEndian.PutUint64(data[1:], uint64(bytes))
	if _, err := f.Write(data[:]); err != nil {
		r.Close()
		return nil, err
	}
	if ok, err := lockCacheFile(f, true); err != nil || !ok {
		r.Close()
		return nil, fmt.Errorf("lock KV reservation: %v", err)
	}
	if err := ctx.Err(); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

type cacheReservation struct{ *os.File }

func (r *cacheReservation) Close() error {
	if r == nil {
		return nil
	}
	// Close before removal for Windows. Allocation scans may reclaim it once
	// unlocked; either remover succeeding is sufficient.
	_ = r.File.Close()
	if err := os.Remove(r.Name()); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// reclaimCacheReservation returns the bytes still owned by a live writer,
// or removes an abandoned write. Called under the allocation lock; per-write
// ownership also works when a surviving runner shares the crashed peer's namespace.
func reclaimCacheReservation(path string) (int64, error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	defer f.Close()
	locked, err := lockCacheFile(f, true)
	if err != nil {
		return 0, err
	}
	if locked {
		if err := os.Remove(strings.TrimSuffix(path, ".reservation")); err != nil && !os.IsNotExist(err) {
			return 0, err
		}
		return 0, (&cacheReservation{f}).Close()
	}
	var data [8]byte
	if _, err := f.ReadAt(data[:], 1); err != nil {
		return 0, fmt.Errorf("read KV reservation: %w", err)
	}
	return int64(binary.LittleEndian.Uint64(data[:])), nil
}

// purgeIncompatible drops records from inactive older runner builds. Lock files
// remain in place so a racing process cannot lock a different inode for the same
// namespace. A live old runner keeps its namespace until it exits.
func (s *cacheDiskStore) purgeIncompatible(ctx context.Context) error {
	namespace := filepath.Base(s.dir)
	if len(namespace) != 129 {
		return nil
	}
	guard, err := s.lock(ctx)
	if err != nil {
		return err
	}
	defer guard.Close()
	dirs, err := os.ReadDir(s.root)
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		if !dir.IsDir() || len(dir.Name()) != 129 || dir.Name()[:64] == namespace[:64] {
			continue
		}
		base := filepath.Join(s.root, dir.Name())
		lease, err := os.OpenFile(filepath.Join(base, ".active"), os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			continue
		}
		locked, err := lockCacheFile(lease, true)
		if err != nil || !locked {
			lease.Close()
			continue
		}
		entries, err := os.ReadDir(base)
		if err == nil {
			for _, entry := range entries {
				if !entry.IsDir() && (strings.HasSuffix(entry.Name(), ".kv") || strings.HasPrefix(entry.Name(), ".pending-")) {
					_ = os.Remove(filepath.Join(base, entry.Name()))
				}
			}
		}
		lease.Close()
	}
	return nil
}
