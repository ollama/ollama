package fileutil

import (
	"context"
	"os"
	"time"
)

// Lock serializes cooperative writers. The OS releases it when a process exits.
func Lock(path string) (func(), error) {
	return LockContext(context.Background(), path)
}

// LockContext stops waiting for another writer when ctx is done.
func LockContext(ctx context.Context, path string) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	adopted := false
	defer func() {
		if !adopted {
			_ = file.Close()
		}
	}()
	retry := time.NewTicker(20 * time.Millisecond)
	defer retry.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		locked, err := tryLockFile(file)
		if err != nil {
			return nil, err
		}
		if locked {
			if err := ctx.Err(); err != nil {
				unlockFile(file)
				return nil, err
			}
			adopted = true
			return func() { unlockFile(file); _ = file.Close() }, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-retry.C:
		}
	}
}
