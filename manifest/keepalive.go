package manifest

import (
	"log/slog"
	"os"
	"sync"
	"time"
)

// LayerKeepAliveInterval controls how often in-flight create operations refresh
// blob mtimes so mark-and-sweep GC does not prune them before the manifest is
// written. Tests may override this.
var LayerKeepAliveInterval = 15 * time.Minute

// LayerKeepAlive refreshes tracked blob mtimes until Close is called.
type LayerKeepAlive struct {
	mu      sync.Mutex
	digests map[string]struct{}
	stop    chan struct{}
	done    chan struct{}
}

// StartLayerKeepAlive begins refreshing tracked blobs in the background.
func StartLayerKeepAlive() *LayerKeepAlive {
	k := &LayerKeepAlive{
		digests: make(map[string]struct{}),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}

	go k.loop()
	return k
}

// Track adds a blob digest to the keepalive set.
func (k *LayerKeepAlive) Track(digest string) {
	if k == nil || digest == "" {
		return
	}

	k.mu.Lock()
	k.digests[digest] = struct{}{}
	k.mu.Unlock()
}

// Close stops the background refresh loop.
func (k *LayerKeepAlive) Close() {
	if k == nil {
		return
	}

	close(k.stop)
	<-k.done
}

func (k *LayerKeepAlive) loop() {
	defer close(k.done)

	ticker := time.NewTicker(LayerKeepAliveInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			k.touchTracked()
		case <-k.stop:
			return
		}
	}
}

func (k *LayerKeepAlive) touchTracked() {
	k.mu.Lock()
	digests := make([]string, 0, len(k.digests))
	for digest := range k.digests {
		digests = append(digests, digest)
	}
	k.mu.Unlock()

	for _, digest := range digests {
		if err := touchLayerDigest(digest); err != nil && !os.IsNotExist(err) {
			slog.Debug("layer keepalive touch failed", "digest", digest, "error", err)
		}
	}
}

func touchLayerDigest(digest string) error {
	blob, err := BlobsPath(digest)
	if err != nil {
		return err
	}

	return touchLayer(blob)
}
