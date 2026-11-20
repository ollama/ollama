package mlxrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// Cache policy controls optional conversation history, never an executing
// request's working state. Presets change how long saved idle state stays in
// RAM; all share background writes, disk limits and pressure-driven eviction.
const (
	// Automatic disk capacity is 10% of the filesystem. Independently leave
	// 5% (at least 1 GiB) free; a configured capacity never bypasses that reserve.
	cacheDiskShare   = 10
	cacheFreeShare   = 20
	cacheMinimumFree = int64(1 << 30)
	// Each layout starts at 1 GiB and grows to roughly four conversations,
	// each with prompt and reply checkpoints, within the shared disk capacity.
	checkpointInitialLimit = int64(1 << 30)
	checkpointSizeMultiple = 8
	// Bound the in-memory index and work spent decoding checkpoint metadata.
	maxCheckpointFiles    = 512
	maxCheckpointMetadata = 64 << 20
	maxCheckpointTokens   = 1 << 20
	// Balanced returns saved idle RAM after this interval, independently of
	// model keep_alive. Lean releases it immediately; warm waits for pressure.
	checkpointIdleAge = 5 * time.Minute
	// Coalesce rapid next turns before copying tensors. Shutdown and urgent
	// pressure bypass this delay so pending saves can finish promptly.
	checkpointWriteDelay = 100 * time.Millisecond
	// Avoid repeated disk work after a failure and excessive timestamp writes.
	checkpointRetryInterval = time.Minute
	checkpointTouchInterval = time.Minute
	// Retry a nonblocking cross-process file lock while honoring cancellation.
	checkpointLockInterval = 10 * time.Millisecond
	// Opening a busy store may delay persistence, never indefinitely block load.
	checkpointOpenTimeout = 5 * time.Second
	// Leave time for runner cleanup within the parent's graceful-stop deadline.
	checkpointShutdownTimeout = 4 * time.Second
	// Only urgent pressure may wait briefly for an already queued write.
	checkpointPressureTimeout = 100 * time.Millisecond
)

type cachePolicy struct {
	dir       string
	diskLimit int64         // zero selects automatic sizing; disabled is handled before opening
	idleAge   time.Duration // negative retains saved idle RAM until pressure
}

func loadCachePolicy() (cachePolicy, bool, error) {
	p := cachePolicy{dir: envconfig.KvCacheDir(), idleAge: checkpointIdleAge}
	switch envconfig.KvCachePolicy() {
	case "", "balanced":
	case "lean":
		p.idleAge = 0
	case "warm":
		p.idleAge = -1
	default:
		return p, false, fmt.Errorf("OLLAMA_KV_CACHE_POLICY must be balanced, lean or warm")
	}
	if value := envconfig.KvCacheSize(); value != "" {
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil || n < 0 {
			return p, false, fmt.Errorf("OLLAMA_KV_CACHE_SIZE must be a nonnegative byte count")
		}
		if n == 0 {
			return p, false, nil
		}
		p.diskLimit = n
	}
	if p.dir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return p, false, err
		}
		p.dir = filepath.Join(base, "ollama", "kv-v1")
	}
	return p, true, nil
}
