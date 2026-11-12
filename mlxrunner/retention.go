package mlxrunner

import "github.com/ollama/ollama/mlx"

// snapshotRetention owns a backend's memory observations and eviction policy.
// The scheduler calls trim on the MLX thread. Changes only wakes that thread
// and never performs GPU work itself.
type snapshotRetention interface {
	trim(caches, idle []*prefixCache, workspace, working int64, admitting bool)
	Changes() <-chan struct{}
	Close()
}

func newSnapshotRetention(parallel int) (snapshotRetention, error) {
	if mlx.MetalIsAvailable() {
		// Preserve lazy observation, including retry after a failed read.
		return &metalRetention{parallel: parallel}, nil
	}
	return nil, nil
}
