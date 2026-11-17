package mlxrunner

import (
	"math"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/mlx"
)

type retentionMonitor struct {
	*discover.MemoryMonitor
}

func newCUDARetention() (*cudaRetention, error) {
	if !mlx.CUDAIsAvailable() || mlx.GPUCount() != 1 {
		return nil, nil
	}
	integrated, err := mlx.IntegratedGPU()
	if err != nil || !integrated {
		return nil, err
	}
	monitor, err := discover.NewMemoryMonitor()
	if err != nil {
		return nil, err
	}
	o := &retentionMonitor{MemoryMonitor: monitor}
	if _, err := o.Read(); err != nil {
		o.Close()
		return nil, err
	}
	return &cudaRetention{memory: o}, nil
}

// Device observations stay on the MLX thread; OS collectors never call MLX.
func (m *retentionMonitor) Read() (retentionObservation, error) {
	device, err := mlx.DeviceMemoryBudget()
	if err != nil {
		return retentionObservation{}, err
	}
	limit, err := mlx.MemoryLimit()
	if err != nil {
		return retentionObservation{}, err
	}
	host, err := m.MemoryMonitor.Read()
	if err != nil {
		return retentionObservation{}, err
	}
	return sharedRetentionObservation(host, device, int64(limit)), nil
}

// sharedRetentionObservation translates host and driver facts for a single
// integrated GPU. All constraints apply to the same shared allocations, so they
// overlap rather than add. A discrete GPU needs its own translation from VRAM
// telemetry; host pressure and host commitment cannot be used unchanged there.
func sharedRetentionObservation(host discover.MemoryHeadroom, device *mlx.MemoryBudget, limit int64) retentionObservation {
	b := retentionBounds{physical: host.AvailableRAM, commit: math.MaxInt64, device: math.MaxInt64, freeTarget: math.MaxInt64}
	if host.FreeMemoryAboveTarget == nil {
		b.physical -= max(1<<30, host.RAMLimit/20)
	} else {
		b.freeTarget = *host.FreeMemoryAboveTarget
	}
	if host.AvailableCommit != nil {
		b.commit = *host.AvailableCommit - 1<<30
	}
	if device != nil {
		b.device = device.Budget - device.Usage - max(256<<20, device.Budget/20)
	}
	return retentionObservation{
		sampledAt: host.SampledAt, pressure: host.Pressure, reclaimBytes: host.ReclaimBytes,
		limit: limit, available: b,
	}
}
