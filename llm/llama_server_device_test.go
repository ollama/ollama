package llm

import (
	"context"
	"testing"

	"github.com/ollama/ollama/ml"
)

// A model placed on the host's second device runs in a child restricted to that
// device, which therefore calls it CUDA0 and reports its buffers under that
// name. Looking the device up by its discovery name ("CUDA1") finds nothing and
// reports the device as unused.
func newSecondDeviceRunner(used uint64) *llamaServerRunner {
	gpus := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "1", Library: "CUDA"},
		Name:        "CUDA1",
		TotalMemory: 100 << 30,
		FreeMemory:  100 << 30,
	}}

	return &llamaServerRunner{
		gpus:           gpus,
		deviceLogNames: ml.RunnerDeviceNames(gpus),
		vramByDevice:   map[string]uint64{"CUDA0": used},
	}
}

func TestVRAMByGPUFilteredChildRenumbers(t *testing.T) {
	const used = 15 << 30

	got := newSecondDeviceRunner(used).VRAMByGPU(ml.DeviceID{ID: "1", Library: "CUDA"})
	if got != used {
		t.Errorf("got %d, want %d: the child reports this device as CUDA0", got, uint64(used))
	}
}

// The same mismatch made a device holding a model look completely free, which is
// worse than a wrong readout: it invites placing more work on a full device.
func TestGetDeviceInfosFilteredChildRenumbers(t *testing.T) {
	const used = 15 << 30

	infos := newSecondDeviceRunner(used).GetDeviceInfos(context.Background())
	if len(infos) != 1 {
		t.Fatalf("expected 1 device, got %d", len(infos))
	}

	want := uint64(100<<30) - used
	if infos[0].FreeMemory != want {
		t.Errorf("free memory: got %d, want %d", infos[0].FreeMemory, want)
	}
}

// A device the child never mentioned still reports zero rather than matching
// some other device's figure.
func TestVRAMByGPUUnknownDevice(t *testing.T) {
	r := newSecondDeviceRunner(15 << 30)
	if got := r.VRAMByGPU(ml.DeviceID{ID: "7", Library: "CUDA"}); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}
