package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestPlacedDevices(t *testing.T) {
	metal := ml.DeviceID{ID: "0", Library: "Metal"}
	second := ml.DeviceID{ID: "1", Library: "Metal"}

	// The scheduler reads an empty list as "on the CPU", so a GPU load must
	// report its device: this is what /api/ps attributes the model to.
	got := placedDevices([]ml.DeviceInfo{{DeviceID: metal}})
	if len(got) != 1 || got[0] != metal {
		t.Errorf("single GPU: got %+v, want [%+v]", got, metal)
	}

	// MLX drives only the first device even when more are present, so claiming
	// both would attribute memory to a card holding none.
	got = placedDevices([]ml.DeviceInfo{{DeviceID: metal}, {DeviceID: second}})
	if len(got) != 1 || got[0] != metal {
		t.Errorf("two GPUs: got %+v, want [%+v] only", got, metal)
	}

	if got = placedDevices(nil); got != nil {
		t.Errorf("no GPUs: got %+v, want nil (CPU)", got)
	}
}
