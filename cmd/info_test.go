package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPrettyPrintSupportedGPUsNoneDetected(t *testing.T) {
	var buf bytes.Buffer
	prettyPrintSupportedGPUs(&buf, " ", api.InfoResponse{})

	out := buf.String()
	if !strings.Contains(out, "Supported GPUs:") {
		t.Errorf("missing heading, got %q", out)
	}
	// "did it find my GPU?" must be answered, not left to an empty section
	if !strings.Contains(out, "None detected") {
		t.Errorf("a server with no GPUs should say so, got %q", out)
	}
}

func TestPrettyPrintSupportedGPUs(t *testing.T) {
	var buf bytes.Buffer
	prettyPrintSupportedGPUs(&buf, " ", api.InfoResponse{
		ComputeInfo: api.ComputeInfo{
			SupportedGPUs: []api.GPUInfo{{
				ID:          "0",
				Name:        "NVIDIA GeForce RTX 4080 Laptop GPU",
				TotalMemory: 12878610432,
				FreeMemory:  12878610432,
				Compute:     "8.9",
				Driver:      "13.2",
				Runner:      "CUDA",
			}},
		},
	})

	out := buf.String()
	for _, want := range []string{"CUDA 0:", "RTX 4080", "Compute:", "8.9", "Driver:", "13.2"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected %q in output, got:\n%s", want, out)
		}
	}
	if strings.Contains(out, "None detected") {
		t.Errorf("a populated list should not report none detected, got:\n%s", out)
	}
}

// A backend that reports no compute capability or driver version (Metal) leaves
// those fields empty; the rows are dropped rather than printed blank.
func TestPrettyPrintSupportedGPUsOmitsEmptyRows(t *testing.T) {
	var buf bytes.Buffer
	prettyPrintSupportedGPUs(&buf, " ", api.InfoResponse{
		ComputeInfo: api.ComputeInfo{
			SupportedGPUs: []api.GPUInfo{{
				ID:          "0",
				Name:        "MTL0",
				TotalMemory: 12712935424,
				FreeMemory:  12711886848,
				Runner:      "Metal",
			}},
		},
	})

	out := buf.String()
	if !strings.Contains(out, "MTL0") {
		t.Fatalf("expected the device, got:\n%s", out)
	}
	if strings.Contains(out, "Compute:") {
		t.Errorf("compute row should be omitted when unknown, got:\n%s", out)
	}
	if strings.Contains(out, "Driver:") {
		t.Errorf("driver row should be omitted when unknown, got:\n%s", out)
	}
}
