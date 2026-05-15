package discover

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestApplyLinuxROCmRefinement(t *testing.T) {
	tests := []struct {
		name           string
		nodes          []fakeROCmNode
		devices        []ml.DeviceInfo
		applied        bool
		wantIntegrated []bool
	}{
		{
			name: "apu is integrated",
			nodes: []fakeROCmNode{{
				node:        1,
				renderMinor: 128,
				gfxVersion:  "90012",
				vramTotal:   2 << 30,
				gttTotal:    32 << 30,
			}},
			devices: []ml.DeviceInfo{{
				DeviceID:  ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:      "ROCm0",
				GFXTarget: "gfx90c",
			}},
			applied:        true,
			wantIntegrated: []bool{true},
		},
		{
			name: "low vram dgpu is not integrated",
			nodes: []fakeROCmNode{{
				node:        1,
				renderMinor: 128,
				gfxVersion:  "100601",
				vramTotal:   4 << 30,
				gttTotal:    32 << 30,
				vramVendor:  true,
				boardInfo:   true,
			}},
			devices: []ml.DeviceInfo{{
				DeviceID:  ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:      "ROCm0",
				GFXTarget: "gfx1061",
			}},
			applied:        true,
			wantIntegrated: []bool{false},
		},
		{
			name: "mixed system follows kfd order not drm order",
			nodes: []fakeROCmNode{
				{
					node:        1,
					renderMinor: 129,
					gfxVersion:  "110000",
					vramTotal:   48 << 30,
					gttTotal:    64 << 30,
					vramVendor:  true,
					boardInfo:   true,
				},
				{
					node:        2,
					renderMinor: 128,
					gfxVersion:  "110003",
					vramTotal:   512 << 20,
					gttTotal:    32 << 30,
				},
			},
			devices: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, Name: "ROCm0", GFXTarget: "gfx1100"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, Name: "ROCm1", GFXTarget: "gfx1103"},
			},
			applied:        true,
			wantIntegrated: []bool{false, true},
		},
		{
			name: "missing kfd data leaves devices unchanged",
			devices: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{ID: "0", Library: "ROCm"},
				Name:       "ROCm0",
				Integrated: true,
			}},
			wantIntegrated: []bool{true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sysfsRoot := t.TempDir()
			for _, node := range tt.nodes {
				writeFakeROCmNode(t, sysfsRoot, node)
			}

			devices := append([]ml.DeviceInfo(nil), tt.devices...)
			applied := applyLinuxROCmRefinement(devices, sysfsRoot)
			if applied != tt.applied {
				t.Fatalf("applied = %v, want %v", applied, tt.applied)
			}
			for i, want := range tt.wantIntegrated {
				if devices[i].Integrated != want {
					t.Fatalf("device %d integrated = %v, want %v", i, devices[i].Integrated, want)
				}
			}
		})
	}
}

type fakeROCmNode struct {
	node        int
	renderMinor int
	gfxVersion  string
	vramTotal   uint64
	gttTotal    uint64
	vramVendor  bool
	boardInfo   bool
}

func writeFakeROCmNode(t *testing.T, sysfsRoot string, node fakeROCmNode) {
	t.Helper()

	nodeDir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes", strconv.Itoa(node.node))
	if err := os.MkdirAll(nodeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	properties := "vendor_id 4098\n" +
		"device_id 1234\n" +
		"drm_render_minor " + strconv.Itoa(node.renderMinor) + "\n" +
		"gfx_target_version " + node.gfxVersion + "\n"
	if err := os.WriteFile(filepath.Join(nodeDir, "properties"), []byte(properties), 0o644); err != nil {
		t.Fatal(err)
	}

	deviceDir := filepath.Join(sysfsRoot, "class", "drm", "renderD"+strconv.Itoa(node.renderMinor), "device")
	if err := os.MkdirAll(deviceDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, deviceDir, "vendor", "0x1002\n")
	writeFakeSysfsFile(t, deviceDir, "driver", "amdgpu\n")
	writeFakeSysfsFile(t, deviceDir, "mem_info_vram_total", strconv.FormatUint(node.vramTotal, 10)+"\n")
	writeFakeSysfsFile(t, deviceDir, "mem_info_gtt_total", strconv.FormatUint(node.gttTotal, 10)+"\n")
	if node.vramVendor {
		writeFakeSysfsFile(t, deviceDir, "mem_info_vram_vendor", "samsung\n")
	}
	if node.boardInfo {
		writeFakeSysfsFile(t, deviceDir, "board_info", "type : cem\n")
	}
}

func writeFakeSysfsFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
