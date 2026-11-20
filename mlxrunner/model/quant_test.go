package model

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

func TestReadInt3AffineMetadata(t *testing.T) {
	for _, group := range []int{32, 64, 128} {
		t.Run(fmt.Sprintf("group%d", group), func(t *testing.T) {
			metadata := map[string]string{"quant_type": "int3", "group_size": fmt.Sprint(group)}
			columns := int32(256 * 3 / 32)
			reader := safetensors.BuildPackedSafetensorsReaderWithMetadata([]*safetensors.TensorData{
				safetensors.NewTensorDataFromBytes("expert.weight", "U32", []int32{64, columns}, make([]byte, 64*columns*4)),
				safetensors.NewTensorDataFromBytes("expert.weight.scale", "BF16", []int32{64, int32(256 / group)}, make([]byte, 64*256/group*2)),
			}, metadata)
			data, err := io.ReadAll(reader)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "blob")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			infos, _, _, err := readBlobTensorQuantInfo(path)
			if err != nil {
				t.Fatal(err)
			}
			info := infos["expert.weight"]
			if info == nil || info.QuantType != "INT3" || info.GroupSize != group {
				t.Fatalf("group %d metadata = %+v", group, info)
			}
			gs, bits, mode, _ := TensorQuantParams(64, 8, "affine", infos, "expert.weight")
			if gs != group || bits != 3 || mode != "affine" {
				t.Fatalf("resolved (%d, %d, %q), want (%d, %d, affine)", gs, bits, mode, group, 3)
			}
		})
	}
}
