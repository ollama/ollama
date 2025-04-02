package model

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
)

func setup(t *testing.T) ml.Backend {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	models := filepath.Join(home, ".ollama", "models")

	b, err := New(context.TODO(), filepath.Join(models, "blobs", "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"), ml.BackendParams{NumGPULayers: 99})
	if err != nil {
		t.Fatal(err)
	}

	return b
}

func TestUnfoldConv(t *testing.T) {
	b := setup(t)
	ctx := b.NewContext().Input()
	t.Cleanup(func() { ctx.Close() })

	tiles, channels, height, width := 5, 3, 336, 336
	patchSize := 14

	tt := ctx.Arange(0, float32(tiles*channels*height*width), 1, ml.DTypeF32).Reshape(ctx, width, height, channels, tiles)
	t.Log("tt", tt.Shape())
	t.Log(ml.Dump(ctx, tt))

	kernel := ctx.Empty(ml.DTypeF32, patchSize, patchSize, channels)
	t.Log("kernel", kernel.Shape())
	t.Log(ml.Dump(ctx, kernel))

	tt = kernel.IM2Col(ctx, tt, patchSize, patchSize, 0, 0, 1, 1)
	t.Log("tt", tt.Shape())
	t.Log(ml.Dump(ctx, tt))

	tt = tt.Reshape(ctx, tt.Dim(0), tt.Dim(1)*tt.Dim(2), tt.Dim(3))
	t.Log("tt", tt.Shape())
	t.Log(ml.Dump(ctx, tt))
}
