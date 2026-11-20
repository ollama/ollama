package mlxrunner

import (
	"context"
	"encoding/binary"
	"errors"
	"maps"
	"math"
	"net/http"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestLoadTensorsSkipsNamedTensors(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	var data []*safetensors.TensorData
	for _, name := range []string{"skipped.weight", "skipped.weight.scale", "skipped.weight.global_scale", "target.weight", "draft.weight"} {
		data = append(data, safetensors.NewTensorDataFromBytes(name, "F32", []int32{1}, binary.LittleEndian.AppendUint32(nil, math.Float32bits(2))))
	}
	layer, err := manifest.NewLayer(safetensors.BuildPackedSafetensorsReader(data), manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatal(err)
	}
	root := &model.Root{Manifest: &manifest.Manifest{Layers: []manifest.Layer{layer, layer}}}
	for _, tc := range []struct {
		name string
		skip []string
		want []string
	}{
		{"all tensors", nil, []string{"draft.weight", "skipped.weight", "skipped.weight.global_scale", "skipped.weight_scale", "target.weight"}},
		{"skip named tensors", []string{"skipped.weight", "skipped.weight.scale", "skipped.weight.global_scale"}, []string{"draft.weight", "target.weight"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				tensors, err := loadTensorsFromManifest(root, tc.skip)
				if err != nil {
					t.Fatal(err)
				}
				if got := slices.Sorted(maps.Keys(tensors)); !slices.Equal(got, tc.want) {
					t.Fatalf("loaded tensors = %v, want %v", got, tc.want)
				}
				for name, arr := range tensors {
					if got := arr.Floats(); !slices.Equal(got, []float32{2}) {
						t.Errorf("%s = %v, want [2]", name, got)
					}
				}
			})
		})
	}
}

func TestRunnerCloseReleasesGenerationCache(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		activeMemory := func() int {
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			return mlx.ActiveMemory()
		}
		baseline := activeMemory()
		pc := newPrefixCache([]cache.Cache{cache.NewKVCache(), nil, cache.NewRotatingKVCache(4)})
		defer pc.close()
		r := &Runner{cache: pc}

		for _, inputs := range [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}} {
			mlx.Scoped(func() {
				session := pc.begin(inputs, nil)
				defer session.close()
				n := len(session.remaining)
				b := &batch.Batch{
					InputIDs:     mlx.FromValues(session.remaining, 1, n),
					SeqOffsets:   []int32{int32(pc.minCacheOffset())},
					SeqQueryLens: []int32{int32(n)},
				}
				keys := mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 8)
				values := mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 8)
				for _, kv := range pc.caches {
					if kv != nil {
						kv.(cache.Attention).Update(b, keys, values)
						mlx.Eval(kv.State()...)
					}
				}
			})
		}
		if pc.minCacheOffset() == 0 || pc.pagedOutBytes == 0 || activeMemory() <= baseline {
			t.Fatal("requests did not populate live and paged-out cache state")
		}

		r.Close()
		if r.cache != nil {
			t.Error("closed runner retains generation cache")
		}
		if pc.root != nil || len(pc.activePath) != 0 || pc.pagedOutBytes != 0 {
			t.Error("closed runner retains prefix snapshots")
		}
		for i, kv := range pc.caches {
			if kv != nil && kv.Offset() != 0 {
				t.Errorf("cache %d retains offset %d after close", i, kv.Offset())
			}
		}
		if got := activeMemory(); got != baseline {
			t.Errorf("close retained %d active bytes above baseline %d", got-baseline, baseline)
		}
		r.Close()
		if got := activeMemory(); got != baseline {
			t.Errorf("repeated close retained %d active bytes above baseline %d", got-baseline, baseline)
		}
	})
}

func TestRunnerCloseEmpty(t *testing.T) {
	for _, r := range []*Runner{{}, {cache: newPrefixCache(nil)}} {
		r.Close()
		r.Close()
		if r.cache != nil {
			t.Error("closed runner retains empty generation cache")
		}
	}
}

func TestRequestQueueTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := &Runner{Requests: make(chan queuedRequest)}
		requests := []queuedRequest{
			request{Ctx: t.Context()},
			&evaluationRequest{ctx: t.Context()},
			&speechRequest{ctx: t.Context()},
		}
		for _, req := range requests {
			start := time.Now()
			err := r.enqueue(req.context(), req)
			var status api.StatusError
			if !errors.As(err, &status) || status.StatusCode != http.StatusTooManyRequests {
				t.Fatalf("%T: got %v, want 429", req, err)
			}
			if elapsed := time.Since(start); elapsed != 5*time.Minute {
				t.Fatalf("%T: queue wait took %s, want 5 minutes", req, elapsed)
			}
		}
		select {
		case <-r.Requests:
			t.Fatal("timed-out request remained queued")
		default:
		}
	})
}

func TestSerialRequestQueueWaitsForCapacity(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := &Runner{Requests: make(chan queuedRequest)}
		go func() {
			for range 3 {
				time.Sleep(90 * time.Second)
				<-r.Requests
			}
		}()
		admitted := make(chan error, 3)
		for range 3 {
			go func() { admitted <- r.enqueue(t.Context(), request{Ctx: t.Context()}) }()
		}
		for range 3 {
			if err := <-admitted; err != nil {
				t.Fatalf("normal serial queue rejected a request: %v", err)
			}
		}
	})
}

func TestAdmittedRequestOutlivesQueueTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := &Runner{Requests: make(chan queuedRequest)}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		go func() {
			req := <-r.Requests
			time.Sleep(6 * time.Minute)
			if req.context().Err() != nil {
				t.Error("queue timeout cancelled an admitted request")
			}
			cancel()
		}()
		if err := r.enqueue(ctx, request{Ctx: ctx}); err != nil {
			t.Fatal(err)
		}
		<-ctx.Done()
		if err := r.enqueue(ctx, request{Ctx: ctx}); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled caller: got %v", err)
		}
	})
}
