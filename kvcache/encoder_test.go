package kvcache

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func TestEncoderCacheNew(t *testing.T) {
	cache := NewEncoderCache()
	defer cache.Close()

	if cache == nil {
		t.Fatal("NewEncoderCache returned nil")
	}

	if cache.ctxs == nil {
		t.Error("ctxs map not initialized")
	}
	if cache.keys == nil {
		t.Error("keys map not initialized")
	}
	if cache.values == nil {
		t.Error("values map not initialized")
	}
}

func TestEncoderCacheInit(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		if cache.backend == nil {
			t.Error("backend not set after Init")
		}
		if cache.config == nil {
			t.Error("config not set after Init")
		}
	})
}

func TestEncoderCacheInitPanicMultipleSequences(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Init with maxSequences > 1 should panic")
		}
	}()

	backend := &testBackend{}
	cache := NewEncoderCache()
	defer cache.Close()

	// Should panic because encoder cache does not support multiple sequences
	cache.Init(backend, ml.DTypeF16, 2, 16, 16)
}

func TestEncoderCacheSetConfig(t *testing.T) {
	cache := NewEncoderCache()
	defer cache.Close()

	config := ml.CacheConfig{PermutedV: true}
	cache.SetConfig(config)

	if cache.config == nil {
		t.Error("config not set after SetConfig")
	}
	if !cache.config.PermutedV {
		t.Error("config PermutedV not set correctly")
	}
}

func TestEncoderCacheSetConfigPanicOnSecondCall(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("SetConfig called twice should panic")
		}
	}()

	cache := NewEncoderCache()
	defer cache.Close()

	config := ml.CacheConfig{PermutedV: true}
	cache.SetConfig(config)
	// Second call should panic
	cache.SetConfig(config)
}

func TestEncoderCacheSetLayer(t *testing.T) {
	cache := NewEncoderCache()
	defer cache.Close()

	cache.SetLayer(5)
	if cache.curLayer != 5 {
		t.Errorf("SetLayer: got %d, want 5", cache.curLayer)
	}

	cache.SetLayer(10)
	if cache.curLayer != 10 {
		t.Errorf("SetLayer: got %d, want 10", cache.curLayer)
	}
}

func TestEncoderCacheEncoderCached(t *testing.T) {
	cache := NewEncoderCache()
	defer cache.Close()

	if cache.EncoderCached() {
		t.Error("EncoderCached should return false initially")
	}

	cache.encoderCached = true
	if !cache.EncoderCached() {
		t.Error("EncoderCached should return true after setting")
	}
}

func TestEncoderCacheStartForward(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{0, 1, 2},
			Sequences: []int{0, 0, 0},
			Multimodal: []input.MultimodalIndex{
				{Index: 2},
			},
		}

		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		if cache.curPos != 2 {
			t.Errorf("curPos: got %d, want 2", cache.curPos)
		}
		if cache.curReserve {
			t.Error("curReserve should be false")
		}
	})
}

func TestEncoderCacheStartForwardReserve(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{0},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		err := cache.StartForward(ctx, batch, true)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		if !cache.curReserve {
			t.Error("curReserve should be true when reserve=true")
		}
	})
}

func TestEncoderCachePutAndGet(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{0},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		cache.SetLayer(0)

		key := ctx.FromFloats([]float32{1, 2, 3, 4}, 2, 2)
		value := ctx.FromFloats([]float32{5, 6, 7, 8}, 2, 2)

		cache.Put(ctx, key, value)

		if !cache.EncoderCached() {
			t.Error("EncoderCached should be true after Put")
		}

		gotKey, gotValue, mask := cache.Get(ctx)

		if gotKey == nil {
			t.Error("Get returned nil key")
		}
		if gotValue == nil {
			t.Error("Get returned nil value")
		}
		if mask != nil {
			t.Error("Get should return nil mask for encoder cache")
		}
	})
}

func TestEncoderCachePutReserveDoesNotUpdateMetadata(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{0},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		// Start with reserve=true
		err := cache.StartForward(ctx, batch, true)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		cache.SetLayer(0)

		key := ctx.FromFloats([]float32{1, 2, 3, 4}, 2, 2)
		value := ctx.FromFloats([]float32{5, 6, 7, 8}, 2, 2)

		cache.Put(ctx, key, value)

		// When reserve=true, metadata should not be updated
		if cache.EncoderCached() {
			t.Error("EncoderCached should be false when Put is called with reserve=true")
		}
	})
}

func TestEncoderCacheRemove(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{5},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		cache.SetLayer(0)

		key := ctx.FromFloats([]float32{1, 2, 3, 4}, 2, 2)
		value := ctx.FromFloats([]float32{5, 6, 7, 8}, 2, 2)

		cache.Put(ctx, key, value)

		if !cache.EncoderCached() {
			t.Fatal("EncoderCached should be true after Put")
		}

		// Remove range that includes the cached position
		err = cache.Remove(0, 0, 10)
		if err != nil {
			t.Fatalf("Remove failed: %v", err)
		}

		if cache.EncoderCached() {
			t.Error("EncoderCached should be false after Remove that includes cached position")
		}
	})
}

func TestEncoderCacheRemoveOutsideRange(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{5},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		cache.SetLayer(0)

		key := ctx.FromFloats([]float32{1, 2, 3, 4}, 2, 2)
		value := ctx.FromFloats([]float32{5, 6, 7, 8}, 2, 2)

		cache.Put(ctx, key, value)

		if !cache.EncoderCached() {
			t.Fatal("EncoderCached should be true after Put")
		}

		// Remove range that does NOT include the cached position (5)
		err = cache.Remove(0, 0, 5)
		if err != nil {
			t.Fatalf("Remove failed: %v", err)
		}

		// Should still be cached since position 5 is not in range [0, 5)
		if !cache.EncoderCached() {
			t.Error("EncoderCached should still be true after Remove that doesn't include cached position")
		}
	})
}

func TestEncoderCacheCanResume(t *testing.T) {
	cache := NewEncoderCache()
	defer cache.Close()

	// CanResume always returns true for encoder cache
	if !cache.CanResume(0, 0) {
		t.Error("CanResume(0, 0) should return true")
	}
	if !cache.CanResume(1, 100) {
		t.Error("CanResume(1, 100) should return true")
	}
}

func TestEncoderCacheCopyPrefixPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("CopyPrefix should panic for encoder cache")
		}
	}()

	cache := NewEncoderCache()
	defer cache.Close()

	cache.CopyPrefix(0, 1, 10)
}

func TestEncoderCacheMultipleLayers(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewEncoderCache()
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 16)

		ctx := backend.NewContext()
		defer ctx.Close()

		batch := input.Batch{
			Positions: []int32{0},
			Sequences: []int{0},
			Multimodal: []input.MultimodalIndex{
				{Index: 0},
			},
		}

		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		// Put data in layer 0
		cache.SetLayer(0)
		key0 := ctx.FromFloats([]float32{1, 2}, 2)
		value0 := ctx.FromFloats([]float32{3, 4}, 2)
		cache.Put(ctx, key0, value0)

		// Put data in layer 1
		cache.SetLayer(1)
		key1 := ctx.FromFloats([]float32{5, 6}, 2)
		value1 := ctx.FromFloats([]float32{7, 8}, 2)
		cache.Put(ctx, key1, value1)

		// Verify layer 0 data
		cache.SetLayer(0)
		gotKey0, gotValue0, _ := cache.Get(ctx)
		if gotKey0 == nil || gotValue0 == nil {
			t.Error("Layer 0 data should be retrievable")
		}

		// Verify layer 1 data
		cache.SetLayer(1)
		gotKey1, gotValue1, _ := cache.Get(ctx)
		if gotKey1 == nil || gotValue1 == nil {
			t.Error("Layer 1 data should be retrievable")
		}
	})
}
