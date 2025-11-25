package llamarunner

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/llama"
)

func TestImageCache(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 4)}

	valA := []llama.MtmdChunk{{Embed: []float32{0.1, 0.2}}, {Embed: []float32{0.3}}}
	valB := []llama.MtmdChunk{{Embed: []float32{0.4}}, {Embed: []float32{0.5}}, {Embed: []float32{0.6}}}
	valC := []llama.MtmdChunk{{Embed: []float32{0.7}}}
	valD := []llama.MtmdChunk{{Embed: []float32{0.8}}}
	valE := []llama.MtmdChunk{{Embed: []float32{0.9}}}

	// Empty cache
	result, err := cache.findImage(0x5adb61d31933a946)
	if err != errImageNotFound {
		t.Errorf("found result in empty cache: result %v, err %v", result, err)
	}

	// Insert A
	cache.addImage(0x5adb61d31933a946, valA)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Insert B
	cache.addImage(0x011551369a34a901, valB)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Replace B with C
	cache.addImage(0x011551369a34a901, valC)

	result, err = cache.findImage(0x5adb61d31933a946)
	if !reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}

	// Evict A
	cache.addImage(0x756b218a517e7353, valB)
	cache.addImage(0x75e5e8d35d7e3967, valD)
	cache.addImage(0xd96f7f268ca0646e, valE)

	result, err = cache.findImage(0x5adb61d31933a946)
	if reflect.DeepEqual(result, valA) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x756b218a517e7353)
	if !reflect.DeepEqual(result, valB) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if !reflect.DeepEqual(result, valC) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0x75e5e8d35d7e3967)
	if !reflect.DeepEqual(result, valD) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
	result, err = cache.findImage(0xd96f7f268ca0646e)
	if !reflect.DeepEqual(result, valE) {
		t.Errorf("failed to find expected value: result %v, err %v", result, err)
	}
}
