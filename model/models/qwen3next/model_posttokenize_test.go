package qwen3next

import (
	"testing"

	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/model/models/qwen3vl"
)

type fakeTensor struct {
	*ggml.Tensor
	dims []int
}

func (t *fakeTensor) Dim(i int) int {
	return t.dims[i]
}

func makeImageInput(hash uint64, width, height, tokens int) *input.Input {
	return &input.Input{
		Multimodal: []input.Multimodal{{
			Tensor: &fakeTensor{dims: []int{1, tokens, 1, 1}},
			Data:   &qwen3vl.Grid{Width: width, Height: height},
		}},
		MultimodalHash: hash,
	}
}

func TestPostTokenizeMultiImageSpans(t *testing.T) {
	m := &Model{
		imageToken:       10,
		visionStart:      11,
		visionEnd:        12,
		spatialMergeSize: 2,
	}

	inputs := []*input.Input{
		{Token: 100},
		makeImageInput(1, 8, 4, 4),
		makeImageInput(2, 4, 8, 4),
		{Token: 200},
	}

	got, err := m.PostTokenize(inputs)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	want := []struct {
		token     int32
		hash      uint64
		sameBatch int
		hasMM     bool
	}{
		{token: 100},
		{token: 11, sameBatch: 5},
		{token: 10, hash: 1, hasMM: true},
		{token: 10},
		{token: 10},
		{token: 10},
		{token: 12},
		{token: 11, sameBatch: 5},
		{token: 10, hash: 2, hasMM: true},
		{token: 10},
		{token: 10},
		{token: 10},
		{token: 12},
		{token: 200},
	}

	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}

	for i := range want {
		if got[i].Token != want[i].token {
			t.Fatalf("got[%d].Token = %d, want %d", i, got[i].Token, want[i].token)
		}
		if got[i].MultimodalHash != want[i].hash {
			t.Fatalf("got[%d].MultimodalHash = %d, want %d", i, got[i].MultimodalHash, want[i].hash)
		}
		if got[i].SameBatch != want[i].sameBatch {
			t.Fatalf("got[%d].SameBatch = %d, want %d", i, got[i].SameBatch, want[i].sameBatch)
		}
		hasMM := len(got[i].Multimodal) > 0
		if hasMM != want[i].hasMM {
			t.Fatalf("got[%d].hasMM = %v, want %v", i, hasMM, want[i].hasMM)
		}
	}

	wantPositions := []int32{0, 1, 2, 2, 2, 2, 6, 7, 8, 8, 8, 8, 12, 13}
	if len(m.positionCache) != len(wantPositions) {
		t.Fatalf("len(positionCache) = %d, want %d", len(m.positionCache), len(wantPositions))
	}
	for i := range wantPositions {
		if m.positionCache[i] != wantPositions[i] {
			t.Fatalf("positionCache[%d] = %d, want %d", i, m.positionCache[i], wantPositions[i])
		}
	}
}
