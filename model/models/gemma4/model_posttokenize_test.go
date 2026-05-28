package gemma4

import (
	"testing"

	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/model/input"
)

type fakeTensor struct {
	*ggml.Tensor
	dims []int
}

func (t *fakeTensor) Dim(i int) int {
	return t.dims[i]
}

func makeImageInput(hash uint64, hidden, tokens int) *input.Input {
	return &input.Input{
		Multimodal:     []input.Multimodal{{Tensor: &fakeTensor{dims: []int{hidden, tokens}}}},
		MultimodalHash: hash,
	}
}

func makeAudioInput(hash uint64, hidden, tokens int) *input.Input {
	return &input.Input{
		Multimodal: []input.Multimodal{{
			Tensor: &fakeTensor{dims: []int{hidden, tokens}},
			Data:   audioTag{},
		}},
		MultimodalHash: hash,
	}
}

func TestPostTokenizeImageWithSpecialTokens(t *testing.T) {
	m := &Model{
		imageTokenID:    50,
		imageEndTokenID: 51,
	}

	in := []*input.Input{
		{Token: 100},
		makeImageInput(7, 2816, 4),
		{Token: 200},
	}

	out, err := m.PostTokenize(in)
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
		{token: 50, sameBatch: 5},
		{token: 0, hash: 7, hasMM: true, sameBatch: 4},
		{token: 0},
		{token: 0},
		{token: 0},
		{token: 51},
		{token: 200},
	}

	if len(out) != len(want) {
		t.Fatalf("len(out) = %d, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i].Token != want[i].token {
			t.Errorf("out[%d].Token = %d, want %d", i, out[i].Token, want[i].token)
		}
		if out[i].MultimodalHash != want[i].hash {
			t.Errorf("out[%d].MultimodalHash = %d, want %d", i, out[i].MultimodalHash, want[i].hash)
		}
		if out[i].SameBatch != want[i].sameBatch {
			t.Errorf("out[%d].SameBatch = %d, want %d", i, out[i].SameBatch, want[i].sameBatch)
		}
		hasMM := len(out[i].Multimodal) > 0
		if hasMM != want[i].hasMM {
			t.Errorf("out[%d].hasMM = %v, want %v", i, hasMM, want[i].hasMM)
		}
	}
}

func TestPostTokenizePlaceholderSameBatchProtection(t *testing.T) {
	m := &Model{
		imageTokenID:    50,
		imageEndTokenID: 51,
	}

	in := []*input.Input{makeImageInput(42, 2816, 256)}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	if len(out) != 258 {
		t.Fatalf("len(out) = %d, want 258 (begin + 256 image + end)", len(out))
	}

	ph := out[1]
	if len(ph.Multimodal) != 1 {
		t.Fatalf("expected multimodal payload on placeholder")
	}

	wantSameBatch := 256 - 1 + 1
	if ph.SameBatch != wantSameBatch {
		t.Fatalf("placeholder SameBatch = %d, want %d", ph.SameBatch, wantSameBatch)
	}
}

func TestPostTokenizeImageWithoutSpecialTokens(t *testing.T) {
	m := &Model{
		imageTokenID:    -1,
		imageEndTokenID: -1,
	}

	in := []*input.Input{makeImageInput(9, 2816, 4)}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	if len(out) != 4 {
		t.Fatalf("len(out) = %d, want 4", len(out))
	}

	if out[0].SameBatch != 3 {
		t.Fatalf("placeholder SameBatch = %d, want 3", out[0].SameBatch)
	}
	if len(out[0].Multimodal) != 1 || out[0].MultimodalHash != 9 {
		t.Fatalf("expected multimodal payload on out[0]")
	}
	for i := 1; i < 4; i++ {
		if out[i].Token != 0 || len(out[i].Multimodal) != 0 {
			t.Fatalf("out[%d] should be a plain padding token", i)
		}
	}
}

func TestPostTokenizeAudio(t *testing.T) {
	m := &Model{
		imageTokenID:    50,
		imageEndTokenID: 51,
		audioTokenID:    60,
		audioEndTokenID: 61,
	}

	in := []*input.Input{makeAudioInput(13, 2048, 4)}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize() error = %v", err)
	}

	if len(out) != 6 {
		t.Fatalf("len(out) = %d, want 6 (begin + 4 audio + end)", len(out))
	}

	if out[0].Token != 60 || out[0].SameBatch != 5 {
		t.Fatalf("out[0] = %+v, want audio begin with SameBatch=5", *out[0])
	}
	if len(out[1].Multimodal) != 1 || out[1].MultimodalHash != 13 || out[1].SameBatch != 4 {
		t.Fatalf("out[1] = %+v, want audio placeholder with SameBatch=4", *out[1])
	}
	if out[5].Token != 61 {
		t.Fatalf("out[5].Token = %d, want 61", out[5].Token)
	}
}
