package lfm2

import (
	"testing"

	"github.com/ollama/ollama/model/input"
)

func TestPostTokenizeWithSpecialImageTokens(t *testing.T) {
	m := &Model{
		imageTokenID:     396,
		imageStartToken:  2,
		imageEndToken:    3,
		useSpecialTokens: true,
	}

	in := []*input.Input{
		{Token: 11},
		{Multimodal: []input.Multimodal{{Data: 64}}, MultimodalHash: 123},
		{Token: 12},
	}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize returned error: %v", err)
	}

	if len(out) != 68 {
		t.Fatalf("expected 68 tokens, got %d", len(out))
	}

	if out[0].Token != 11 {
		t.Fatalf("out[0].Token = %d, want 11", out[0].Token)
	}
	if out[1].Token != 2 {
		t.Fatalf("out[1].Token = %d, want 2", out[1].Token)
	}

	firstImage := out[2]
	if firstImage.Token != 396 {
		t.Fatalf("out[2].Token = %d, want 396", firstImage.Token)
	}
	if len(firstImage.Multimodal) != 1 {
		t.Fatalf("expected multimodal payload on first image token")
	}
	if firstImage.MultimodalHash != 123 {
		t.Fatalf("out[2].MultimodalHash = %d, want 123", firstImage.MultimodalHash)
	}
	if firstImage.SameBatch != 63 {
		t.Fatalf("out[2].SameBatch = %d, want 63", firstImage.SameBatch)
	}

	for i := 3; i < 66; i++ {
		if out[i].Token != 396 {
			t.Fatalf("out[%d].Token = %d, want 396", i, out[i].Token)
		}
		if len(out[i].Multimodal) != 0 {
			t.Fatalf("out[%d] should not carry multimodal payload", i)
		}
	}

	if out[66].Token != 3 {
		t.Fatalf("out[66].Token = %d, want 3", out[66].Token)
	}
	if out[67].Token != 12 {
		t.Fatalf("out[67].Token = %d, want 12", out[67].Token)
	}
}

func TestPostTokenizeWithoutSpecialImageTokens(t *testing.T) {
	m := &Model{
		imageTokenID:     777,
		useSpecialTokens: false,
	}

	in := []*input.Input{
		{Multimodal: []input.Multimodal{{Data: 5}}, MultimodalHash: 9},
	}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize returned error: %v", err)
	}

	if len(out) != 5 {
		t.Fatalf("expected 5 tokens, got %d", len(out))
	}
	if out[0].Token != 777 || out[0].SameBatch != 4 || len(out[0].Multimodal) != 1 {
		t.Fatalf("unexpected first token: %+v", *out[0])
	}
	for i := 1; i < 5; i++ {
		if out[i].Token != 777 {
			t.Fatalf("out[%d].Token = %d, want 777", i, out[i].Token)
		}
		if len(out[i].Multimodal) != 0 {
			t.Fatalf("out[%d] should not carry multimodal payload", i)
		}
	}
}

func TestPostTokenizeMultiTileLayoutTokens(t *testing.T) {
	m := &Model{
		imageTokenID:     396,
		imageStartToken:  498,
		imageEndToken:    499,
		imageThumbnailID: 497,
		imageRowColIDs: map[imageGridPos]int32{
			{row: 1, col: 1}: 397,
			{row: 1, col: 2}: 398,
		},
		useSpecialTokens: true,
	}

	layout := &visionEmbeddingLayout{rows: 1, cols: 2, hasThumbnail: true}
	in := []*input.Input{{
		Multimodal: []input.Multimodal{
			{Data: visionChunkData{tokens: 3, row: 1, col: 1, layout: layout}},
			{Data: visionChunkData{tokens: 3, row: 1, col: 2}},
			{Data: visionChunkData{tokens: 2, thumbnail: true}},
		},
		MultimodalHash: 1,
	}}

	out, err := m.PostTokenize(in)
	if err != nil {
		t.Fatalf("PostTokenize returned error: %v", err)
	}

	got := make([]int32, len(out))
	for i := range out {
		got[i] = out[i].Token
	}

	want := []int32{
		498, // <|image_start|>
		397, // <|img_row_1_col_1|>
		396, 396, 396,
		398, // <|img_row_1_col_2|>
		396, 396, 396,
		497, // <|img_thumbnail|>
		396, 396,
		499, // <|image_end|>
	}

	if len(got) != len(want) {
		t.Fatalf("len(out) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d].Token = %d, want %d", i, got[i], want[i])
		}
	}

	if len(out[2].Multimodal) != 1 || len(out[6].Multimodal) != 1 || len(out[10].Multimodal) != 1 {
		t.Fatalf("expected multimodal payload on first token of each chunk")
	}
	if out[2].SameBatch != 2 || out[6].SameBatch != 2 || out[10].SameBatch != 1 {
		t.Fatalf("unexpected SameBatch values: [%d %d %d]", out[2].SameBatch, out[6].SameBatch, out[10].SameBatch)
	}
}
