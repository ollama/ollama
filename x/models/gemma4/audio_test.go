package gemma4

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func newAudioTestModel() *Model {
	audio := &AudioConfig{ModelType: "gemma4_audio"}
	applyAudioDefaults(audio)
	return &Model{
		Audio:      audio,
		EmbedAudio: &MultimodalEmbedder{},
		MM: multimodalConfig{
			AudioConfig:  audio,
			AudioTokenID: 258881,
			BOATokenID:   256000,
			EOATokenID:   258883,
		},
	}
}

func TestPrepareAudioMedia(t *testing.T) {
	m := newAudioTestModel()

	prepared, err := m.PrepareMedia([]base.Segment{
		{Tokens: []int32{2, 5}},
		{Kind: "audio", Data: wavPCM16(16000)},
		{Tokens: []int32{7}},
	})
	if err != nil {
		t.Fatal(err)
	}

	// One second: 99 mel frames -> 25 soft tokens.
	wantLen := 2 + 1 + 25 + 1 + 1
	if len(prepared.Tokens) != wantLen {
		t.Fatalf("%d tokens, want %d", len(prepared.Tokens), wantLen)
	}
	if prepared.Tokens[2] != m.MM.BOATokenID || prepared.Tokens[28] != m.MM.EOATokenID {
		t.Fatalf("delimiters = %d, %d", prepared.Tokens[2], prepared.Tokens[28])
	}
	for _, tok := range prepared.Tokens[3:28] {
		if tok != m.MM.AudioTokenID {
			t.Fatalf("soft token = %d", tok)
		}
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("%d items", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{3, 28} || item.Source != 1 || !item.Causal {
		t.Fatalf("item = %+v", item)
	}
	if item.Dims[0] != 99 || item.Dims[1] != audioMelBins {
		t.Fatalf("dims = %v", item.Dims)
	}
	if p := item.Opaque.(preparedAudio); p.numTokens != 25 {
		t.Fatalf("numTokens = %d", p.numTokens)
	}
}

func TestPrepareAudioMediaChunks(t *testing.T) {
	m := newAudioTestModel()

	// 61 s: three chunks of at most 30 s, one soft-token run each, back to
	// back inside one boa/eoa pair.
	prepared, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: wavPCM16(976000)}})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 3 {
		t.Fatalf("%d items", len(prepared.Items))
	}
	next := 1
	for i, item := range prepared.Items {
		n := item.Opaque.(preparedAudio).numTokens
		if item.Range != [2]int{next, next + int(n)} || item.Source != 0 || !item.Causal {
			t.Fatalf("item %d range %v source %d causal %v, want [%d %d]", i, item.Range, item.Source, item.Causal, next, next+int(n))
		}
		if n > 750 {
			t.Fatalf("item %d: %d tokens exceed one 30 s chunk", i, n)
		}
		next += int(n)
	}
	if len(prepared.Tokens) != next+1 {
		t.Fatalf("%d tokens, want %d", len(prepared.Tokens), next+1)
	}
	if prepared.Tokens[0] != m.MM.BOATokenID || prepared.Tokens[next] != m.MM.EOATokenID {
		t.Fatalf("delimiters = %d, %d", prepared.Tokens[0], prepared.Tokens[next])
	}
}

func TestPrepareAudioMediaRejections(t *testing.T) {
	m := newAudioTestModel()

	_, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte("ID3\x04\x00junk")}})
	if err == nil || !strings.Contains(err.Error(), "unrecognized audio format") {
		t.Fatalf("mp3 error = %v", err)
	}

	_, err = m.PrepareMedia([]base.Segment{{Kind: "video", Data: []byte{1}}})
	if err == nil || !strings.Contains(err.Error(), "does not support video input") {
		t.Fatalf("video error = %v", err)
	}

	noAudio := &Model{MM: multimodalConfig{ImageTokenID: 258880}}
	_, err = noAudio.PrepareMedia([]base.Segment{{Kind: "audio", Data: wavPCM16(16000)}})
	if err == nil || !strings.Contains(err.Error(), "does not support audio input") {
		t.Fatalf("no-audio error = %v", err)
	}
}

func TestPrepareUnifiedAudioMedia(t *testing.T) {
	m := newAudioTestModel()
	m.Audio.ModelType = "gemma4_unified_audio"
	m.Audio.SamplesPerToken = 640

	// 31 s stays one item on the unified path: no chunking, one token per
	// 640-sample frame, the final partial frame zero-padded.
	prepared, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: wavPCM16(496001)}})
	if err != nil {
		t.Fatal(err)
	}
	wantTokens := (496001 + 639) / 640
	if want := 1 + wantTokens + 1; len(prepared.Tokens) != want {
		t.Fatalf("%d tokens, want %d", len(prepared.Tokens), want)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("%d items", len(prepared.Items))
	}
	item := prepared.Items[0]
	if item.Range != [2]int{1, 1 + wantTokens} || !item.Causal {
		t.Fatalf("item = %+v", item)
	}
	if item.Dims[0] != wantTokens || item.Dims[1] != 640 {
		t.Fatalf("dims = %v", item.Dims)
	}
	if len(item.MediaData) != wantTokens*640 {
		t.Fatalf("media data length = %d", len(item.MediaData))
	}
}

func TestParseAudioConfig(t *testing.T) {
	mm, err := parseMultimodalConfig([]byte(`{
		"audio_config": {"model_type": "gemma4_unified_audio", "audio_samples_per_token": 640},
		"audio_token_id": 258881, "boa_token_id": 256000, "eoa_token_index": 258883
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if mm.AudioConfig == nil || !mm.AudioConfig.unified() {
		t.Fatalf("audio config = %+v", mm.AudioConfig)
	}
	if mm.EOATokenID != 258883 {
		t.Fatalf("eoa = %d", mm.EOATokenID)
	}
	if mm.AudioConfig.RMSNormEps != 1e-6 {
		t.Fatalf("eps = %v", mm.AudioConfig.RMSNormEps)
	}

	if _, err := parseMultimodalConfig([]byte(`{
		"audio_config": {"model_type": "gemma4_unified_audio", "audio_samples_per_token": -640}}`)); err == nil {
		t.Fatal("negative audio_samples_per_token accepted")
	}

	mm, err = parseMultimodalConfig([]byte(`{"audio_config": {"model_type": "someday_audio"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if mm.AudioConfig != nil {
		t.Fatalf("unknown audio model type accepted: %+v", mm.AudioConfig)
	}

	mm, err = parseMultimodalConfig([]byte(`{"audio_config": null}`))
	if err != nil || mm.AudioConfig != nil {
		t.Fatalf("null audio config: %+v, %v", mm.AudioConfig, err)
	}
}

// TestAudioAttentionMask checks every (block, query, context slot) of the
// uploaded mask against the window rule, covering ragged final blocks.
func TestAudioAttentionMask(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := newAudioTestModel()
		a := m.Audio
		chunk, left := int(a.ChunkSize), int(a.ContextLeft-1)
		ctx := chunk + left + int(a.ContextRight)

		for _, n := range []int{5, 24, 26, 100} {
			mask := m.audioAttentionMask(n).AsType(mlx.DTypeFloat32)
			mlx.Eval(mask)
			got := mask.Floats()

			blocks := (n + chunk - 1) / chunk
			if len(got) != blocks*chunk*ctx {
				t.Fatalf("n=%d: %d values", n, len(got))
			}
			for b := range blocks {
				for q := range chunk {
					for j := range ctx {
						gq, gk := b*chunk+q, b*chunk+j-left
						dist := gq - gk
						want := float32(0)
						if gq < n && gk >= 0 && gk < n && dist >= 0 && dist < left {
							want = 1
						}
						if v := got[(b*chunk+q)*ctx+j]; v != want {
							t.Fatalf("n=%d block %d q %d j %d: %v, want %v", n, b, q, j, v, want)
						}
					}
				}
			}
		}
	})
}
