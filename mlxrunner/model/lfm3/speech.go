package lfm3

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lfm3/lfm3speech"
	"github.com/ollama/ollama/model/renderers"
)

// speechCodecPrefix names the codec tensors create imports from the
// checkpoint's codec subfolder.
const speechCodecPrefix = "codec."

var (
	_ model.SpeechModel = (*SpeechModel)(nil)
	_ model.MediaModel  = (*SpeechModel)(nil)
)

// SpeechModel is an Audio or Omni checkpoint: it chats like VLModel and reads
// text aloud. Speech runs the backbone in audio mode on its own caches: the
// depth decoder samples a frame of codec codes from each final hidden state,
// the frame's code embedding is the next input, and the codec streams audio
// until the model ends the segment.
type SpeechModel struct {
	*VLModel
	// Speech is nil when the model was created without its codec.
	Speech *SpeechOutput
	setup  *speechSetup
}

// SpeechOutput holds the speech-only weights.
type SpeechOutput struct {
	Depth *lfm3speech.DepthDecoder
	Codes *lfm3speech.CodeEmbedding
	Codec *lfm3speech.Codec
}

type speechSetup struct {
	cfg         *lfm3speech.Config
	names       lfm3speech.TensorNames
	codecConfig []byte
	omni        bool
}

func newSpeechModel(root *model.Root, vl *VLModel, configJSON []byte, omni bool) (*SpeechModel, error) {
	// The float32 codec needs full float32 matmuls; TF32 (the M5 default)
	// drops it to about 50 dB. It costs no measurable speech speed.
	if err := os.Setenv("MLX_ENABLE_TF32", "0"); err != nil {
		return nil, err
	}
	cfg, err := lfm3speech.ParseConfig(configJSON)
	if err != nil {
		return nil, err
	}
	names := lfm3speech.AudioNames
	if omni {
		names = lfm3speech.OmniNames
	}
	codecPath := "audio_tokenizer/config.json"
	if _, ok := root.Manifest.ConfigLayer(codecPath); !ok {
		slog.Warn("LFM3 speech output is unavailable; recreate the model to import its codec", "missing", codecPath)
		return &SpeechModel{VLModel: vl}, nil
	}
	codecConfig, err := root.Manifest.ReadConfig(codecPath)
	if err != nil {
		return nil, fmt.Errorf("load LFM3 speech codec config: %w", err)
	}
	return &SpeechModel{VLModel: vl, setup: &speechSetup{cfg: cfg, names: names, codecConfig: codecConfig, omni: omni}}, nil
}

func (m *SpeechModel) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.VLModel.LoadWeights(tensors); err != nil {
		return err
	}
	s := m.setup
	if s == nil {
		return nil
	}
	depth, err := lfm3speech.LoadDepthDecoder(tensors, s.names, s.cfg)
	if err != nil {
		return err
	}
	codes, err := lfm3speech.LoadCodeEmbedding(tensors, s.names.CodeEmbedding, s.cfg)
	if err != nil {
		return err
	}
	if got := codes.Weight.Dim(1); got != int(m.HiddenSize) {
		return fmt.Errorf("LFM3 code embedding width %d, want %d", got, m.HiddenSize)
	}
	codec, err := lfm3speech.LoadCodec(tensors, speechCodecPrefix, s.codecConfig)
	if err != nil {
		return err
	}
	cc := codec.Config()
	if cc.NumCodebooks != s.cfg.NumCodebooks || cc.CodebookSize() != int(s.cfg.EOSCode) {
		return fmt.Errorf("LFM3 codec has %d codebooks of %d codes; the model emits %d of %d plus EOS", cc.NumCodebooks, cc.CodebookSize(), s.cfg.NumCodebooks, s.cfg.EOSCode)
	}
	if codec.SampleRate() != 24000 {
		return fmt.Errorf("LFM3 codec rate %d Hz, want 24000", codec.SampleRate())
	}
	m.Speech = &SpeechOutput{Depth: depth, Codes: codes, Codec: codec}
	return nil
}

func (m *SpeechModel) Speak(ctx context.Context, req llm.SpeechRequest, emit func([]float32) error) error {
	if m.Speech == nil {
		return api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: "this model was created without its speech codec; recreate it to use speech"}
	}
	voice, err := renderers.ParseLFM3Voice(req.Voice)
	if err != nil {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	count := func(s string) int { return len(m.tok.Encode(s, false)) }
	chunks := renderers.LFM3SpeechChunks(req.Input, renderers.LFM3SpeechChunkTokens, count)
	if len(chunks) == 0 {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "input has no text to speak"}
	}
	rng := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	stream := m.Speech.Codec.NewStream()
	defer stream.Close()
	// One codec stream carries every chunk so the audio has no seams.
	for _, text := range chunks {
		if err := m.speakChunk(ctx, voice, text, count(text), rng, stream, emit); err != nil {
			return err
		}
	}
	return emit(stream.Flush())
}

// speakChunk generates one utterance with fresh backbone state.
func (m *SpeechModel) speakChunk(ctx context.Context, voice renderers.LFM3Voice, text string, textTokens int, rng *rand.Rand, stream *lfm3speech.Stream, emit func([]float32) error) error {
	sampler, err := lfm3speech.NewFrameSampler(m.setup.cfg, lfm3speech.DefaultSampleOptions(m.setup.omni), rng)
	if err != nil {
		return err
	}
	prompt := m.tok.Encode(speechPrompt(m.setup.omni, voice, text), false)
	budget := renderers.LFM3SpeechFrameBudget(textTokens)

	caches := m.NewCaches()
	defer func() {
		for _, c := range caches {
			c.Free()
		}
	}()
	held := mlx.NewScope()
	defer held.Close()

	hidden, err := m.forwardSpeech(caches, 0, func() (*mlx.Array, *mlx.Array, error) {
		ids := mlx.FromValues(prompt, 1, len(prompt))
		return ids, m.EmbedTokens.Forward(ids), nil
	})
	if err != nil {
		return err
	}
	held.Attach(hidden)
	pos := len(prompt)
	for n := range budget {
		if err := ctx.Err(); err != nil {
			return err
		}
		frame, eos := m.Speech.Depth.DecodeFrame(hidden, sampler)
		if eos {
			return nil
		}
		// Queue the next backbone step before decoding audio so the GPU
		// is not idle while the host prepares the codec input.
		if n+1 < budget {
			next, err := m.forwardSpeech(caches, pos, func() (*mlx.Array, *mlx.Array, error) {
				embedded, err := m.Speech.Codes.Forward(frame)
				if err != nil {
					return nil, nil, err
				}
				return mlx.Zeros(mlx.DTypeInt32, 1, 1), mlx.ExpandDims(embedded, 0), nil
			})
			if err != nil {
				return err
			}
			held.Discard(hidden)
			hidden = next
			held.Attach(hidden)
			pos++
		}
		samples, err := stream.Push(frame)
		if err != nil {
			return err
		}
		if len(samples) > 0 {
			if err := emit(samples); err != nil {
				return err
			}
		}
	}
	slog.Warn("LFM3 speech hit its frame budget before ending", "frames", budget, "text_tokens", textTokens)
	return nil
}

// forwardSpeech runs one backbone forward at pos and returns the final-norm
// hidden state of its last position, [1, 1, hidden], queued for evaluation
// with the updated caches.
func (m *SpeechModel) forwardSpeech(caches []cache.Cache, pos int, input func() (ids, embedded *mlx.Array, err error)) (*mlx.Array, error) {
	var err error
	out := mlx.ScopedArrays(func() []*mlx.Array {
		var ids, embedded *mlx.Array
		if ids, embedded, err = input(); err != nil {
			return nil
		}
		L := int32(ids.Dim(1))
		h, _ := m.forwardHidden(embedded, &batch.Batch{
			InputIDs:     ids,
			SeqOffsets:   []int32{int32(pos)},
			SeqQueryLens: []int32{L},
		}, caches)
		return []*mlx.Array{mlx.SliceStartStop(h, []int32{0, L - 1, 0}, []int32{1, L, m.HiddenSize})}
	})
	if err != nil {
		return nil, err
	}
	eval := slices.Clone(out)
	for _, c := range caches {
		for _, a := range c.State() {
			if a != nil {
				eval = append(eval, a)
			}
		}
	}
	mlx.AsyncEval(eval...)
	return out[0], nil
}

// speechPrompt renders the publisher's TTS conversation. Audio follows its
// README: a TTS system prompt and an assistant turn prefilled with
// <|audio_start|>. Omni's README system prompt with the default generation
// prompt and the marker its audio generation mode appends.
func speechPrompt(omni bool, v renderers.LFM3Voice, text string) string {
	system := fmt.Sprintf("Perform TTS in %s using a %s voice.", v.Language, v.Gender)
	assistant := "<|audio_start|>"
	if omni {
		system = fmt.Sprintf("Read this with a %s voice in %s.", v.Gender, v.Language)
		assistant = "<think></think><|audio_start|>"
	}
	return "<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + renderers.LFM3EscapeSpeechText(text) + "<|im_end|>\n<|im_start|>assistant\n" + assistant
}
