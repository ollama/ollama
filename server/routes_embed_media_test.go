package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type recordedEmbedInput struct {
	text  string
	media [][]byte
}

type recordingEmbedRunner struct {
	llm.LlamaServer
	mu      sync.Mutex
	inputs  []recordedEmbedInput
	vectors map[string][]float32
}

func (r *recordingEmbedRunner) Embedding(_ context.Context, text string, media [][]byte) ([]float32, int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.inputs = append(r.inputs, recordedEmbedInput{text, media})
	if vector, ok := r.vectors[text]; ok {
		return append([]float32(nil), vector...), 1, nil
	}
	return []float32{1, 0, 0}, 1, nil
}

func (r *recordingEmbedRunner) takeInputs() []recordedEmbedInput {
	r.mu.Lock()
	defer r.mu.Unlock()
	inputs := r.inputs
	r.inputs = nil
	return inputs
}

func newMediaEmbedServer(t *testing.T) (*Server, *recordingEmbedRunner) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	writeShowManifestVariant(t, "embed-media", manifest.RunnerMLX, manifest.FormatSafetensors, model.ConfigV2{
		ModelFormat:  manifest.FormatSafetensors,
		ContextLen:   8192,
		Capabilities: []string{string(model.CapabilityEmbedding)},
	}, nil)
	runner := &recordingEmbedRunner{}
	return newServerWithMockRunner(t, &mockRunner{LlamaServer: runner}), runner
}

func TestEmbedHandlerMediaOrder(t *testing.T) {
	s, runner := newMediaEmbedServer(t)
	// Distinct payloads let the test observe ordering and exact decoded bytes
	// at the runner boundary without loading a multimodal model.
	image, audio := []byte("image payload"), []byte("audio payload")
	fields := []string{
		`"text":"Describe this clip"`,
		fmt.Sprintf(`"image":%q`, base64.StdEncoding.EncodeToString(image)),
		fmt.Sprintf(`"audio":%q`, base64.StdEncoding.EncodeToString(audio)),
	}
	for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			body := json.RawMessage(`{"model":"embed-media","input":{` + fields[order[0]] + "," + fields[order[1]] + "," + fields[order[2]] + `}}`)
			want := []recordedEmbedInput{{"Describe this clip", [][]byte{image, audio}}}
			// Map iteration can change between requests even for identical JSON.
			for i := range 256 {
				w := createRequest(t, s.EmbedHandler, body)
				if w.Code != http.StatusOK {
					t.Fatalf("request %d: status %d: %s", i, w.Code, w.Body)
				}
				if got := runner.takeInputs(); !reflect.DeepEqual(got, want) {
					t.Fatalf("request %d: runner inputs = %v, want %v", i, got, want)
				}
			}
		})
	}
}

func TestEmbedHandlerSingleModality(t *testing.T) {
	s, runner := newMediaEmbedServer(t)
	for _, tc := range []struct {
		name, input string
		want        recordedEmbedInput
	}{
		{"string", `"hello"`, recordedEmbedInput{text: "hello"}},
		{"text", `{"text":"hello"}`, recordedEmbedInput{text: "hello"}},
		{"image", `{"image":"aW1hZ2U="}`, recordedEmbedInput{media: [][]byte{[]byte("image")}}},
		{"audio", `{"audio":"YXVkaW8="}`, recordedEmbedInput{media: [][]byte{[]byte("audio")}}},
		{"empty image", `{"image":""}`, recordedEmbedInput{media: [][]byte{{}}}},
		{"empty audio", `{"audio":""}`, recordedEmbedInput{media: [][]byte{{}}}},
		{"image and audio", `{"audio":"YXVkaW8=","image":"aW1hZ2U="}`, recordedEmbedInput{media: [][]byte{[]byte("image"), []byte("audio")}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := createRequest(t, s.EmbedHandler, json.RawMessage(`{"model":"embed-media","input":`+tc.input+`}`))
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body)
			}
			if got := runner.takeInputs(); !reflect.DeepEqual(got, []recordedEmbedInput{tc.want}) {
				t.Fatalf("runner inputs = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEmbedHandlerMixedMediaBatch(t *testing.T) {
	s, runner := newMediaEmbedServer(t)
	runner.vectors = map[string][]float32{"plain": {1, 0, 0}, "first": {0, 1, 0}, "second": {0, 0, 1}}
	request := api.EmbedRequest{Model: "embed-media", Input: []any{
		"plain",
		map[string]string{"text": "first", "image": "aW1hZ2Ux", "audio": "YXVkaW8x"},
		map[string]string{"text": "second", "image": "aW1hZ2Uy", "audio": "YXVkaW8y"},
	}}
	w := createRequest(t, s.EmbedHandler, request)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	inputs := runner.takeInputs()
	if len(inputs) != 3 {
		t.Fatalf("runner received %d inputs, want 3", len(inputs))
	}
	wantMedia := map[string][][]byte{
		"plain":  nil,
		"first":  {[]byte("image1"), []byte("audio1")},
		"second": {[]byte("image2"), []byte("audio2")},
	}
	for _, input := range inputs {
		want, ok := wantMedia[input.text]
		if !ok || !reflect.DeepEqual(input.media, want) {
			t.Fatalf("unexpected runner input: %v", input)
		}
		delete(wantMedia, input.text)
	}
	var response api.EmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	wantEmbeddings := [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if !reflect.DeepEqual(response.Embeddings, wantEmbeddings) {
		t.Fatalf("embeddings = %v, want %v", response.Embeddings, wantEmbeddings)
	}
}

func TestEmbedHandlerInvalidMediaInput(t *testing.T) {
	s, runner := newMediaEmbedServer(t)
	for _, tc := range []struct{ input, message string }{
		{`{"image":"!"}`, "input.image: illegal base64 data"},
		{`{"audio":"!"}`, "input.audio: illegal base64 data"},
		{`{"text":1}`, "input.text must be a string"},
		{`{"image":1}`, "input.image must be a string"},
		{`{"audio":null}`, "input.audio must be a string"},
		{`{"other":"value"}`, `unknown input field "other"`},
		{`{"video":"value"}`, "input.video not supported"},
		{`{}`, "input item has no text, image, or audio"},
		{`{"text":""}`, "input item has no text, image, or audio"},
	} {
		t.Run(tc.input, func(t *testing.T) {
			// Both the object and batch paths must reject invalid items before
			// dispatching any work to the runner.
			for _, input := range []string{tc.input, `["valid",` + tc.input + `]`} {
				w := createRequest(t, s.EmbedHandler, json.RawMessage(`{"model":"embed-media","input":`+input+`}`))
				var response struct{ Error string }
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if w.Code != http.StatusBadRequest || !strings.Contains(response.Error, tc.message) {
					t.Fatalf("status %d: %s; want 400 containing %q", w.Code, w.Body, tc.message)
				}
				if got := runner.takeInputs(); len(got) != 0 {
					t.Fatalf("invalid request reached runner: %v", got)
				}
			}
		})
	}
}
