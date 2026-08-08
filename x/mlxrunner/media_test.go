package mlxrunner

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// fakeVisionModel expands every image to a fixed soft-token count.
type fakeVisionModel struct {
	soft    int
	decoded [][]byte
}

type fakeVisionInput struct{ soft int }

func (f *fakeVisionInput) SoftTokens() int { return f.soft }

func (f *fakeVisionModel) SupportsVision() bool                { return true }
func (f *fakeVisionModel) VisionTokens() (int32, int32, int32) { return 9001, 9002, 9003 }
func (f *fakeVisionModel) NewVisionInput(data []byte, _ api.Options) (base.VisionInput, error) {
	f.decoded = append(f.decoded, data)
	return &fakeVisionInput{soft: f.soft}, nil
}
func (f *fakeVisionModel) EncodeVision(base.VisionInput) *mlx.Array { return nil }
func (f *fakeVisionModel) MergedEmbeddings(*mlx.Array, []*mlx.Array, [][2]int32) *mlx.Array {
	return nil
}

// runeEncode tokenizes one rune per token; addBOS prepends 1.
func runeEncode(s string, addBOS bool) []int32 {
	var toks []int32
	if addBOS {
		toks = append(toks, 1)
	}
	for _, r := range s {
		toks = append(toks, int32(r))
	}
	return toks
}

func TestExpandMediaSingleImage(t *testing.T) {
	vm := &fakeVisionModel{soft: 3}
	exp, err := expandMedia("ab[img-0]cd", []llm.MediaData{{Data: []byte("payload"), ID: 0, Kind: llm.MediaKindImage}},
		vm, api.Options{}, runeEncode, true)
	if err != nil {
		t.Fatal(err)
	}
	// BOS a b boi img img img eoi c d
	want := []int32{1, 'a', 'b', 9001, 9002, 9002, 9002, 9003, 'c', 'd'}
	if len(exp.Tokens) != len(want) {
		t.Fatalf("tokens = %v, want %v", exp.Tokens, want)
	}
	for i := range want {
		if exp.Tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d (%v)", i, exp.Tokens[i], want[i], exp.Tokens)
		}
	}
	if len(exp.Spans) != 1 || exp.Spans[0] != [2]int32{4, 7} {
		t.Fatalf("spans = %v, want [[4 7]]", exp.Spans)
	}
	if len(exp.Salts) != len(exp.Tokens) {
		t.Fatalf("salts length %d != tokens length %d", len(exp.Salts), len(exp.Tokens))
	}
	for i, s := range exp.Salts {
		inSpan := i >= 4 && i < 7
		if inSpan && s == 0 {
			t.Fatalf("salt[%d] = 0 inside image span", i)
		}
		if !inSpan && s != 0 {
			t.Fatalf("salt[%d] = %d outside image span", i, s)
		}
	}
	if len(exp.Inputs) != 1 || exp.Inputs[0].SoftTokens() != 3 {
		t.Fatalf("inputs = %v", exp.Inputs)
	}
	if string(vm.decoded[0]) != "payload" {
		t.Fatalf("payload not forwarded: %q", vm.decoded[0])
	}
}

func TestExpandMediaSaltsDifferByContent(t *testing.T) {
	a := mediaSalts([]byte("image-a"), 4)
	b := mediaSalts([]byte("image-b"), 4)
	if a[0] == b[0] {
		t.Fatal("different images must diverge at the first soft token")
	}
	a2 := mediaSalts([]byte("image-a"), 4)
	for i := range a {
		if a[i] != a2[i] {
			t.Fatal("identical images must produce identical salts")
		}
	}
	if a[0] == a[1] {
		t.Fatal("salts must vary by position")
	}
}

func TestExpandMediaMultipleImagesAndPrefixMarker(t *testing.T) {
	vm := &fakeVisionModel{soft: 2}
	exp, err := expandMedia("[img-0] x [img-1]", []llm.MediaData{
		{Data: []byte("first"), ID: 0, Kind: llm.MediaKindImage},
		{Data: []byte("second"), ID: 1, Kind: llm.MediaKindImage},
	}, vm, api.Options{}, runeEncode, true)
	if err != nil {
		t.Fatal(err)
	}
	// Marker-first prompt: no text before the first image, so BOS is never
	// emitted by a text segment. boi img img eoi ' ' 'x' ' ' boi img img eoi
	want := []int32{9001, 9002, 9002, 9003, ' ', 'x', ' ', 9001, 9002, 9002, 9003}
	if len(exp.Tokens) != len(want) {
		t.Fatalf("tokens = %v, want %v", exp.Tokens, want)
	}
	for i := range want {
		if exp.Tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, exp.Tokens[i], want[i])
		}
	}
	if len(exp.Spans) != 2 || exp.Spans[0] != [2]int32{1, 3} || exp.Spans[1] != [2]int32{8, 10} {
		t.Fatalf("spans = %v", exp.Spans)
	}
	if exp.Salts[1] == exp.Salts[8] {
		t.Fatal("different images share a salt at their first soft token")
	}
}

func TestExpandMediaUnknownID(t *testing.T) {
	vm := &fakeVisionModel{soft: 2}
	_, err := expandMedia("a[img-7]b", []llm.MediaData{{Data: []byte("x"), ID: 0}}, vm, api.Options{}, runeEncode, false)
	if err == nil || !strings.Contains(err.Error(), "img-7") {
		t.Fatalf("expected unknown-id error, got %v", err)
	}
}

func TestExpandMediaRejectsAudio(t *testing.T) {
	vm := &fakeVisionModel{soft: 2}
	_, err := expandMedia("a[img-0]", []llm.MediaData{{Data: []byte("RIFFxxxxWAVE"), ID: 0, Kind: llm.MediaKindAudio}},
		vm, api.Options{}, runeEncode, false)
	if err == nil || !strings.Contains(err.Error(), "audio") {
		t.Fatalf("expected audio rejection, got %v", err)
	}
}
