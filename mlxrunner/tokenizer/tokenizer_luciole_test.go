package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

func TestLucioleHuggingFaceParity(t *testing.T) {
	path := os.Getenv("LUCIOLE_TOKENIZER_PATH")
	if path == "" {
		t.Skip("set LUCIOLE_TOKENIZER_PATH to pinned tokenizer.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile("testdata/luciole-parity.json")
	if err != nil {
		t.Fatal(err)
	}
	var refs []struct {
		Text    string  `json:"text"`
		Decoded string  `json:"decoded"`
		IDs     []int32 `json:"ids"`
	}
	if err := json.Unmarshal(data, &refs); err != nil {
		t.Fatal(err)
	}
	for _, ref := range refs {
		if got := tok.Encode(ref.Text, false); !slices.Equal(got, ref.IDs) {
			t.Errorf("%q: ids=%v, HuggingFace=%v", ref.Text, got, ref.IDs)
		}
		if got := tok.Decode(ref.IDs); got != ref.Decoded {
			t.Errorf("decode %q: got %q", ref.Text, got)
		}
		var stream strings.Builder
		for i, id := range ref.IDs {
			if i == 0 {
				stream.WriteString(tok.Decode([]int32{id}))
			} else {
				stream.WriteString(tok.DecodeRaw([]int32{id}))
			}
		}
		if stream.String() != ref.Decoded {
			t.Errorf("stream decode %q: got %q, want %q", ref.Text, stream.String(), ref.Decoded)
		}
	}
}

func TestLucioleNormalizationAndMetaspace(t *testing.T) {
	data := []byte(`{"model":{"type":"BPE","vocab":{"▁":0,"J":1,"'":2,"a":3,"i":4,"m":5,"e":6," ":7,"c":8,"f":9,"é":10,"H":11,"l":12,"o":13,"w":14,"r":15,"d":16,"▁Hello":17,"▁world":18},"merges":[]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i: ?(?:[jtmslndcy]['’]|qu['’]|jusqu['’]|lorsqu['’]|puisqu['’]|quoiqu['’]|aujourd['’]hui)|['’][stmd]|['’][rv]e|['’]ll)| ?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false}]},"decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false},{"type":"Fuse"}]},"normalizer":{"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"Regex":"[‘’ʼ‛＇]"},"content":"'"},{"type":"Replace","pattern":{"String":"ﬁ"},"content":"fi"},{"type":"NFC"}]}}`)
	tok, err := LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("J’aime cafe\u0301 ﬁ", false)
	var want []int32
	for _, r := range "▁J'aime▁café▁fi" {
		want = append(want, tok.vocab.Reverse[string(r)])
	}
	if !slices.Equal(got, want) {
		t.Fatalf("ids=%v, want=%v", got, want)
	}
	if text := tok.Decode(got); text != "J'aime café fi" {
		t.Fatalf("decoded=%q", text)
	}
	hello, world := tok.vocab.Reverse["▁Hello"], tok.vocab.Reverse["▁world"]
	if text := tok.Decode([]int32{hello}) + tok.DecodeRaw([]int32{world}); text != "Hello world" {
		t.Fatalf("streamed text=%q", text)
	}
}
