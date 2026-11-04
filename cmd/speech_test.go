package cmd

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestSpeechCut(t *testing.T) {
	long := strings.Repeat("This sentence is long enough to say. ", 3)
	for _, tc := range []struct {
		text string
		want int
	}{
		{"Too short to say yet. ", 0},
		{"Heading\n\nBody", 7},
		{long + "And then", len(long) - 1},
		{"no sentence end yet " + strings.Repeat("words ", 20), 0},
	} {
		if got := speechCut(tc.text); got != tc.want {
			t.Errorf("speechCut(%q) = %d, want %d", tc.text, got, tc.want)
		}
	}
}

// Speech starts once a few sentences arrive, before the input ends.
func TestSpeechPiecesStream(t *testing.T) {
	r, w := io.Pipe()
	pieces := speechPieces(t.Context(), r)
	go w.Write([]byte(strings.Repeat("This sentence is long enough to say. ", 3) + "Unfinished"))
	select {
	case p := <-pieces:
		if !strings.HasSuffix(p, "to say.") {
			t.Errorf("first piece = %q", p)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("speech waited for the end of the input")
	}
	go func() {
		w.Write([]byte(" tail"))
		w.Close()
	}()
	var rest []string
	for p := range pieces {
		rest = append(rest, p)
	}
	if !slices.Equal(rest, []string{"Unfinished tail"}) {
		t.Errorf("rest = %q", rest)
	}
}

func TestSpeechChunks(t *testing.T) {
	for _, tc := range []struct {
		text  string
		limit int
		want  []string
	}{
		{"Hello.", 100, []string{"Hello."}},
		{"One.\n\n Two.\r\n\r\nThree.", 100, []string{"One.", "Two.", "Three."}},
		{"One two. Three four. Five six.", 21, []string{"One two. Three four.", "Five six."}},
		{"one two three four", 10, []string{"one two", "three four"}},
		{"abcdefghijkl", 5, []string{"abcde", "fghij", "kl"}},
		{"ééééé", 2, []string{"éé", "éé", "é"}},
		{" \n\n ", 100, nil},
	} {
		if got := speechChunks(tc.text, tc.limit); !slices.Equal(got, tc.want) {
			t.Errorf("speechChunks(%q, %d) = %q, want %q", tc.text, tc.limit, got, tc.want)
		}
	}
}

func TestSynthesize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Input, Voice string }
		if r.URL.Path != "/v1/audio/speech" || json.NewDecoder(r.Body).Decode(&req) != nil || req.Input == "" || req.Voice != "Toby" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"input is required"}}`))
			return
		}
		wav := make([]byte, 48)
		binary.LittleEndian.PutUint32(wav[4:], 0xffffffff)
		binary.LittleEndian.PutUint32(wav[40:], 0xffffffff)
		copy(wav[44:], []byte{1, 0, 2, 0})
		w.Write(wav)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	wav, err := synthesize(t.Context(), "phonon", "Toby", "hello")
	if err != nil || binary.LittleEndian.Uint32(wav[4:]) != 40 || binary.LittleEndian.Uint32(wav[40:]) != 4 {
		t.Fatalf("WAV lengths not recorded: %x, %v", wav, err)
	}
	if _, err := synthesize(t.Context(), "phonon", "Toby", ""); err == nil || err.Error() != "input is required" {
		t.Fatalf("server error lost: %v", err)
	}
}
