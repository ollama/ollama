package llm

import (
	"encoding/base64"
	"testing"
)

// The media markers are positional: llama-server replaces each occurrence of
// the marker with the next entry in multimodal_data. So the data has to be in
// the order the markers appear in the prompt, and the number of markers has
// to match the number of images.
func TestEmbeddingMultimodalPromptOrder(t *testing.T) {
	const marker = "<__ollama_media_test__>"

	runner := &llamaServerRunner{mediaMarker: marker}

	images := [][]byte{[]byte("zero"), []byte("one"), []byte("two")}
	encoded := func(i int) string {
		return base64.StdEncoding.EncodeToString(images[i])
	}

	tests := []struct {
		name      string
		input     string
		wantOrder []int
	}{
		{
			name:      "no image named",
			input:     "just text",
			wantOrder: []int{0, 1, 2},
		},
		{
			name:      "the images named in order",
			input:     "a [img-0] b [img-1] c [img-2]",
			wantOrder: []int{0, 1, 2},
		},
		{
			name:      "the images named out of order",
			input:     "x [img-2] y [img-0] z [img-1]",
			wantOrder: []int{2, 0, 1},
		},
		{
			name:      "the images named in reverse",
			input:     "[img-2][img-1][img-0]",
			wantOrder: []int{2, 1, 0},
		},
		{
			name:      "only the last image named",
			input:     "a b [img-2]",
			wantOrder: []int{0, 1, 2},
		},
		{
			name:      "only the middle image named",
			input:     "a [img-1] b",
			wantOrder: []int{0, 2, 1},
		},
		{
			// "[img-7]" names no image, so it is left in the text and takes no
			// marker of its own. The images it does not name still get one,
			// prepended, so all three are sent.
			name:      "a placeholder that names no image is left alone",
			input:     "a [img-7] b [img-0]",
			wantOrder: []int{1, 2, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := runner.embeddingMultimodalPrompt(tt.input, images)
			if err != nil {
				t.Fatalf("embeddingMultimodalPrompt error: %v", err)
			}

			// Every image is sent, in the order the markers appear.
			if len(got.MultimodalData) != len(tt.wantOrder) {
				t.Fatalf("MultimodalData len = %d, want %d", len(got.MultimodalData), len(tt.wantOrder))
			}
			for i, image := range tt.wantOrder {
				if want := encoded(image); got.MultimodalData[i] != want {
					t.Errorf("MultimodalData[%d] = %q, want %q (image %d)", i, got.MultimodalData[i], want, image)
				}
			}

			// The markers in the prompt are in the same order as the data.
			wantMarkers := 0
			for range tt.wantOrder {
				wantMarkers++
			}
			if n := countMarker(got.PromptString, marker); n != wantMarkers {
				t.Errorf("marker count = %d, want %d", n, wantMarkers)
			}
		})
	}
}

// countMarker counts the occurrences of the marker in the prompt.
func countMarker(s, marker string) int {
	count := 0
	for i := 0; i+len(marker) <= len(s); i++ {
		if s[i:i+len(marker)] == marker {
			count++
			i += len(marker) - 1
		}
	}
	return count
}
