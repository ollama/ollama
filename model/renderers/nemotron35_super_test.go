package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestNemotronSuperImageLabels(t *testing.T) {
	for _, tc := range []struct {
		name, content string
		count         int
		want          string
	}{
		{"one", "Describe.", 1, "[img-0]Describe."},
		{"two", "Compare.", 2, "<image 1>[img-0]<image 2>[img-1]Compare."},
		{"interleaved", "First [img] then [img] end.", 2, "First <image 1>[img-0] then <image 2>[img-1] end."},
		{"explicit source placeholders", "First <image> then <image> end.", 2, "First [img-0] then [img-1] end."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := []api.Message{{Role: "user", Content: tc.content, Images: make([]api.ImageData, tc.count)}}
			got, err := RenderWithRenderer("nemotron-3.5-super", messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + tc.want + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatalf("render mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
