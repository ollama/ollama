package renderers

import "testing"

func TestRenderContentWithImageTags(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		imageCount  int
		imageOffset int
		want        string
		wantOffset  int
	}{
		{
			name:        "prefixes when there are no placeholders",
			content:     "describe this image",
			imageCount:  2,
			imageOffset: 0,
			want:        "[img-0][img-1] describe this image",
			wantOffset:  2,
		},
		{
			name:        "replaces explicit placeholders in order",
			content:     "compare [img] and [img]",
			imageCount:  2,
			imageOffset: 3,
			want:        "compare [img-3] and [img-4]",
			wantOffset:  5,
		},
		{
			name:        "prefixes extra images after placeholders are exhausted",
			content:     "compare [img]",
			imageCount:  2,
			imageOffset: 0,
			want:        "[img-1] compare [img-0]",
			wantOffset:  2,
		},
		{
			name:        "leaves leftover placeholders when there are fewer images",
			content:     "compare [img] and [img]",
			imageCount:  1,
			imageOffset: 0,
			want:        "compare [img-0] and [img]",
			wantOffset:  1,
		},
		{
			name:        "preserves already-numbered placeholders",
			content:     "compare [img-0] and [img-1]",
			imageCount:  2,
			imageOffset: 0,
			want:        "compare [img-0] and [img-1]",
			wantOffset:  2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotOffset := renderContentWithImageTags(tt.content, tt.imageCount, tt.imageOffset)
			if got != tt.want {
				t.Fatalf("content = %q, want %q", got, tt.want)
			}
			if gotOffset != tt.wantOffset {
				t.Fatalf("offset = %d, want %d", gotOffset, tt.wantOffset)
			}
		})
	}
}
