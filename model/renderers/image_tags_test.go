package renderers

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkRenderContentWithImageTags(b *testing.B) {
	for _, count := range []int{16, 128, 512} {
		for _, placed := range []bool{false, true} {
			name := "prefix"
			if placed {
				name = "placeholders"
			}
			b.Run(fmt.Sprintf("%s/%d", name, count), func(b *testing.B) {
				content := "text"
				var want strings.Builder
				if placed {
					content = "start" + strings.Repeat(" [img]", count) + " end"
					want.WriteString("start")
				}
				for i := range count {
					if placed {
						want.WriteByte(' ')
					}
					fmt.Fprintf(&want, "[img-%d]", i)
				}
				if placed {
					want.WriteString(" end")
				} else {
					want.WriteString(" " + content)
				}
				got, offset := renderContentWithImageTags(content, count, 0)
				if got != want.String() || offset != count {
					b.Fatal("incorrect fixture output")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					renderContentWithImageTags(content, count, 0)
				}
			})
		}
	}
}

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
			name:        "leaves content unchanged without images",
			content:     "literal [img] text",
			imageOffset: 3,
			want:        "literal [img] text",
			wantOffset:  3,
		},
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
		{
			name:        "preserves mixed indexed and unindexed placeholders",
			content:     "[img-99] and [img]",
			imageCount:  2,
			imageOffset: 3,
			want:        "[img-99] and [img]",
			wantOffset:  5,
		},
		{
			name:       "preserves leading whitespace after prefixes",
			content:    "\t[img]",
			imageCount: 2,
			want:       "[img-1]\t[img-0]",
			wantOffset: 2,
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

func TestRenderContentWithManyImageTags(t *testing.T) {
	const count, offset = 256, 7
	for _, slots := range []int{0, 128, 300} {
		t.Run(fmt.Sprint(slots), func(t *testing.T) {
			content := "start" + strings.Repeat(" [img]", slots) + " end"
			want := content
			var prefix strings.Builder
			for i := range count {
				tag := fmt.Sprintf("[img-%d]", offset+i)
				if strings.Contains(want, "[img]") {
					want = strings.Replace(want, "[img]", tag, 1)
				} else {
					prefix.WriteString(tag)
				}
			}
			if prefix.Len() > 0 {
				want = prefix.String() + " " + want
			}
			got, gotOffset := renderContentWithImageTags(content, count, offset)
			if got != want || gotOffset != offset+count {
				t.Fatalf("unexpected content or offset: got offset %d", gotOffset)
			}
		})
	}
}
