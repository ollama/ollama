package create

import "testing"

func TestQuantizeBlobMetadata(t *testing.T) {
	tests := []struct {
		name         string
		items        []quantizeItem
		wantMixed    bool
		wantMetadata map[string]string
	}{
		{
			name: "uniform quantized",
			items: []quantizeItem{
				{name: "a", quantize: "mxfp8"},
				{name: "b", quantize: "mxfp8"},
			},
			wantMetadata: map[string]string{"quant_type": "mxfp8", "group_size": "32"},
		},
		{
			name: "plain tensor before quantized tensors is mixed",
			items: []quantizeItem{
				{name: "a"},
				{name: "b", quantize: "mxfp8"},
				{name: "c", quantize: "mxfp8"},
			},
			wantMixed: true,
		},
		{
			name: "quantized tensor before plain tensor is mixed",
			items: []quantizeItem{
				{name: "a", quantize: "mxfp8"},
				{name: "b"},
			},
			wantMixed: true,
		},
		{
			name: "different quantized types are mixed",
			items: []quantizeItem{
				{name: "a", quantize: "mxfp8"},
				{name: "b", quantize: "nvfp4"},
			},
			wantMixed: true,
		},
		{
			name: "unquantized only has no quant metadata",
			items: []quantizeItem{
				{name: "a"},
				{name: "b"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMetadata, gotMixed := quantizeBlobMetadata(tt.items)
			if gotMixed != tt.wantMixed {
				t.Fatalf("mixed = %v, want %v", gotMixed, tt.wantMixed)
			}
			if len(gotMetadata) != len(tt.wantMetadata) {
				t.Fatalf("metadata = %v, want %v", gotMetadata, tt.wantMetadata)
			}
			for k, want := range tt.wantMetadata {
				if got := gotMetadata[k]; got != want {
					t.Fatalf("metadata[%q] = %q, want %q (all metadata: %v)", k, got, want, gotMetadata)
				}
			}
		})
	}
}
