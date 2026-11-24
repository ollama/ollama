package mlxrunner

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlxrunner/model"
)

func TestPrefillPlan(t *testing.T) {
	atomic := &model.PreparedItem{}
	causal := &model.PreparedItem{Causal: true}
	for _, tc := range []struct {
		name   string
		length int
		items  []mediaItem
		ends   []int
	}{
		{"decode seed only", 1, nil, nil},
		{"short prompt", 3, nil, []int{2}},
		{"final snapshot", 2048, nil, []int{2044, 2047}},
		{"periodic snapshot", 8200, nil, []int{2048, 4096, 6144, 8192, 8196, 8199}},
		{"atomic media crosses batch", 4097, []mediaItem{{pos: 2000, length: 800, item: atomic}}, []int{2000, 4048, 4093, 4096}},
		{"atomic media crosses snapshot", 11200, []mediaItem{{pos: 8000, length: 3000, item: atomic}}, []int{2048, 4096, 6144, 8000, 11000, 11196, 11199}},
		{"atomic media crosses final snapshot", 2048, []mediaItem{{pos: 2038, length: 8, item: atomic}}, []int{2046, 2047}},
		{"causal media can span batches", 4100, []mediaItem{{pos: 2040, length: 20, item: causal}}, []int{2048, 4096, 4099}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := planPrefill(tc.length, tc.items); !slices.Equal(got, tc.ends) {
				t.Fatalf("batch ends = %v, want %v", got, tc.ends)
			}
		})
	}
}

func TestPrefixCacheAlignedPrefill(t *testing.T) {
	const length = 2048
	pc := newRecurrentEnv().pc
	pc.alignPrefill = true
	t.Cleanup(pc.close)

	for _, tc := range []struct {
		token int32
		want  int
	}{{2, 0}, {3, 0}, {4, 0}, {3, length - 4}} {
		prompt := slices.Repeat([]int32{tc.token}, length)
		for i := range prompt[:30] {
			prompt[i] = 1
		}
		s := pc.begin(prompt, nil)
		if got := length - len(s.remaining); got != tc.want {
			t.Fatalf("prompt %d reused %d tokens, want %d", tc.token, got, tc.want)
		}
		if tc.want == 0 && len(s.pendingSnapshots) != 0 {
			t.Fatalf("prompt %d scheduled an interior capture: %v", tc.token, s.pendingSnapshots)
		}
		finishAlignedPrefill(s)
	}
}

func TestPrefixCacheAlignedPrefillReusesCompatibleWidths(t *testing.T) {
	for _, peer := range []bool{false, true} {
		name := "local"
		if peer {
			name = "peer"
		}
		t.Run(name, func(t *testing.T) {
			pc := newRecurrentEnv().pc
			pc.alignPrefill = true
			t.Cleanup(pc.close)
			prompt := slices.Repeat([]int32{1}, prefillSnapshotInterval+8)
			finishAlignedPrefill(pc.begin(prompt, nil))

			// The shared periodic batch keeps its width. The first short
			// terminal batch changes width with the longer prompt.
			longer := append(prompt, slices.Repeat([]int32{2}, 25)...)
			dest := pc
			var peers []*prefixCache
			if peer {
				dest = newRecurrentEnv().pc
				dest.alignPrefill = true
				t.Cleanup(dest.close)
				peers = append(peers, pc)
			}
			s := dest.begin(longer, nil, peers...)
			if got := len(longer) - len(s.remaining); got != prefillSnapshotInterval {
				t.Fatalf("longer prompt reused %d tokens, want %d", got, prefillSnapshotInterval)
			}
			finishAlignedPrefill(s)
		})
	}
}

func finishAlignedPrefill(s *cacheSession) {
	s.scheduleGenerationSnapshots()
	pos := len(s.inputs) - len(s.remaining)
	for _, end := range s.prefill {
		if end > pos {
			feedAll(s.caches, s.inputs[pos:end])
			pos = end
		}
	}
	s.attachPrefillSnapshots()
	feedAll(s.caches, s.inputs[pos:]) // decode consumes the final prompt token
	s.close()
}

// An extension that changes the terminal batch still has the same completed
// full batches, including ones shifted by an indivisible media span.
func TestPrefixCacheRestoresLastFullPrefillBatch(t *testing.T) {
	for _, tc := range []struct {
		name  string
		media []mediaItem
		want  int
	}{
		{"raw", nil, 4096},
		{"atomic media", []mediaItem{{pos: 1900, length: 400, fold: 42, item: &model.PreparedItem{}}}, 3948},
	} {
		for _, peer := range []bool{false, true} {
			name := "local"
			if peer {
				name = "peer"
			}
			t.Run(tc.name+"/"+name, func(t *testing.T) {
				src := newRecurrentEnv().pc
				t.Cleanup(src.close)
				src.alignPrefill = true
				original := slices.Repeat([]int32{1}, 4400)
				finishAlignedPrefill(src.begin(original, tc.media))
				dst := src
				var peers []*prefixCache
				if peer {
					dst = newRecurrentEnv().pc
					t.Cleanup(dst.close)
					dst.alignPrefill = true
					peers = []*prefixCache{src}
				}
				longer := append(slices.Clone(original), 2, 3, 4)
				s := dst.begin(longer, tc.media, peers...)
				if got := len(longer) - len(s.remaining); got != tc.want {
					t.Fatalf("extended prompt restored %d tokens, want %d", got, tc.want)
				}
				finishAlignedPrefill(s)
				retry := dst.begin(longer, tc.media)
				if got := len(longer) - len(retry.remaining); got != len(longer)-4 {
					t.Fatalf("identical prompt restored %d tokens, want terminal snapshot at %d", got, len(longer)-4)
				}
				finishAlignedPrefill(retry)
			})
		}
	}
}
