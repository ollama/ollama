package mlxrunner

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"log/slog"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

var imgTagPattern = regexp.MustCompile(`\[img-(\d+)\]`)

// mediaItem is one media occurrence in a request's token stream: the
// absolute position and length of its placeholder expansion, its trie-key
// fold value, and the prepared item.
type mediaItem struct {
	pos    int
	length int
	fold   uint32
	item   *base.PreparedItem
}

// foldValue derives the trie-key substitute for a media item: a hash of the
// raw bytes and the preprocessing dims (which pin the feature geometry for
// given bytes), with bit 31 forced so it can never equal a token ID.
func foldValue(data []byte, dims []int) uint32 {
	h := fnv.New64a()
	h.Write(data)
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		h.Write(b[:])
	}
	sum := h.Sum64()
	return (uint32(sum>>32) ^ uint32(sum)) | 1<<31
}

// requestMedia manages one request's media features: encoded on first
// use, released when the expansion is fully evaluated. A nil
// *requestMedia is a text-only request; every method is nil-safe.
type requestMedia struct {
	model    base.MediaModel
	items    []mediaItem
	inputLen int

	// manifest is the request-scoped batch view of items; Features is
	// toggled in place so every batch shares the same slice.
	manifest []batch.MediaItem
	features []*mlx.Array // parallel to items; nil until encoded

	// layout is the request's one-row Batch.Layout, shared by every batch
	// like the manifest; nil when the model returned no layout.
	layout []any
}

func (r *Runner) openMedia(request Request) *requestMedia {
	if len(request.MediaItems) == 0 {
		return nil
	}
	m := &requestMedia{
		model:    r.Model.(base.MediaModel),
		items:    request.MediaItems,
		inputLen: len(request.Tokens),
		manifest: make([]batch.MediaItem, len(request.MediaItems)),
		features: make([]*mlx.Array, len(request.MediaItems)),
	}
	if request.Layout != nil {
		m.layout = []any{request.Layout}
	}
	for i, item := range m.items {
		m.manifest[i] = batch.MediaItem{Pos: item.pos, Opaque: item.item.Opaque}
	}
	return m
}

// rowLayout returns the request's per-row Batch.Layout value.
func (m *requestMedia) rowLayout() []any {
	if m == nil {
		return nil
	}
	return m.layout
}

func (item *mediaItem) atomic() bool { return !item.item.Causal }

// extendChunk keeps a chunk from ending strictly inside an atomic
// expansion: cut before one starting inside the chunk, else grow to its
// end, clipped one short of the prompt to preserve the decode seed.
func (m *requestMedia) extendChunk(pos, n int) int {
	if m == nil {
		return n
	}
	end := pos + n
	for i := range m.items {
		item := &m.items[i]
		if !item.atomic() {
			continue
		}
		if item.pos < end && end < item.pos+item.length {
			if item.pos > pos {
				return item.pos - pos
			}
			return min(item.pos+item.length, m.inputLen-1) - pos
		}
	}
	return n
}

// batchMedia returns the manifest for chunk [pos, pos+n), encoding and
// pinning each item's features on first overlap; nothing evaluates here —
// the consuming forward pulls the encoder.
func (m *requestMedia) batchMedia(pos, n int) []batch.MediaItem {
	if m == nil {
		return nil
	}
	for i, item := range m.items {
		if item.pos >= pos+n || item.pos+item.length <= pos {
			continue
		}
		if m.features[i] == nil {
			data := mlx.FromValues(item.item.MediaData, item.item.Dims...)
			m.features[i] = m.model.EncodeMedia(item.item, data)
			mlx.Pin(m.features[i])
			// The upload copied the pixels; free them here — release never
			// passes the end of an expansion reaching the prompt's last token.
			item.item.MediaData = nil
		}
		m.manifest[i].Features = m.features[i]
	}
	return m.manifest
}

// release frees what items fully evaluated or restored at position pos no
// longer need: the pinned features and the preprocessed pixel buffer.
func (m *requestMedia) release(pos int) {
	if m == nil {
		return
	}
	for i, item := range m.items {
		if item.pos+item.length <= pos {
			item.item.MediaData = nil
			if m.features[i] != nil {
				mlx.Unpin(m.features[i])
				m.features[i] = nil
				m.manifest[i].Features = nil
			}
		}
	}
}

// close unpins whatever remains when the pipeline exits.
func (m *requestMedia) close() {
	if m == nil {
		return
	}
	for i, f := range m.features {
		if f != nil {
			mlx.Unpin(f)
			m.features[i] = nil
			m.manifest[i].Features = nil
		}
	}
}

// expandMedia tokenizes the [img-N]-tagged prompt into segments, expands
// them in a single PrepareMedia call, and validates the authored items
// before keying cache identity on them.
func (r *Runner) expandMedia(mm base.MediaModel, prompt string, media []llm.MediaData) (*base.PreparedRequest, []mediaItem, error) {
	matches := imgTagPattern.FindAllStringSubmatch(prompt, -1)
	parts := imgTagPattern.Split(prompt, -1)

	referenced := make([]bool, len(media))
	var segments []base.Segment
	for i, part := range parts {
		segments = append(segments, base.Segment{Tokens: r.Tokenizer.Encode(part, i == 0 && r.Tokenizer.AddBOS())})
		if i >= len(matches) {
			continue
		}

		id, _ := strconv.Atoi(matches[i][1])
		idx := -1
		for j := range media {
			if media[j].ID == id {
				idx = j
				break
			}
		}
		if idx < 0 {
			return nil, nil, fmt.Errorf("invalid image index: %d", id)
		}
		referenced[idx] = true
		segments = append(segments, base.Segment{Kind: string(media[idx].Kind), Data: media[idx].Data})
	}

	for j := range media {
		if !referenced[j] {
			slog.Warn("media not referenced by prompt", "id", media[j].ID)
		}
	}

	prepared, err := mm.PrepareMedia(segments)
	if err != nil {
		return nil, nil, err
	}
	items, err := bindItems(prepared, segments)
	if err != nil {
		return nil, nil, err
	}
	return prepared, items, nil
}

// bindItems validates the authored ranges before cache identity is keyed
// on them and binds each item to its source segment's bytes.
func bindItems(prepared *base.PreparedRequest, segments []base.Segment) ([]mediaItem, error) {
	covered := make([]bool, len(segments))
	items := make([]mediaItem, 0, len(prepared.Items))
	end := 0
	for i := range prepared.Items {
		item := &prepared.Items[i]
		rg := item.Range
		if rg[0] < end || rg[1] <= rg[0] || rg[1] > len(prepared.Tokens) {
			return nil, fmt.Errorf("media expansion has invalid range %v", rg)
		}
		if item.Source < 0 || item.Source >= len(segments) || segments[item.Source].Data == nil {
			return nil, fmt.Errorf("media expansion references non-media segment %d", item.Source)
		}
		covered[item.Source] = true
		end = rg[1]

		items = append(items, mediaItem{
			pos:    rg[0],
			length: rg[1] - rg[0],
			fold:   foldValue(segments[item.Source].Data, item.Dims),
			item:   item,
		})
	}
	for s, seg := range segments {
		if seg.Data != nil && !covered[s] {
			return nil, errors.New("media expansion produced no tokens")
		}
	}
	return items, nil
}
