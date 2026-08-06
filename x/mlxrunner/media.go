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

// expandMedia tokenizes a prompt whose [img-N] tags reference media items,
// hands the model the resulting segments — text runs and media in stream
// order — for expansion in a single PrepareMedia call, and validates the
// items the model authored before keying cache identity on their ranges.
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

// bindItems validates the ranges the model authored — identity is keyed on
// them, so they must be ordered, non-overlapping, in bounds, and cover every
// media segment — and binds each item to its source segment's bytes.
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
