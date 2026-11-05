package mlxrunner

import "strings"

// stopBuffer holds response chunks until a possible stop prefix is resolved.
// Whole chunks preserve token logprobs; only a token crossing a matched stop
// is truncated, and its logprobs are omitted rather than assigned to a fragment.
type stopBuffer struct {
	stops   []string
	pending []CompletionResponse
}

func (b *stopBuffer) add(response CompletionResponse) ([]CompletionResponse, bool) {
	if len(b.stops) == 0 {
		return []CompletionResponse{response}, false
	}
	b.pending = append(b.pending, response)
	var text strings.Builder
	for _, part := range b.pending {
		text.WriteString(part.Content)
	}
	content := text.String()
	match, hold := -1, 0
	for _, stop := range b.stops {
		if stop == "" {
			continue
		}
		if index := strings.Index(content, stop); index >= 0 && (match < 0 || index < match) {
			match = index
		}
		for size := min(len(content), len(stop)-1); size > hold; size-- {
			if strings.HasSuffix(content, stop[:size]) {
				hold = size
				break
			}
		}
	}
	if match >= 0 {
		ready := b.take(match, true)
		b.pending = nil
		return ready, true
	}
	return b.take(len(content)-hold, false), false
}

func (b *stopBuffer) take(size int, split bool) []CompletionResponse {
	var ready []CompletionResponse
	for len(b.pending) > 0 {
		part := b.pending[0]
		if len(part.Content) > size {
			if split && size > 0 {
				part.Content = part.Content[:size]
				part.Logprobs = nil
				ready = append(ready, part)
			}
			break
		}
		ready = append(ready, part)
		size -= len(part.Content)
		b.pending = b.pending[1:]
	}
	return ready
}

func (b *stopBuffer) flush() []CompletionResponse {
	ready := b.pending
	b.pending = nil
	return ready
}
