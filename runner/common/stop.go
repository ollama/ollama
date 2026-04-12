package common

import (
	"strings"
)

func FindStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

// FindStopAfterAppend finds stop sequences that could have become visible after
// appending the last piece to a previously checked sequence.
func FindStopAfterAppend(pieces []string, stops []string) (bool, string) {
	if len(pieces) == 0 {
		return false, ""
	}

	last := pieces[len(pieces)-1]
	maxStop := 0
	for _, stop := range stops {
		if stop == "" || strings.Contains(last, stop) {
			return true, stop
		}
		if len(stop) > maxStop {
			maxStop = len(stop)
		}
	}

	if maxStop <= 1 || len(pieces) == 1 {
		return false, ""
	}

	tail := joinPreviousSuffixAndLast(pieces, maxStop-1)
	for _, stop := range stops {
		if strings.Contains(tail, stop) {
			return true, stop
		}
	}

	return false, ""
}

func ContainsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

func ContainsStopSuffixInPieces(pieces []string, stops []string) bool {
	maxStop := 0
	for _, stop := range stops {
		if len(stop) > maxStop {
			maxStop = len(stop)
		}
	}

	return ContainsStopSuffix(suffixString(pieces, maxStop), stops)
}

// TruncateStop removes the provided stop string from pieces,
// returning the partial pieces with stop removed, including truncating
// the last piece if required (and signalling if this was the case)
func TruncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

	// Split truncated string back into pieces of original lengths
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string
	tokenTruncated := false
	start := 0
	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
			tokenTruncated = true
		}
		result = append(result, joined[start:end])
		start = end
	}

	return result, tokenTruncated
}

func IncompleteUnicodeInPieces(pieces []string) bool {
	return IncompleteUnicode(suffixString(pieces, 4))
}

func IncompleteUnicode(token string) bool {
	incomplete := false

	// check if there is incomplete UTF-8 character at the end
	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			// continuation byte: 10xxxxxx
			continue
		}

		if (c & 0xe0) == 0xc0 {
			// 2-byte character: 110xxxxx ...
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			// 3-byte character: 1110xxxx ...
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			// 4-byte character: 11110xxx ...
			incomplete = i < 4
		}

		// else 1-byte character or invalid byte
		break
	}

	return incomplete
}

func joinPreviousSuffixAndLast(pieces []string, n int) string {
	if len(pieces) == 0 {
		return ""
	}

	last := pieces[len(pieces)-1]
	prefix := suffixString(pieces[:len(pieces)-1], n)
	if prefix == "" {
		return last
	}

	var b strings.Builder
	b.Grow(len(prefix) + len(last))
	b.WriteString(prefix)
	b.WriteString(last)
	return b.String()
}

func suffixString(pieces []string, n int) string {
	if n <= 0 || len(pieces) == 0 {
		return ""
	}

	total := 0
	start := len(pieces)
	for start > 0 && total < n {
		start--
		total += len(pieces[start])
	}
	if total == 0 {
		return ""
	}

	skip := total - n
	if skip < 0 {
		skip = 0
	}

	var b strings.Builder
	b.Grow(total - skip)
	for _, piece := range pieces[start:] {
		if skip >= len(piece) {
			skip -= len(piece)
			continue
		}
		if skip > 0 {
			piece = piece[skip:]
			skip = 0
		}
		b.WriteString(piece)
	}
	return b.String()
}
