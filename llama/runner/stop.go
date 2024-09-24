package main

import (
	"strings"
)

func findStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

func containsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

// truncateStop removes the provided stop string from pieces,
// returning the partial pieces with stop removed, including truncating
// the last piece if required
func truncateStop(pieces []string, stop string) []string {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces
	}

	joined = joined[:index]

	// Split truncated string back into pieces of original lengths
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string
	start := 0
	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
		}
		result = append(result, joined[start:end])
		start = end
	}

	return result
}

func incompleteUnicode(token string) bool {
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
