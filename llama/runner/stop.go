package runner

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
// the last piece if required (and signalling if this was the case)
func truncateStop(pieces []CompletionResponse, stop string) ([]CompletionResponse, bool) {
	// Build complete string and find stop position
	var completeStr string
	for _, piece := range pieces {
		completeStr += piece.Content
	}

	stopStart := strings.Index(completeStr, stop)
	if stopStart == -1 {
		return pieces, false
	}

	// Build result up to stop position
	result := make([]CompletionResponse, 0)
	accumulated := 0

	truncated := false
	for _, piece := range pieces {
		if accumulated+len(piece.Content) <= stopStart {
			result = append(result, piece)
			accumulated += len(piece.Content)
			continue
		}

		if accumulated < stopStart {
			truncPiece := piece
			truncPiece.Content = piece.Content[:stopStart-accumulated]
			if len(truncPiece.Content) > 0 {
				result = append(result, truncPiece)
				truncated = true
			}
		}
		break
	}

	// Signal if we had to truncate the last piece
	return result, truncated
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
