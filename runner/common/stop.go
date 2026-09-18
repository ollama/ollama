package common

import "strings"

func FindStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
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

func TruncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

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

func IncompleteUnicode(token string) bool {
	incomplete := false

	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			continue
		}

		if (c & 0xe0) == 0xc0 {
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			incomplete = i < 4
		}

		break
	}

	return incomplete
}
