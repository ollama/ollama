package tokenizer

// addedTokenMatcher is a byte trie, built at load time and read-only during
// encoding. Matching walks token prefixes rather than rescanning the remaining
// input for every added token after each match.
type addedTokenMatcher struct {
	children map[byte]*addedTokenMatcher
	token    string // Original content, even when the matching spelling is normalized.
}

func (m *addedTokenMatcher) add(pattern, token string) {
	if pattern == "" {
		return
	}
	for i := range len(pattern) {
		if m.children == nil {
			m.children = make(map[byte]*addedTokenMatcher)
		}
		if m.children[pattern[i]] == nil {
			m.children[pattern[i]] = &addedTokenMatcher{}
		}
		m = m.children[pattern[i]]
	}
	if m.token == "" {
		m.token = token
	}
}

func (m *addedTokenMatcher) split(s string) []encodeChunk {
	if s == "" {
		return nil
	}
	if len(m.children) == 0 {
		return []encodeChunk{{text: s}}
	}

	var result []encodeChunk
	start := 0
	for i := 0; i < len(s); {
		end, token := i, ""
		node := m
		for j := i; j < len(s); j++ {
			node = node.children[s[j]]
			if node == nil {
				break
			}
			if node.token != "" {
				end, token = j+1, node.token
			}
		}
		if token == "" {
			i++
			continue
		}
		// Keep the longest match at the earliest position.
		if start < i {
			result = append(result, encodeChunk{text: s[start:i]})
		}
		result = append(result, encodeChunk{text: token, isSpecial: true})
		i, start = end, end
	}
	if start < len(s) {
		result = append(result, encodeChunk{text: s[start:]})
	}
	return result
}
