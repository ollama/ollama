//go:build mlx

package grammar

// terminalTokenGroups contains pre-partitioned tokens for a terminal.
// This enables O(1) lookup of tokens that exactly match vs need DP validation.
type terminalTokenGroups struct {
	// ExactMatches are tokens that exactly match this terminal (O(1) validation)
	ExactMatches []int32

	// DPCandidates are tokens that start with this terminal but need DP validation
	DPCandidates []int
}

// tokenAnalysis contains precomputed terminal matches for a token
type tokenAnalysis struct {
	// The token string
	Token string

	// TokenID in the vocabulary
	TokenID int

	// Matches at each byte position
	// MatchesAtPos[i] = terminals matching at position i with their lengths
	MatchesAtPos [][]terminalMatch

	// Fast path: if token exactly matches one terminal
	// -1 if no exact match
	exactMatch int

	// Whether this token can be consumed at all (has at least one match)
	HasMatches bool
}

// analyzer precomputes terminal matches for a vocabulary
type analyzer struct {
	matcher  *terminalMatcher
	analyses []tokenAnalysis // Indexed by token ID
	vocab    []string

	// Pre-partitioned tokens by terminal (exact match vs DP candidates)
	// This enables direct slice appends instead of per-token branching
	tokensByTerminal []terminalTokenGroups
}

// newAnalyzer creates an analyzer for the given vocabulary and terminals
func newAnalyzer(vocab []string, matcher *terminalMatcher) *analyzer {
	a := &analyzer{
		matcher:  matcher,
		analyses: make([]tokenAnalysis, len(vocab)),
		vocab:    vocab,
	}

	// Precompute analysis for each token
	for i, token := range vocab {
		a.analyses[i] = a.analyze(token, i)
	}

	// Build pre-partitioned token groups for fast ApplyMask
	a.buildTokenPartitions()

	return a
}

// analyze computes terminal matches for a single token
func (a *analyzer) analyze(token string, tokenID int) tokenAnalysis {
	analysis := tokenAnalysis{
		Token:        token,
		TokenID:      tokenID,
		MatchesAtPos: make([][]terminalMatch, len(token)),
		exactMatch:   -1,
		HasMatches:   false,
	}

	if len(token) == 0 {
		return analysis
	}

	// Compute matches at each position
	data := []byte(token)
	for pos := 0; pos < len(data); pos++ {
		matches := a.matcher.matchesAt(data, pos)
		analysis.MatchesAtPos[pos] = matches
		if len(matches) > 0 {
			analysis.HasMatches = true
		}
	}

	// Exact match is only valid when a single terminal spans the entire token
	if len(analysis.MatchesAtPos) > 0 {
		var exactID int = -1
		for _, match := range analysis.MatchesAtPos[0] {
			if match.Length != len(token) {
				continue
			}
			if exactID >= 0 && exactID != match.TerminalID {
				exactID = -1
				break
			}
			exactID = match.TerminalID
		}
		analysis.exactMatch = exactID
	}

	return analysis
}

// analysis returns the precomputed analysis for a token ID
func (a *analyzer) analysis(tokenID int) tokenAnalysis {
	if tokenID < 0 || tokenID >= len(a.analyses) {
		return tokenAnalysis{exactMatch: -1}
	}
	return a.analyses[tokenID]
}

// vocabSize returns the vocabulary size
func (a *analyzer) vocabSize() int {
	return len(a.vocab)
}

// buildTokenPartitions pre-partitions tokens into exact-match vs needs-DP groups per terminal.
// This enables ApplyMask to use direct slice appends instead of per-token branching.
func (a *analyzer) buildTokenPartitions() {
	numTerminals := a.matcher.terminalCount()
	a.tokensByTerminal = make([]terminalTokenGroups, numTerminals)

	for tokenID, analysis := range a.analyses {
		if !analysis.HasMatches {
			continue
		}

		if analysis.exactMatch >= 0 {
			// Token exactly matches one terminal - fast path (O(1) validation)
			tid := analysis.exactMatch
			a.tokensByTerminal[tid].ExactMatches = append(
				a.tokensByTerminal[tid].ExactMatches, int32(tokenID))
		} else {
			// Token needs DP validation - add to all terminals it can start with
			// This way, when a terminal is valid, we know exactly which tokens need DP
			if len(analysis.MatchesAtPos) > 0 {
				seen := make(map[int]bool)
				for _, match := range analysis.MatchesAtPos[0] {
					tid := match.TerminalID
					if !seen[tid] {
						seen[tid] = true
						a.tokensByTerminal[tid].DPCandidates = append(
							a.tokensByTerminal[tid].DPCandidates, tokenID)
					}
				}
			}
		}
	}
}

// terminalGroups returns the pre-partitioned token groups for a terminal ID
func (a *analyzer) terminalGroups(terminalID int) terminalTokenGroups {
	if terminalID < 0 || terminalID >= len(a.tokensByTerminal) {
		return terminalTokenGroups{}
	}
	return a.tokensByTerminal[terminalID]
}
