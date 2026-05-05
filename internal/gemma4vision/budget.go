// Package gemma4vision implements Gemma 4 visual token budget helpers.
// See https://ai.google.dev/gemma/docs/capabilities/vision (visual token budgets).
package gemma4vision

// VisualTokenLadder is the official Gemma 4 visual token budget rung set.
var VisualTokenLadder = []int{70, 140, 280, 560, 1120}

const (
	DefaultMinVisualTokens = 70
	DefaultMaxVisualTokens = 560
	MaxVisualTokens        = 1120
)

// SnapGemma4VisualTokens maps n to the nearest value on VisualTokenLadder.
// For ties, the lower rung is chosen (less VRAM).
// n <= 0 returns DefaultMinVisualTokens. n >= MaxVisualTokens returns MaxVisualTokens.
func SnapGemma4VisualTokens(n int) int {
	if n <= 0 {
		return DefaultMinVisualTokens
	}
	if n >= MaxVisualTokens {
		return MaxVisualTokens
	}
	best := VisualTokenLadder[0]
	bestDist := abs(n - best)
	for _, v := range VisualTokenLadder[1:] {
		d := abs(n - v)
		if d < bestDist || (d == bestDist && v < best) {
			best = v
			bestDist = d
		}
	}
	return best
}

// NormalizeGemma4ImageBudgets converts raw API ints (0 = unset per side) to snapped
// ladder min/max token counts, then enforces minTok <= maxTok.
// Negative raw values are treated as unset (0) for that side.
func NormalizeGemma4ImageBudgets(minRaw, maxRaw int) (minTok, maxTok int) {
	if minRaw < 0 {
		minRaw = 0
	}
	if maxRaw < 0 {
		maxRaw = 0
	}
	if minRaw == 0 {
		minTok = DefaultMinVisualTokens
	} else {
		minTok = SnapGemma4VisualTokens(minRaw)
	}
	if maxRaw == 0 {
		maxTok = DefaultMaxVisualTokens
	} else {
		maxTok = SnapGemma4VisualTokens(maxRaw)
	}
	if minTok > maxTok {
		maxTok = minTok
	}
	return minTok, maxTok
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
