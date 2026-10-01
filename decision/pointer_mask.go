package decision

// PointerAllow is the attention allow-rule llama.cpp writes into the KQ mask.
// Token i may attend to token j. Segment 0 is the bidirectional state. A
// positive segment is one isolated causal question branch. Negative is padding.
// The diagonal is always allowed. Index order is not the RoPE position.
func PointerAllow(seg []int32, i, j int, stateBidir bool) bool {
	if i == j {
		return true
	}
	n := len(seg)
	if i < 0 || j < 0 || i >= n || j >= n || seg[j] < 0 {
		return false
	}
	if j <= i && (seg[j] == 0 || seg[j] == seg[i]) {
		return true
	}
	return stateBidir && seg[i] == 0 && seg[j] == 0
}
