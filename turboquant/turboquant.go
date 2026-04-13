// Package turboquant implements primitives for TurboQuant KV cache compression.
//
// TurboQuant is a data-oblivious vector quantization algorithm from Google Research
// (arxiv 2504.19874, ICLR 2026) that compresses high-dimensional vectors using:
//  1. Random orthogonal rotation (Randomized Hadamard Transform)
//  2. Per-coordinate Lloyd-Max optimal scalar quantization
//  3. QJL (Quantized Johnson-Lindenstrauss) residual correction
//
// The rotation is the core technique: it distributes information uniformly across
// all coordinates, transforming any input distribution into a concentrated Beta
// distribution (≈ Gaussian N(0,1/d) in high dimensions). This eliminates outlier
// channels that cause standard block quantization to fail, enabling near-optimal
// compression at 3-4 bits per coordinate with zero accuracy loss.
package turboquant

const (
	BitWidthTQ2 = 2
	BitWidthTQ3 = 3
	BitWidthTQ4 = 4
)

// TurboQuantSeed is the deterministic seed for key rotation and QJL matrices.
// Using a fixed seed ensures reproducibility across model reloads.
const TurboQuantSeed uint64 = 0x5475_7262_6F51_7561 // "TurboQua"

// TurboQuantValueSeed is a separate seed for value rotation.
// Different from key seed to avoid correlation between key and value rotations.
const TurboQuantValueSeed uint64 = 0x5651_616E_7456_616C // "VQuantVal"

func isPowerOf2(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
