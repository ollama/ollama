package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestQuantizeQ8_0_MatchesGGML(t *testing.T) {
	const N = 1 << 20

	src := make([]float32, N)
	for i := range src {
		src[i] = float32(i%2000-1000) * 0.001
	}

	goOut := quantizeQ8_0(src)
	goHash := sha256.Sum256(goOut)

	ggmlLib := filepath.Join("..", "build", "lib", "ollama")
	if _, err := os.Stat(filepath.Join(ggmlLib, "libggml-base.so")); err != nil {
		t.Skipf("ggml not built (run cmake --preset CPU && cmake --build build --preset CPU): %v", err)
	}

	ggmlSrc := filepath.Join("..", "ml", "backend", "ggml", "ggml", "src")
	ggmlInc := filepath.Join("..", "ml", "backend", "ggml", "ggml", "include")

	dir := t.TempDir()
	harness := filepath.Join(dir, "q8_harness")

	cCode := fmt.Sprintf(`#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-quants.h"
int main() {
    const int N = %d;
    float *data = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        data[i] = (float)(i %% 2000 - 1000) * 0.001f;
    int nb = N / 32;
    size_t bsz = sizeof(block_q8_0);
    void *out = calloc(nb, bsz);
    quantize_row_q8_0_ref(data, out, N);
    fwrite(out, bsz, nb, stdout);
    free(data); free(out);
    return 0;
}`, N)

	cSrc := filepath.Join(dir, "harness.c")
	if err := os.WriteFile(cSrc, []byte(cCode), 0o644); err != nil {
		t.Fatal(err)
	}

	compileArgs := []string{
		"-O2", "-o", harness, cSrc,
		"-I" + ggmlInc, "-I" + ggmlSrc,
		"-L" + ggmlLib, "-lggml-base",
		"-lm", "-lpthread",
		"-Wl,-rpath," + ggmlLib,
	}
	if out, err := exec.Command("gcc", compileArgs...).CombinedOutput(); err != nil {
		t.Fatalf("compile failed: %v\n%s", err, out)
	}

	cOutput, err := exec.Command(harness).Output()
	if err != nil {
		t.Fatalf("harness failed: %v", err)
	}

	cHash := sha256.Sum256(cOutput)

	t.Logf("Go   output: %d bytes, SHA-256: %s", len(goOut), hex.EncodeToString(goHash[:]))
	t.Logf("GGML output: %d bytes, SHA-256: %s", len(cOutput), hex.EncodeToString(cHash[:]))

	if goHash != cHash {
		for i := range min(len(goOut), len(cOutput)) {
			if goOut[i] != cOutput[i] {
				block := i / 34
				pos := i % 34
				t.Logf("first mismatch at byte %d (block %d, offset %d): Go=0x%02x GGML=0x%02x", i, block, pos, goOut[i], cOutput[i])
				break
			}
		}
		t.Fatalf("HASH MISMATCH: Go Q8_0 output does not match ggml's quantize_row_q8_0_ref")
	}
}
