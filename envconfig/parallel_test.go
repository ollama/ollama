package envconfig

import "testing"

func TestNumParallelDefault(t *testing.T) {
	t.Setenv("OLLAMA_NUM_PARALLEL", "")
	if NumParallel() != 0 {
		t.Fatal("unset parallelism must select the backend default")
	}
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")
	if NumParallel() != 1 {
		t.Fatal("explicit serial setting was lost")
	}
}
