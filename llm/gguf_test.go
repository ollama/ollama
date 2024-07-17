package llm

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// TestGGUFDecode tests the decoding and rewriting of (unsorted) GGUF files
// To run, add GGUF files to /llm/testdata and add the name of the file to the tests slice
// This creates a temporary file in /llm/testdata that will deleted only if the test passes
// Note: map[Tensor.Name + " offset"] is commented since sorting will reorder the tensors
// Comment out sort.Sort(gguf.Tensors) in gguf.go to test offsets
func TestGGUFRewrite(t *testing.T) {
	tests := []string{
		"phi3.gguf",
	}

	for i := range tests {
		tt := tests[i]
		t.Run(tt, func(t *testing.T) {
			t.Parallel()
			p := filepath.Join("testdata", tt)

			if _, err := os.Stat(p); err != nil {
				t.Skip("file not found", p)
			}

			wantFile, err := os.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer wantFile.Close()

			// decode original gguf
			_, wantGGML, err := decodeGGML(t, wantFile)
			if err != nil {
				t.Fatal(err)
			}

			gotFile, err := os.CreateTemp("testdata", tt)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				gotFile.Close()
				if !t.Failed() {
					os.Remove(gotFile.Name())
				}
			}()

			_, gotGGML, err := rewriteGGML(t, wantGGML, gotFile, wantFile)

			if err != nil {
				t.Fatal(err)
			}

			diff, diff2 := compareGGML(t, gotGGML, wantGGML, gotFile, wantFile) 
			if cmp.Diff(diff, diff2) != "" {
				t.Fatalf("diff: \n%s", cmp.Diff(diff, diff2))
			}
		})
	}
}

func compareGGML(t *testing.T, gotGGML, wantGGML *GGML, f *os.File, f2 *os.File) (map[string]string, map[string]string) {
	got := make(map[string]string)
	want := make(map[string]string)

	gotKV := gotGGML.KV()
	wantKV := wantGGML.KV()

	if len(gotKV) != len(wantKV) {
		t.Fatalf("got length: %d != want length: %d", len(gotKV), len(wantKV))
	}

	for k, v := range gotKV {
		switch t := v.(type) {
		case *array:
			if diffy := cmp.Diff(t.values, wantKV[k].(*array).values); diffy != "" {
				got[k] = diffy
			}
		default:
			if v != wantKV[k] {
				got[k] = fmt.Sprintf("kv1: %v, kv2: %v", v, want[k])
			}
		}
	}

	gotTensors := gotGGML.Tensors().Items
	gotOffset := gotGGML.Tensors().Offset
	wantTensors := wantGGML.Tensors().Items
	wantOffset := wantGGML.Tensors().Offset

	if len(gotTensors) != len(wantTensors) {
		got["lenTensors"] = fmt.Sprintf("t1: %d, t2: %d", len(gotTensors), len(wantTensors))
	}

	for _, tensor := range gotTensors {
		sha256sum := sha256.New()
		sr := io.NewSectionReader(f, gotOffset+int64(tensor.Offset), int64(tensor.Size()))
		var s int64
		s, err := io.Copy(sha256sum, sr)
		if err != nil {
			t.Fatalf("error: %v", err)
		}

		got[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
		got[tensor.Name+" size"] = fmt.Sprintf("%d", s)
		// got[tensor.Name+" offset"] = fmt.Sprintf("%v", tensor.Offset)
	}

	for _, tensor := range wantTensors {
		sha256sum := sha256.New()
		var s int64
		sr := io.NewSectionReader(f2, wantOffset +int64(tensor.Offset), int64(tensor.Size()))
		s, err := io.Copy(sha256sum, sr)
		if err != nil {
			t.Fatalf("error: %v", err)
		}

		want[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
		want[tensor.Name+" size"] = fmt.Sprintf("%d", s)
		// want[tensor.Name+" offset"] = fmt.Sprintf("%v", tensor.Offset)
	}
	return got, want
}

func decodeGGML(t *testing.T, f *os.File) (int64, *GGML, error) {
	ggml, n, err := DecodeGGML(f, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}
	return n, ggml, nil
}

func rewriteGGML(t *testing.T, ggml *GGML, gotFile *os.File, wantFile *os.File) (int64, *GGML, error) {
	var tensors []*Tensor

	for _, tensor := range ggml.Tensors().Items {
		shape := make([]uint64, len(tensor.Shape))
		for i := range len(tensor.Shape) {
			shape[i] = tensor.Shape[len(tensor.Shape)-i-1]
		}

		tensors = append(tensors, &Tensor{
			Name:  tensor.Name,
			Kind:  tensor.Kind,
			Shape: shape,

			WriterTo: TensorWriter{
				Reader: io.NewSectionReader(wantFile, ggml.Tensors().Offset+int64(tensor.Offset), int64(tensor.Size())),
			},
		})
	}

	reader := &GGUFWriter{
		KV: ggml.KV(),
		Tensors: Tensors{
			Items:  tensors,
			Offset: ggml.Tensors().Offset,
		},
	}

	n, err := io.Copy(gotFile, reader)
	if err != nil {
		t.Fatal(err)
	}

	file, err := os.Open(gotFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	ggml2, _, err := DecodeGGML(file, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}

	return n, ggml2, nil
}
