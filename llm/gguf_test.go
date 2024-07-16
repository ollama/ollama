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

func TestGGUFRewrite(t *testing.T) {
	// to test this GGUF Rewrite, add gguf files to /llm/testdata
	// add the name of the file to the tests slice
	tests := []string{}

	for i := range tests {
		tt := tests[i]
		t.Run(tt, func(t *testing.T) {
			t.Parallel()
			p := filepath.Join("testdata", tt)

			if _, err := os.Stat(p); err != nil {
				t.Fatalf("%s not found", p)
			}

			f, err := os.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			// decode original gguf
			ggml, _, err := decodeGGML(t, f)
			if err != nil {
				t.Fatal(err)
			}

			temp, err := os.CreateTemp("testdata", "2"+tt)
			if err != nil {
				t.Fatal(err)
			}
			defer temp.Close()

			_, ggml2, err := rewriteGGML(t, ggml, temp, f)

			if err != nil {
				t.Fatal(err)
			}

			if diff, diff2, ok := compareGGML(ggml2, ggml, temp, f); !ok {
				if cmp.Diff(diff, diff2) != "" {
					t.Fatalf("diff: \n%s", cmp.Diff(diff, diff2))
				}
			}
		})
	}
}

func compareGGML(ggml1, ggml2 *GGML, f *os.File, f2 *os.File) (map[string]string, map[string]string, bool) {
	diff := make(map[string]string)
	diff2 := make(map[string]string)

	kv1 := ggml1.KV()
	kv2 := ggml2.KV()

	if len(kv1) != len(kv2) {
		diff["lenKV"] = fmt.Sprintf("kv1: %d, kv2: %d", len(kv1), len(kv2))
		fmt.Println("lenKV", diff["lenKV"])
	}

	for k, v := range kv1 {
		switch t := v.(type) {
		case *array:
			if diffy := cmp.Diff(t.values, kv2[k].(*array).values); diffy != "" {
				diff[k] = diffy
			}
		default:
			if v != kv2[k] {
				diff[k] = fmt.Sprintf("kv1: %v, kv2: %v", v, kv2[k])
			}
		}
	}

	t1 := ggml1.Tensors()
	t2 := ggml2.Tensors()

	if len(t1.Items) != len(t2.Items) {
		diff["lenTensors"] = fmt.Sprintf("t1: %d, t2: %d", len(t1.Items), len(t2.Items))
	}

	for _, tensor := range t1.Items {
		sha256sum := sha256.New()
		sr := io.NewSectionReader(f, t1.Offset+int64(tensor.Offset), int64(tensor.Size()))
		var s int64
		s, err := io.Copy(sha256sum, sr)
		if err != nil {
			fmt.Println(err)
		}

		diff[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
		diff[tensor.Name+" size"] = fmt.Sprintf("%d", s)
		diff[tensor.Name+" offset"] = fmt.Sprintf("%v", tensor.Offset)
	}

	/* sha256Sum2 := sha256.New()
	sr1 := io.NewSectionReader(f2, 0, n)
	s1, err := io.Copy(sha256Sum2, sr1)
	if err != nil {
		return nil, nil, true
	}

	sha256Sum3 := sha256.New()
	sr2 := io.NewSectionReader(f, 0, n)
	s2, err := io.Copy(sha256Sum3, sr2)
	if err != nil {
		return nil, nil, true
	}

	diff["sha"] = fmt.Sprintf("%d", s1)
	diff2["sha"] = fmt.Sprintf("%d", s2) */

	for _, tensor := range t2.Items {
		sha256sum := sha256.New()
		var s int64
		sr := io.NewSectionReader(f2, t1.Offset+int64(tensor.Offset), int64(tensor.Size()))
		s, err := io.Copy(sha256sum, sr)
		if err != nil {
			fmt.Println(err)
		}

		diff2[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
		diff2[tensor.Name+" size"] = fmt.Sprintf("%d", s)
		diff2[tensor.Name+" offset"] = fmt.Sprintf("%v", tensor.Offset)
	}
	return diff, diff2, len(diff) == 0

}
func decodeGGML(t *testing.T, f *os.File) (*GGML, int64, error) {

	ggml, n, err := DecodeGGML(f, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}
	return ggml, n, nil
}

func rewriteGGML(t *testing.T, ggml *GGML, temp *os.File, f *os.File) (int64, *GGML, error) {
	var tensors []*Tensor

	fmt.Println("11111111111111111111111111111111111111111")
	for _, tensor := range ggml.Tensors().Items {
		shape := make([]uint64, len(tensor.Shape))
		for i := range len(tensor.Shape) {
			shape[i] = tensor.Shape[len(tensor.Shape)-i-1]
		}

		fmt.Println("tensors", tensor.Name, shape, tensor.Kind, tensor.Offset)
		fmt.Println(ggml.Tensors().Offset)
		tensors = append(tensors, &Tensor{
			Name:  tensor.Name,
			Kind:  tensor.Kind,
			Shape: shape,

			WriterTo: TensorWriter{
				Reader: io.NewSectionReader(f, ggml.Tensors().Offset+int64(tensor.Offset), int64(tensor.Size())),
			},
		})
	}

	reader := &GGUFWriter{
		KV: ggml.KV(),
		// Update .Tensors
		Tensors: Tensors{
			Items:  tensors,
			Offset: ggml.Tensors().Offset,
		},
	}

	n, err := io.Copy(temp, reader)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(n, " is my offset")
	file, err := os.Open(temp.Name())
	if err != nil {
		t.Fatal(err)
	}

	ggml2, _, err := DecodeGGML(file, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}

	return n, ggml2, nil
}
