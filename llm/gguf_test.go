package llm

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

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
				t.Fatalf("%s not found", p)
			}

			f, err := os.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			ggml, _, err := decodeGGML(t, f)
			if err != nil {
				t.Fatal(err)
			}

			temp, err := os.CreateTemp("testdata", "2"+tt)
			if err != nil {
				t.Fatal(err)
			}
			defer temp.Close()

			n, ggml2, err := rewriteGGML(t, ggml, temp, f)

			/* if n != m {
				t.Fatalf("n: %d, m: %d", n, m)
			} */

			if err != nil {
				t.Fatal(err)
			}
			//t.Fatal("FULL SIZE JFAKFJJEFJAJFLAEJJAFAJKLFJ", n)

			if diff, diff2, ok := compareGGML(n, ggml2, ggml, temp, f); !ok {
				if cmp.Diff(diff, diff2) != "" {
					t.Fatalf("\n%s,\n%s\n%s\n%s\ndiff: %s", diff["token_embd.weight"], diff2["token_embd.weight"], diff["token_embd.weight size"], diff["token_embd.weight offset"], cmp.Diff(diff, diff2))
				}
			}
		})
	}
}

func formatDiff(diff map[string]string) string {
	var builder strings.Builder
	for k, v := range diff {
		builder.WriteString(fmt.Sprintf("%s: %s\n", k, v))
	}
	return builder.String()
}

func compareGGML(n int64, ggml1, ggml2 *GGML, f *os.File, f2 *os.File) (map[string]string, map[string]string, bool) {
	diff := make(map[string]string)
	diff2 := make(map[string]string)

	kv1 := ggml1.KV()
	kv2 := ggml2.KV()

	if len(kv1) != len(kv2) {
		diff["lenKV"] = fmt.Sprintf("kv1: %d, kv2: %d", len(kv1), len(kv2))
		fmt.Println("lenKV", diff["lenKV"])
	}

	for k, v := range kv1 {
		// if v2, ok := kv2[k]; !ok {
		// diff[k] = fmt.Sprintf("missing key %s", k)
		// } else if v != v2 {
		// diff[fmt.Sprintf("%s type diff", k)] = fmt.Sprintf("kv1 type: %T, kv2 type: %T", v.(*array).values, v2.(*array).values)
		// diff[k] = fmt.Sprintf("kv1: %d, kv2: %d", len(v.(*array).values), len(v2.(*array).values))
		// diff[fmt.Sprintf("%s values first 10", k)] = fmt.Sprintf("\nkv1: %#v, \nkv2: %#v", v.(*array).values[0:10], v2.(*array).values[0:10])
		// diff[fmt.Sprintf("%s values last 10", k)] = fmt.Sprintf("\nkv1: %#v, \nkv2: %#v", v.(*array).values[len(v.(*array).values)-10:], v2.(*array).values[len(v2.(*array).values)-10:])
		// diff[fmt.Sprintf("%s diff", k)] = cmp.Diff(v.(*array).values, v2.(*array).values)

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

		// }
	}

	t1 := ggml1.Tensors()
	t2 := ggml2.Tensors()

	if len(t1) != len(t2) {
		diff["lenTensors"] = fmt.Sprintf("t1: %d, t2: %d", len(t1), len(t2))
	}

	for _, tensor := range t1 {
		sha256sum := sha256.New()
		sr := io.NewSectionReader(f, n+int64(tensor.Offset), int64(tensor.Size()))
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

	for _, tensor := range t2 {
		sha256sum := sha256.New()
		var s int64
		sr := io.NewSectionReader(f2, n+int64(tensor.Offset), int64(tensor.Size()))
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
	var tensors Tensors

	fmt.Println("11111111111111111111111111111111111111111")
	for _, tensor := range ggml.Tensors() {
		shape := make([]uint64, len(tensor.Shape))
		for i := range len(tensor.Shape) {
			shape[i] = tensor.Shape[len(tensor.Shape)-i-1]
		}

		fmt.Println("tensors", tensor.Name, shape, tensor.Kind, 737414+int64(tensor.Offset))
		tensors = append(tensors, &Tensor{
			Name:  tensor.Name,
			Kind:  tensor.Kind,
			Shape: shape,

			WriterTo: TensorWriter{
				Reader: io.NewSectionReader(f, 737414+int64(tensor.Offset), int64(tensor.Size())),
			},
		})
	}

	reader := &GGUFWriter{
		KV: ggml.KV(),
		// Update .Tensors
		Tensors: tensors,
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
