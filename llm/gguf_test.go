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
		"glm2.gguf",
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

			ggml, m, err := decodeGGML(t, f)
			if err != nil {
				t.Fatal(err)
			}

			temp, err := os.CreateTemp("testdata", "2"+tt)
			if err != nil {
				t.Fatal(err)
			}
			defer temp.Close()

			n, ggml2, err := rewriteGGML(t, ggml, temp)

			if n != m {
				t.Fatalf("n: %d, m: %d", n, m)
			}

			if err != nil {
				t.Fatal(err)
			}

			if diff, diff2, ok := compareGGML(n, ggml2, ggml, temp, f); !ok {
				if cmp.Diff(diff, diff2) != "" {
					t.Fatalf("\n%s,\n%s\ndiff: %s", diff["token_embd.weight"], diff2["token_embd.weight"], cmp.Diff(diff, diff2))
				}
			}

			/* // Reset the file offset to the beginning
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				t.Fatal(err)
			}
			if _, err := temp.Seek(0, io.SeekStart); err != nil {
				t.Fatal(err)
			}

			content1, err := io.ReadAll(f)
			if err != nil {
				t.Fatalf("failed to read file1: %v", err)
			}

			content2, err := io.ReadAll(temp)
			if err != nil {
				t.Fatalf("failed to read file1: %v", err)
			}

			if byteCmp := cmp.Diff(content1, content2); byteCmp != "" {
				t.Fatalf("content diff: %s", byteCmp)
			} */
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
		if _, err := io.Copy(sha256sum, sr); err != nil {
			fmt.Println(err)
		}

		diff[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
	}

	for _, tensor := range t2 {
		sha256sum := sha256.New()
		sr := io.NewSectionReader(f2, n+int64(tensor.Offset), int64(tensor.Size()))
		if _, err := io.Copy(sha256sum, sr); err != nil {
			fmt.Println(err)
		}

		diff2[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
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

func rewriteGGML(t *testing.T, ggml *GGML, temp *os.File) (int64, *GGML, error) {
	var tensors Tensors

	for _, tensor := range ggml.Tensors() {
		shape := make([]uint64, len(tensor.Shape))
		for i := range len(tensor.Shape) {
			shape[i] = tensor.Shape[len(tensor.Shape)-i-1]
		}

		tensors = append(tensors, &Tensor{
			Name:  tensor.Name,
			Kind:  tensor.Kind,
			Shape: shape,

			WriterTo: TensorWriter{
				Reader: io.NewSectionReader(temp, int64(tensor.Offset), int64(tensor.Size())),
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

	fmt.Println(n)
	temp.Seek(0, io.SeekStart)
	file, err := os.Open(temp.Name())
	if err != nil {
		t.Fatal(err)
	}
	ggml2, n, err := DecodeGGML(file, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}

	return n, ggml2, nil
}
