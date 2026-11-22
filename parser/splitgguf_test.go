package parser

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

// writeSplitGGUFShard writes a minimal but valid GGUF shard with the given
// split metadata to dir/name and returns its path.
func writeSplitGGUFShard(t *testing.T, dir, name string, splitNo, splitCount uint32) string {
	t.Helper()

	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	kv := gguftest.KV{
		"general.architecture": "llama",
		"general.file_type":    uint32(gguf.FileTypeF32),
		"split.no":             splitNo,
		"split.count":          splitCount,
		"split.tensors.count":  int32(splitCount),
	}
	tensors := []*gguftest.Tensor{{
		Name:     fmt.Sprintf("blk.%d.attn_q.weight", splitNo),
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{1, 1},
		WriterTo: strings.NewReader(string(make([]byte, 4))),
	}}
	if err := gguftest.Write(f, kv, tensors); err != nil {
		t.Fatal(err)
	}
	return path
}

// writePlainGGUF writes a single-file (non-split) GGUF to dir/name.
func writePlainGGUF(t *testing.T, dir, name string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	kv := gguftest.KV{
		"general.architecture": "llama",
		"general.file_type":    uint32(gguf.FileTypeF32),
	}
	tensors := []*gguftest.Tensor{{
		Name:     "blk.0.attn_q.weight",
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{1, 1},
		WriterTo: strings.NewReader(string(make([]byte, 4))),
	}}
	if err := gguftest.Write(f, kv, tensors); err != nil {
		t.Fatal(err)
	}
	return path
}

func fileDigest(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", digest)
}

func createRequestFiles(t *testing.T, input, dir string) *api.CreateRequest {
	t.Helper()

	modelfile, err := ParseFile(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	req, err := modelfile.CreateRequest(dir)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

func TestCreateRequestExpandsSingleSplitGGUFShard(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)

	// Referencing only the first shard must gather every shard.
	req := createRequestFiles(t, "FROM ./model-00001-of-00002.gguf\n", dir)
	want := map[string]string{first: fileDigest(t, first), second: fileDigest(t, second)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
	if req.From != "" {
		t.Errorf("From = %q, want empty", req.From)
	}
}

func TestCreateRequestExpandsLastSplitGGUFShard(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00003.gguf", 0, 3)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00003.gguf", 1, 3)
	third := writeSplitGGUFShard(t, dir, "model-00003-of-00003.gguf", 2, 3)

	// Referencing any shard (here the last) must gather every shard.
	req := createRequestFiles(t, "FROM ./model-00003-of-00003.gguf\n", dir)
	want := map[string]string{
		first:  fileDigest(t, first),
		second: fileDigest(t, second),
		third:  fileDigest(t, third),
	}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestExpandsSplitGGUFShardAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)

	req := createRequestFiles(t, fmt.Sprintf("FROM %s\n", first), t.TempDir())
	want := map[string]string{first: fileDigest(t, first), second: fileDigest(t, second)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestSplitGGUFShardsAcrossFromLines(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)

	// Separate FROM instructions listing each shard keep working.
	req := createRequestFiles(t, fmt.Sprintf("FROM %s\nFROM %s\n", first, second), t.TempDir())
	want := map[string]string{first: fileDigest(t, first), second: fileDigest(t, second)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestSplitGGUFGlobStillWorks(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)
	writePlainGGUF(t, dir, "projector.gguf")

	req := createRequestFiles(t, "FROM ./model-*.gguf\n", dir)
	want := map[string]string{first: fileDigest(t, first), second: fileDigest(t, second)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestMissingSplitGGUFShardErrors(t *testing.T) {
	dir := t.TempDir()
	writeSplitGGUFShard(t, dir, "model-00001-of-00003.gguf", 0, 3)
	writeSplitGGUFShard(t, dir, "model-00003-of-00003.gguf", 2, 3)

	modelfile, err := ParseFile(strings.NewReader("FROM ./model-00001-of-00003.gguf\n"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = modelfile.CreateRequest(dir)
	if err == nil {
		t.Fatal("CreateRequest() succeeded, want missing shard error")
	}
	if !strings.Contains(err.Error(), "missing shard(s)") || !strings.Contains(err.Error(), "model-00002-of-00003.gguf") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCreateRequestShardNamedSingleFileNotExpanded(t *testing.T) {
	dir := t.TempDir()
	// A single-file GGUF whose name matches the split pattern but whose
	// metadata has no split.count must not be expanded or rejected.
	single := writePlainGGUF(t, dir, "model-00001-of-00002.gguf")

	req := createRequestFiles(t, "FROM ./model-00001-of-00002.gguf\n", dir)
	want := map[string]string{single: fileDigest(t, single)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestPlainGGUFUnchanged(t *testing.T) {
	dir := t.TempDir()
	single := writePlainGGUF(t, dir, "model.gguf")

	req := createRequestFiles(t, "FROM ./model.gguf\n", dir)
	want := map[string]string{single: fileDigest(t, single)}
	if diff := cmp.Diff(want, req.Files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestShardPatternRemoteNameUnchanged(t *testing.T) {
	// A FROM argument that matches the shard pattern but does not exist
	// locally is still treated as a remote model reference.
	req := createRequestFiles(t, "FROM model-00001-of-00002.gguf\n", t.TempDir())
	if req.From != "model-00001-of-00002.gguf" {
		t.Errorf("From = %q, want %q", req.From, "model-00001-of-00002.gguf")
	}
	if len(req.Files) != 0 {
		t.Errorf("Files = %v, want empty", req.Files)
	}
}

func TestCreateRequestExpandsDraftSplitGGUFShard(t *testing.T) {
	dir := t.TempDir()
	base := writePlainGGUF(t, dir, "base.gguf")
	first := writeSplitGGUFShard(t, dir, "draft-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "draft-00002-of-00002.gguf", 1, 2)

	req := createRequestFiles(t, fmt.Sprintf("FROM %s\nDRAFT ./draft-00001-of-00002.gguf\n", base), dir)
	wantDraft := map[string]string{first: fileDigest(t, first), second: fileDigest(t, second)}
	if diff := cmp.Diff(wantDraft, req.DraftFiles); diff != "" {
		t.Errorf("draft files mismatch (-want +got):\n%s", diff)
	}
}

func TestExpandSplitGGUFShards(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)
	plain := writePlainGGUF(t, dir, "other.gguf")

	cases := []struct {
		name  string
		paths []string
		want  []string
		err   string
	}{
		{
			name:  "single shard expands to all shards",
			paths: []string{first},
			want:  []string{first, second},
		},
		{
			name:  "second shard expands to all shards",
			paths: []string{second},
			want:  []string{first, second},
		},
		{
			name:  "duplicates are deduplicated",
			paths: []string{first, second, first},
			want:  []string{first, second},
		},
		{
			name:  "plain gguf is unchanged",
			paths: []string{plain},
			want:  []string{plain},
		},
		{
			name:  "missing path is left for remote handling",
			paths: []string{filepath.Join(dir, "absent-00001-of-00002.gguf")},
			want:  []string{filepath.Join(dir, "absent-00001-of-00002.gguf")},
		},
		{
			name:  "mixed plain and split shards",
			paths: []string{plain, first},
			want:  []string{first, second, plain},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := expandSplitGGUFShards(c.paths)
			if c.err != "" {
				if err == nil || !strings.Contains(err.Error(), c.err) {
					t.Fatalf("expandSplitGGUFShards() error = %v, want containing %q", err, c.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, c.want) {
				t.Errorf("expandSplitGGUFShards() = %v, want %v", got, c.want)
			}
		})
	}
}

func TestExpandSplitGGUFShardsMissingShard(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)

	_, err := expandSplitGGUFShards([]string{first})
	if err == nil {
		t.Fatal("expandSplitGGUFShards() succeeded, want missing shard error")
	}
	if !strings.Contains(err.Error(), "model-00002-of-00002.gguf") {
		t.Fatalf("error %q does not name the missing shard", err)
	}
}

func TestExpandSplitGGUFShardsSortedKeys(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUFShard(t, dir, "model-00001-of-00002.gguf", 0, 2)
	second := writeSplitGGUFShard(t, dir, "model-00002-of-00002.gguf", 1, 2)

	req := createRequestFiles(t, "FROM ./model-00002-of-00002.gguf\n", dir)
	got := sortedKeys(req.Files)
	want := []string{first, second}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("file keys = %v, want %v", got, want)
	}
}
