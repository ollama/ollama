package llm

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestMaterializeSplitModel(t *testing.T) {
	dir := t.TempDir()
	manifestDigest := strings.Repeat("a", 64)
	first := filepath.Join(dir, "first")
	second := filepath.Join(dir, "second")
	if err := os.WriteFile(first, []byte("first shard"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(second, []byte("second shard"), 0o600); err != nil {
		t.Fatal(err)
	}

	projectors := []string{first, "projector.gguf"}
	launch, err := materializeSplitModels([]string{first, second}, projectors, LlamaServerConfig{ManifestDigest: manifestDigest})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = removeSplitModelDirs(launch.dirs) })
	modelPath, splitDir := launch.modelPath, launch.dirs[0]
	if want := filepath.Join(dir, splitModelDirPrefix+manifestDigest[:12]+"-"); !strings.HasPrefix(splitDir, want) {
		t.Fatalf("split directory = %q, want prefix %q", splitDir, want)
	}
	if launch.projectors[0] != modelPath || launch.projectors[1] != projectors[1] {
		t.Fatalf("projectors = %q, want model alias and %q", launch.projectors, projectors[1])
	}
	if projectors[0] != first {
		t.Fatalf("input projectors modified: %q", projectors)
	}
	other, err := materializeSplitModels([]string{first, second}, nil, LlamaServerConfig{ManifestDigest: manifestDigest})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = removeSplitModelDirs(other.dirs) })
	if other.dirs[0] == splitDir {
		t.Fatalf("concurrent materializations share directory %q", splitDir)
	}

	if got, want := modelPath, filepath.Join(splitDir, "model-00001-of-00002.gguf"); got != want {
		t.Fatalf("model path = %q, want %q", got, want)
	}

	for i, source := range []string{first, second} {
		alias := filepath.Join(splitDir, []string{"model-00001-of-00002.gguf", "model-00002-of-00002.gguf"}[i])
		got, err := os.ReadFile(alias)
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(want) {
			t.Fatalf("alias %d contents = %q, want %q", i, got, want)
		}

		if runtime.GOOS != "windows" {
			sourceInfo, err := os.Stat(source)
			if err != nil {
				t.Fatal(err)
			}
			aliasInfo, err := os.Stat(alias)
			if err != nil {
				t.Fatal(err)
			}
			if !os.SameFile(sourceInfo, aliasInfo) {
				t.Fatalf("alias %d is not a hard link to its source", i)
			}
		}
	}
}

func TestMaterializeSplitModelsDraft(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model")
	draft := filepath.Join(dir, "draft-first")
	draftShard := filepath.Join(dir, "draft-second")
	for _, path := range []string{model, draft, draftShard} {
		if err := os.WriteFile(path, []byte(path), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	launch, err := materializeSplitModels([]string{model}, nil, LlamaServerConfig{
		ManifestDigest:       strings.Repeat("b", 64),
		DraftModelPath:       draft,
		DraftModelShardPaths: []string{draftShard},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = removeSplitModelDirs(launch.dirs) })
	if launch.modelPath != model {
		t.Fatalf("model path = %q, want %q", launch.modelPath, model)
	}
	if got, want := launch.draftModelPath, filepath.Join(launch.dirs[0], "model-00001-of-00002.gguf"); got != want {
		t.Fatalf("draft path = %q, want %q", got, want)
	}
}

func TestMaterializeSplitModelSingleFile(t *testing.T) {
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	gotPath, gotDir, err := materializeSplitModel(modelPath, nil, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != modelPath || gotDir != "" {
		t.Fatalf("materializeSplitModel() = %q, %q, want %q, empty", gotPath, gotDir, modelPath)
	}
}

func TestSplitModelDirPattern(t *testing.T) {
	digest := strings.Repeat("0123456789abcdef", 4)
	if got, want := splitModelDirPattern(digest), splitModelDirPrefix+digest[:12]+"-"; got != want {
		t.Fatalf("splitModelDirPattern() = %q, want %q", got, want)
	}
	for _, digest := range []string{"", "sha256:" + digest, strings.Repeat("g", 64), "../../models"} {
		if got := splitModelDirPattern(digest); got != splitModelDirPrefix {
			t.Errorf("splitModelDirPattern(%q) = %q, want %q", digest, got, splitModelDirPrefix)
		}
	}
}

func TestPruneSplitModelDirs(t *testing.T) {
	root := t.TempDir()
	stale := filepath.Join(root, splitModelDirPrefix+"stale")
	if err := os.Mkdir(stale, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stale, "model-00001-of-00002.gguf"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}
	active := filepath.Join(root, splitModelDirPrefix+"active")
	if err := os.Mkdir(active, 0o700); err != nil {
		t.Fatal(err)
	}
	keep := filepath.Join(root, "models")
	if err := os.Mkdir(keep, 0o700); err != nil {
		t.Fatal(err)
	}

	if err := PruneSplitModelDirs(root, time.Now().Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("split model directory still exists after cleanup: %v", err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("unrelated directory was removed: %v", err)
	}
	if _, err := os.Stat(active); err != nil {
		t.Fatalf("recent split model directory was removed: %v", err)
	}
}

func TestLlamaServerRunnerRemovesSplitDirs(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "split")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	runner := &llamaServerRunner{splitDirs: []string{dir}}
	if err := runner.removeSplitDirs(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("split directory still exists after cleanup: %v", err)
	}
	if len(runner.splitDirs) != 0 {
		t.Fatalf("runner retained %d cleaned split directories", len(runner.splitDirs))
	}
}
