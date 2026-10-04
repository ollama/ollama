package tokenizer_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/transfer"
	"github.com/ollama/ollama/types/model"
)

func loadTokenizerReference(t testing.TB, modelName string) []byte {
	t.Helper()
	name := model.ParseName(modelName)
	fetch := os.Getenv("FETCH_TOKENIZERS") != "" || os.Getenv("VERIFY_TOKENIZERS") != ""
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()

	var m *manifest.Manifest
	if fetch {
		// Resolve the tag on every fetch run so cached data cannot hide updates.
		url := name.BaseURL().JoinPath("v2", name.DisplayNamespaceModel(), "manifests", name.Tag)
		m = fetchTokenizerManifest(t, ctx, url.String(), "")
		if m.MediaType == manifest.MediaTypeManifestList {
			child, err := manifest.SelectManifestReferenceForRunner(m.Manifests, manifest.RunnerMLX)
			if err != nil {
				t.Fatal(err)
			}
			digest := child.BlobDigest()
			if digest == "" {
				t.Fatal("MLX manifest reference has no digest")
			}
			url = name.BaseURL().JoinPath("v2", name.DisplayNamespaceModel(), "blobs", digest)
			m = fetchTokenizerManifest(t, ctx, url.String(), digest)
		}
	} else {
		var err error
		m, err = manifest.ParseNamedManifestForRunner(name, manifest.RunnerMLX)
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, manifest.ErrNoCompatibleManifest) {
			t.Skipf("%s is not installed; set FETCH_TOKENIZERS=1 to download tokenizer data only", modelName)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	layer, ok := m.ConfigLayer("tokenizer.json")
	if !ok {
		t.Fatalf("%s manifest has no tokenizer.json", modelName)
	}
	path, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) && fetch {
		// These blobs have no model manifest: keep them out of model-store pruning.
		cacheDir := filepath.Join("..", "..", ".cache", "tokenizers")
		if err := transfer.Download(ctx, transfer.DownloadOptions{
			BaseURL:    name.BaseURL().String(),
			Repository: name.DisplayNamespaceModel(),
			DestDir:    cacheDir,
			Blobs:      []transfer.Blob{{Digest: layer.Digest, Size: layer.Size}},
		}); err != nil {
			t.Fatalf("fetch %s tokenizer: %v", modelName, err)
		}
		path = filepath.Join(cacheDir, strings.ReplaceAll(layer.Digest, ":", "-"))
		data, err = os.ReadFile(path)
	}
	if errors.Is(err, os.ErrNotExist) && !fetch {
		t.Skipf("%s tokenizer is missing; set FETCH_TOKENIZERS=1 to download tokenizer data only", modelName)
	}
	if err != nil {
		t.Fatalf("read %s tokenizer: %v", modelName, err)
	}
	if int64(len(data)) != layer.Size {
		t.Fatalf("%s tokenizer size = %d, want %d", modelName, len(data), layer.Size)
	}
	if got := fmt.Sprintf("sha256:%x", sha256.Sum256(data)); got != layer.Digest {
		t.Fatalf("%s tokenizer digest = %s, want %s", modelName, got, layer.Digest)
	}
	return data
}

func fetchTokenizerManifest(t testing.TB, ctx context.Context, url, digest string) *manifest.Manifest {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", manifest.MediaTypeManifestList+", "+manifest.MediaTypeManifest)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("fetch manifest %s: %s", url, resp.Status)
	}
	const maxManifestSize = 16 << 20
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxManifestSize+1))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) > maxManifestSize {
		t.Fatalf("manifest %s exceeds %d bytes", url, maxManifestSize)
	}
	if digest != "" {
		if got := fmt.Sprintf("sha256:%x", sha256.Sum256(data)); got != digest {
			t.Fatalf("manifest %s digest = %s, want %s", url, got, digest)
		}
	}
	var m manifest.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("decode manifest %s: %v", url, err)
	}
	return &m
}

func TestTokenizerReferenceFetch(t *testing.T) {
	root := t.TempDir()
	t.Setenv("OLLAMA_MODELS", filepath.Join(root, "models"))
	t.Setenv("FETCH_TOKENIZERS", "1")
	t.Setenv("VERIFY_TOKENIZERS", "")
	work := filepath.Join(root, "work", "tokenizer")
	if err := os.MkdirAll(work, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(work)

	tokenizers := [][]byte{[]byte(`{"version":"first"}`), []byte(`{"version":"updated"}`)}
	digests := make([]string, len(tokenizers))
	manifests := make([][]byte, len(tokenizers))
	blobs := make(map[string][]byte)
	for i, data := range tokenizers {
		digests[i] = fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		blobs[digests[i]] = data
		m := manifest.Manifest{
			SchemaVersion: 2,
			MediaType:     manifest.MediaTypeManifest,
			Runner:        manifest.RunnerMLX,
			Layers: []manifest.Layer{
				{MediaType: "application/vnd.ollama.image.json", Name: "tokenizer.json", Digest: digests[i], Size: int64(len(data))},
				{MediaType: "application/vnd.ollama.image.tensor", Digest: "sha256:" + strings.Repeat("1", 64), Size: 1 << 30},
			},
		}
		var err error
		manifests[i], err = json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
	}
	childDigest := fmt.Sprintf("sha256:%x", sha256.Sum256(manifests[1]))
	blobs[childDigest] = manifests[1]
	child, err := manifest.NewManifestReference(childDigest, manifest.RunnerMLX, manifest.FormatSafetensors)
	if err != nil {
		t.Fatal(err)
	}
	manifests[1], err = json.Marshal(manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifestList,
		Manifests:     []manifest.Manifest{{Runner: manifest.RunnerLlamaCPP}, child},
	})
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	current := manifests[0]
	requests := make(map[string]int)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests[r.URL.Path]++
		data := blobs[strings.TrimPrefix(r.URL.Path, "/v2/library/reference/blobs/")]
		if r.URL.Path == "/v2/library/reference/manifests/latest" {
			data = current
		}
		mu.Unlock()
		if data == nil {
			t.Errorf("unexpected registry request: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	t.Cleanup(ts.Close)

	var firstDownloads int
	for i, version := range []int{0, 0, 1} {
		mu.Lock()
		current = manifests[version]
		mu.Unlock()
		if got := loadTokenizerReference(t, ts.URL+"/library/reference:latest"); !bytes.Equal(got, tokenizers[version]) {
			t.Fatalf("run %d tokenizer = %s, want %s", i, got, tokenizers[version])
		}
		mu.Lock()
		downloads := requests["/v2/library/reference/blobs/"+digests[0]]
		mu.Unlock()
		if i == 0 {
			firstDownloads = downloads
		} else if downloads != firstDownloads {
			t.Fatalf("cached tokenizer was downloaded again: requests = %d, want %d", downloads, firstDownloads)
		}
	}
	mu.Lock()
	manifestRequests := requests["/v2/library/reference/manifests/latest"]
	mu.Unlock()
	if manifestRequests != 3 {
		t.Fatalf("manifest requests = %d, want 3", manifestRequests)
	}
}
