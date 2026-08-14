package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// The Hugging Face registry proxy refuses sharded (multi-part) GGUF tags at
// manifest resolution rather than returning a manifest with one layer per
// shard, so the shards never reach the client and the merge implemented on the
// create path (mergeSplitGGUFLayers) is never given anything to work on.
//
// This falls back to fetching the shard files straight from Hugging Face and
// handing them to the same code path `ollama create` uses for a local
// directory of shards, which makes `ollama pull` work for those tags without
// requiring a registry-side change.
//
// See https://github.com/ollama/ollama/issues/5245 and
// https://github.com/huggingface/hub-docs/issues/2702.

const hfEndpoint = "https://huggingface.co"

// hfEndpointOverride is set by tests to point at a stub server.
var hfEndpointOverride string

// hfBase returns the Hugging Face endpoint to use, honouring HF_ENDPOINT the
// same way the huggingface_hub tooling does.
func hfBase() string {
	if hfEndpointOverride != "" {
		return hfEndpointOverride
	}
	if v := os.Getenv("HF_ENDPOINT"); v != "" {
		return strings.TrimSuffix(v, "/")
	}
	return hfEndpoint
}

var shardedRefusalRe = regexp.MustCompile(`(?i)sharded GGUF`)

// isShardedGGUFRefusal reports whether err is the registry's refusal to serve a
// sharded GGUF tag.
func isShardedGGUFRefusal(err error) bool {
	return err != nil && shardedRefusalRe.MatchString(err.Error())
}

// isHuggingFaceHost reports whether host is a Hugging Face registry host.
func isHuggingFaceHost(host string) bool {
	switch strings.ToLower(host) {
	case "hf.co", "huggingface.co":
		return true
	}
	return false
}

type hfTreeEntry struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// hfShardSet returns the ordered shard files in repo that belong to tag, along
// with their total size. Shards are matched either as a quant subdirectory
// (Q4_1/model-00001-of-00003.gguf) or as root level files whose name contains
// the tag. Ordering follows the NNNNN-of-NNNNN suffix rather than the order the
// API happens to return.
func hfShardSet(ctx context.Context, repo, tag string) ([]hfTreeEntry, int64, error) {
	u := fmt.Sprintf("%s/api/models/%s/tree/main?recursive=true", hfBase(), repo)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("listing %s: %s", repo, resp.Status)
	}

	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, 0, err
	}

	lower := strings.ToLower(tag)
	var shards []hfTreeEntry
	for _, e := range entries {
		if !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		if _, _, _, ok := splitGGUFName(e.Path); !ok {
			continue
		}
		dir, base := path.Split(e.Path)
		dir = strings.TrimSuffix(dir, "/")
		// quant directory (Q4_1/...) or a root file naming the quant
		if strings.ToLower(dir) == lower || (dir == "" && strings.Contains(strings.ToLower(base), lower)) {
			shards = append(shards, e)
		}
	}
	if len(shards) == 0 {
		return nil, 0, fmt.Errorf("no sharded GGUF files for tag %q in %s", tag, repo)
	}

	slices.SortFunc(shards, func(a, b hfTreeEntry) int {
		_, ai, _, _ := splitGGUFName(a.Path)
		_, bi, _, _ := splitGGUFName(b.Path)
		return int(ai) - int(bi)
	})

	var total int64
	for _, s := range shards {
		total += s.Size
	}
	return shards, total, nil
}

// hfToken returns a Hugging Face token from the environment, if set, so private
// repositories work the same way they do for the rest of the Hugging Face
// tooling.
func hfToken() string {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

// progressReader reports download progress for a single shard.
type progressReader struct {
	r         io.Reader
	digest    string
	total     int64
	completed int64
	fn        func(api.ProgressResponse)
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.completed += int64(n)
	p.fn(api.ProgressResponse{
		Status:    fmt.Sprintf("pulling %s", p.digest),
		Digest:    p.digest,
		Total:     p.total,
		Completed: p.completed,
	})
	return n, err
}

// downloadShard fetches a single shard into the blob store and returns its
// digest.
func downloadShard(ctx context.Context, repo string, e hfTreeEntry, fn func(api.ProgressResponse)) (string, error) {
	u := fmt.Sprintf("%s/%s/resolve/main/%s", hfBase(), repo, (&url.URL{Path: e.Path}).EscapedPath())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("downloading %s: %s", e.Path, resp.Status)
	}

	total := e.Size
	if total == 0 {
		total = resp.ContentLength
	}

	layer, err := manifest.NewLayer(&progressReader{
		r:      resp.Body,
		digest: path.Base(e.Path),
		total:  total,
		fn:     fn,
	}, "application/vnd.ollama.image.model")
	if err != nil {
		return "", err
	}
	return layer.Digest, nil
}

// pullShardedGGUF downloads the shard set for n directly from Hugging Face and
// creates the model from it, reusing the create path's merge. It is the
// in-process equivalent of downloading the shards by hand and running
// `ollama create` in that directory.
func pullShardedGGUF(ctx context.Context, n model.Name, fn func(api.ProgressResponse)) error {
	if !isHuggingFaceHost(n.Host) {
		return errors.New("sharded GGUF pull is only supported for Hugging Face models")
	}

	repo := n.Namespace + "/" + n.Model
	slog.Info("registry refused sharded GGUF tag, fetching shards directly", "repo", repo, "tag", n.Tag)

	fn(api.ProgressResponse{Status: "pulling manifest"})

	shards, total, err := hfShardSet(ctx, repo, n.Tag)
	if err != nil {
		return err
	}
	slog.Info("found sharded GGUF", "repo", repo, "tag", n.Tag, "shards", len(shards), "bytes", total)

	files := make(map[string]string, len(shards))
	for _, e := range shards {
		digest, err := downloadShard(ctx, repo, e, fn)
		if err != nil {
			return err
		}
		files[path.Base(e.Path)] = digest
	}

	// Same entry point `ollama create` uses for a directory of shards: this is
	// what groups them via splitGGUFGroupKey and merges them with
	// mergeSplitGGUFLayers into a single model layer.
	baseLayers, err := convertModelFromFiles(files, nil, false, fn)
	if err != nil {
		return err
	}

	config := &model.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
	}
	if err := createModel(api.CreateRequest{Model: n.DisplayShortest()}, n, baseLayers, config, fn); err != nil {
		return err
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}
