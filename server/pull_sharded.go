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

// shardedRefusalRe matches the registry's refusal to serve a sharded GGUF. The
// wording has changed at least once (it was reworded on 2026-08-14 to point at
// the `ollama create` workaround, and differs between a sharded tag and a
// repository containing only sharded files), so match the stable part rather
// than a whole sentence. shardedFallbackCandidate does not rely on this alone.
var shardedRefusalRe = regexp.MustCompile(`(?is)shard(ed)?[^.]{0,60}gguf|gguf[^.]{0,60}shard(ed)?`)

// isShardedGGUFRefusal reports whether err looks like the registry refusing a
// sharded GGUF tag.
func isShardedGGUFRefusal(err error) bool {
	return err != nil && shardedRefusalRe.MatchString(err.Error())
}

// isManifestClientError reports whether err came back as a 4xx from the
// registry, as opposed to a transport failure or a server error, so the
// fallback is only considered for requests the registry actively rejected.
func isManifestClientError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, os.ErrNotExist) {
		return true
	}
	return regexp.MustCompile(`^4\d\d:`).MatchString(strings.TrimSpace(err.Error()))
}

// shardedFallbackCandidate decides whether a failed manifest pull should be
// retried by fetching shards directly, and returns the shard set if so.
//
// The registry's error wording is not a stable contract, so a prose match is
// only used as a fast path: on any 4xx for a Hugging Face model the repository
// is probed for a shard set matching the tag, and the fallback runs only if one
// actually exists. That way a reworded refusal still works, and unrelated
// rejections (an invalid quantization scheme, a genuinely missing tag) fall
// through to the original error.
func shardedFallbackCandidate(ctx context.Context, n model.Name, err error) ([]hfTreeEntry, int64, bool) {
	if !isHuggingFaceHost(n.Host) {
		return nil, 0, false
	}
	if !isShardedGGUFRefusal(err) && !isManifestClientError(err) {
		return nil, 0, false
	}

	shards, total, listErr := hfShardSet(ctx, n.Namespace+"/"+n.Model, n.Tag)
	if listErr != nil {
		slog.Debug("no sharded GGUF fallback available", "model", n.DisplayShortest(), "error", listErr)
		return nil, 0, false
	}
	return shards, total, true
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
// (Q4_1/model-00001-of-00003.gguf) or as root level files with a matching tag
// suffix. Ordering follows the NNNNN-of-NNNNN suffix rather than the order the
// API happens to return.
func hfShardSet(ctx context.Context, repo, tag string) ([]hfTreeEntry, int64, error) {
	u := fmt.Sprintf("%s/api/models/%s/tree/main?recursive=true", hfBase(), repo)
	var entries []hfTreeEntry
	for u != "" {
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
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, 0, fmt.Errorf("listing %s: %s", repo, resp.Status)
		}

		var page []hfTreeEntry
		if err := json.NewDecoder(resp.Body).Decode(&page); err != nil {
			resp.Body.Close()
			return nil, 0, err
		}
		next, err := hfNextPage(resp)
		resp.Body.Close()
		if err != nil {
			return nil, 0, err
		}
		entries = append(entries, page...)
		u = next
	}

	lower := strings.ToLower(tag)
	var shards []hfTreeEntry
	for _, e := range entries {
		if !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		prefix, _, _, ok := splitGGUFName(e.Path)
		if !ok {
			continue
		}
		dir, _ := path.Split(e.Path)
		dir = strings.TrimSuffix(dir, "/")
		// quant directory (Q4_1/...) or a root file naming the quant
		if strings.ToLower(dir) == lower || (dir == "" && strings.HasSuffix(strings.ToLower(prefix), "-"+lower)) {
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

	// Validate the set before anything is downloaded. The create path already
	// rejects an unusable shard set, but only once every blob is in the store
	// (see #17946), which on this path would mean paying for the whole transfer
	// first. Everything checked here comes from the listing, so it costs
	// nothing.
	if err := validateShardSet(repo, tag, shards); err != nil {
		return nil, 0, err
	}

	var total int64
	for _, s := range shards {
		total += s.Size
	}
	return shards, total, nil
}

// validateShardSet reports whether shards form exactly one complete split GGUF
// set: a single prefix, a consistent count, and every shard present exactly
// once. shards must be sorted by index.
//
// splitGGUFName returns a zero-based index, matching the split.no metadata,
// while the filenames themselves are one-based, so indices run 0..count-1 here
// and are reported one-based to match what a user sees.
func validateShardSet(repo, tag string, shards []hfTreeEntry) error {
	wantPrefix, _, wantCount, ok := splitGGUFName(shards[0].Path)
	if !ok {
		return fmt.Errorf("shard %q does not use the llama.cpp split filename pattern", shards[0].Path)
	}

	seen := make(map[uint16]string, len(shards))
	for _, s := range shards {
		prefix, index, count, ok := splitGGUFName(s.Path)
		if !ok {
			return fmt.Errorf("shard %q does not use the llama.cpp split filename pattern", s.Path)
		}
		if prefix != wantPrefix || count != wantCount {
			return fmt.Errorf("tag %q in %s matches more than one shard set: %q and %q",
				tag, repo, shards[0].Path, s.Path)
		}
		if index >= count {
			return fmt.Errorf("shard %q is numbered %d of %d", s.Path, index+1, count)
		}
		if dup, exists := seen[index]; exists {
			return fmt.Errorf("duplicate shard %d for tag %q in %s: %q and %q",
				index+1, tag, repo, dup, s.Path)
		}
		seen[index] = s.Path
	}

	if len(seen) != int(wantCount) {
		missing := make([]string, 0, int(wantCount)-len(seen))
		for i := uint16(0); i < wantCount; i++ {
			if _, exists := seen[i]; !exists {
				missing = append(missing, fmt.Sprintf("%05d-of-%05d", i+1, wantCount))
			}
		}
		return fmt.Errorf("incomplete shard set for tag %q in %s: found %d of %d shards, missing %s",
			tag, repo, len(seen), wantCount, strings.Join(missing, ", "))
	}

	return nil
}

// hfNextPage returns the next tree API page from a standard HTTP Link header.
func hfNextPage(resp *http.Response) (string, error) {
	for _, header := range resp.Header.Values("Link") {
		for _, link := range strings.Split(header, ",") {
			parts := strings.Split(link, ";")
			if len(parts) < 2 {
				continue
			}

			for _, param := range parts[1:] {
				name, value, ok := strings.Cut(strings.TrimSpace(param), "=")
				if !ok || !strings.EqualFold(strings.TrimSpace(name), "rel") {
					continue
				}
				for _, rel := range strings.Fields(strings.Trim(value, `"`)) {
					if !strings.EqualFold(rel, "next") {
						continue
					}
					target := strings.TrimSpace(parts[0])
					target = strings.TrimPrefix(strings.TrimSuffix(target, ">"), "<")
					next, err := resp.Request.URL.Parse(target)
					if err != nil {
						return "", fmt.Errorf("parsing next page URL: %w", err)
					}
					return next.String(), nil
				}
			}
		}
	}
	return "", nil
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

// pullShardedGGUF downloads the given shard set directly from Hugging Face and
// creates the model from it, reusing the create path's merge. It is the
// in-process equivalent of downloading the shards by hand and running
// `ollama create` in that directory.
func pullShardedGGUF(ctx context.Context, n model.Name, shards []hfTreeEntry, total int64, fn func(api.ProgressResponse)) error {
	if !isHuggingFaceHost(n.Host) {
		return errors.New("sharded GGUF pull is only supported for Hugging Face models")
	}
	if len(shards) == 0 {
		return errors.New("no sharded GGUF files to pull")
	}

	repo := n.Namespace + "/" + n.Model
	slog.Info("registry refused sharded GGUF tag, fetching shards directly",
		"repo", repo, "tag", n.Tag, "shards", len(shards), "bytes", total)

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
