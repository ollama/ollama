package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
	"github.com/ollama/ollama/x/imagegen/transfer"
)

var (
	errCapabilities         = errors.New("does not support")
	errCapabilityCompletion = errors.New("completion")
	errCapabilityTools      = errors.New("tools")
	errCapabilityInsert     = errors.New("insert")
	errCapabilityVision     = errors.New("vision")
	errCapabilityEmbedding  = errors.New("embedding")
	errCapabilityThinking   = errors.New("thinking")
	errCapabilityImage      = errors.New("image generation")
	errInsecureProtocol     = errors.New("insecure protocol http")
)

type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

type Model struct {
	Name           string `json:"name"`
	Config         model.ConfigV2
	ShortName      string
	ModelPath      string
	ParentModel    string
	AdapterPaths   []string
	ProjectorPaths []string
	System         string
	License        []string
	Digest         string
	Options        map[string]any
	Messages       []api.Message

	Template *template.Template
}

// Capabilities returns the capabilities that the model supports
func (m *Model) Capabilities() []model.Capability {
	capabilities := []model.Capability{}

	// Check for image generation model via config capabilities
	if slices.Contains(m.Config.Capabilities, "image") {
		return []model.Capability{model.CapabilityImage}
	}

	// Check for completion capability
	if m.ModelPath != "" {
		f, err := gguf.Open(m.ModelPath)
		if err == nil {
			defer f.Close()

			if f.KeyValue("pooling_type").Valid() {
				capabilities = append(capabilities, model.CapabilityEmbedding)
			} else {
				// If no embedding is specified, we assume the model supports completion
				capabilities = append(capabilities, model.CapabilityCompletion)
			}
			if f.KeyValue("vision.block_count").Valid() {
				capabilities = append(capabilities, model.CapabilityVision)
			}
		} else {
			slog.Error("couldn't open model file", "error", err)
		}
	} else if len(m.Config.Capabilities) > 0 {
		for _, c := range m.Config.Capabilities {
			capabilities = append(capabilities, model.Capability(c))
		}
	} else {
		slog.Warn("unknown capabilities for model", "model", m.Name)
	}

	if m.Template == nil {
		return capabilities
	}

	builtinParser := parsers.ParserForName(m.Config.Parser)
	// Check for tools capability
	v, err := m.Template.Vars()
	if err != nil {
		slog.Warn("model template contains errors", "error", err)
	}
	if slices.Contains(v, "tools") || (builtinParser != nil && builtinParser.HasToolSupport()) {
		capabilities = append(capabilities, model.CapabilityTools)
	}

	// Check for insert capability
	if slices.Contains(v, "suffix") {
		capabilities = append(capabilities, model.CapabilityInsert)
	}

	// Check for vision capability in projector-based models
	if len(m.ProjectorPaths) > 0 {
		capabilities = append(capabilities, model.CapabilityVision)
	}

	// Skip the thinking check if it's already set
	if slices.Contains(capabilities, "thinking") {
		return capabilities
	}

	// Check for thinking capability
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	hasTags := openingTag != "" && closingTag != ""
	isGptoss := slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily)
	if hasTags || isGptoss || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

// CheckCapabilities checks if the model has the specified capabilities returning an error describing
// any missing or unknown capabilities
func (m *Model) CheckCapabilities(want ...model.Capability) error {
	available := m.Capabilities()
	var errs []error

	// Map capabilities to their corresponding error
	capToErr := map[model.Capability]error{
		model.CapabilityCompletion: errCapabilityCompletion,
		model.CapabilityTools:      errCapabilityTools,
		model.CapabilityInsert:     errCapabilityInsert,
		model.CapabilityVision:     errCapabilityVision,
		model.CapabilityEmbedding:  errCapabilityEmbedding,
		model.CapabilityThinking:   errCapabilityThinking,
		model.CapabilityImage:      errCapabilityImage,
	}

	for _, cap := range want {
		err, ok := capToErr[cap]
		if !ok {
			slog.Error("unknown capability", "capability", cap)
			return fmt.Errorf("unknown capability: %s", cap)
		}

		if !slices.Contains(available, cap) {
			errs = append(errs, err)
		}
	}

	var err error
	if len(errs) > 0 {
		err = fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))
	}

	if slices.Contains(errs, errCapabilityThinking) {
		if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
			// append a message to the existing error
			return fmt.Errorf("%w. Pull the model again to get the latest version with full thinking support", err)
		}
	}

	return err
}

func (m *Model) String() string {
	var modelfile parser.Modelfile

	modelfile.Commands = append(modelfile.Commands, parser.Command{
		Name: "model",
		Args: m.ModelPath,
	})

	for _, adapter := range m.AdapterPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "adapter",
			Args: adapter,
		})
	}

	for _, projector := range m.ProjectorPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "model",
			Args: projector,
		})
	}

	if m.Template != nil {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "template",
			Args: m.Template.String(),
		})
	}

	if m.System != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "system",
			Args: m.System,
		})
	}

	if m.Config.Renderer != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "renderer",
			Args: m.Config.Renderer,
		})
	}

	if m.Config.Parser != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "parser",
			Args: m.Config.Parser,
		})
	}

	for k, v := range m.Options {
		switch v := v.(type) {
		case []any:
			for _, s := range v {
				modelfile.Commands = append(modelfile.Commands, parser.Command{
					Name: k,
					Args: fmt.Sprintf("%v", s),
				})
			}
		default:
			modelfile.Commands = append(modelfile.Commands, parser.Command{
				Name: k,
				Args: fmt.Sprintf("%v", v),
			})
		}
	}

	for _, license := range m.License {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "license",
			Args: license,
		})
	}

	for _, msg := range m.Messages {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "message",
			Args: fmt.Sprintf("%s: %s", msg.Role, msg.Content),
		})
	}

	return modelfile.String()
}

func GetManifest(mp ModelPath) (*Manifest, string, error) {
	fp, err := mp.GetManifestPath()
	if err != nil {
		return nil, "", err
	}

	f, err := os.Open(fp)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()

	sha256sum := sha256.New()

	var manifest Manifest
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&manifest); err != nil {
		return nil, "", err
	}

	return &manifest, hex.EncodeToString(sha256sum.Sum(nil)), nil
}

func GetModel(name string) (*Model, error) {
	mp := ParseModelPath(name)
	manifest, digest, err := GetManifest(mp)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name:      mp.GetFullTagname(),
		ShortName: mp.GetShortTagname(),
		Digest:    digest,
		Template:  template.DefaultTemplate,
	}

	if manifest.Config.Digest != "" {
		filename, err := GetBlobsPath(manifest.Config.Digest)
		if err != nil {
			return nil, err
		}

		configFile, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer configFile.Close()

		if err := json.NewDecoder(configFile).Decode(&model.Config); err != nil {
			return nil, err
		}
	}

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
			model.ParentModel = layer.From
		case "application/vnd.ollama.image.embed":
			// Deprecated in versions  > 0.1.2
			// TODO: remove this warning in a future version
			slog.Info("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			model.AdapterPaths = append(model.AdapterPaths, filename)
		case "application/vnd.ollama.image.projector":
			model.ProjectorPaths = append(model.ProjectorPaths, filename)
		case "application/vnd.ollama.image.prompt",
			"application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template, err = template.Parse(string(bts))
			if err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.System = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// parse model options parameters into a map so that we can see which fields have been specified explicitly
			if err = json.NewDecoder(params).Decode(&model.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.messages":
			msgs, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer msgs.Close()

			if err = json.NewDecoder(msgs).Decode(&model.Messages); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			model.License = append(model.License, string(bts))
		}
	}

	return model, nil
}

func CopyModel(src, dst model.Name) error {
	if !dst.IsFullyQualified() {
		return model.Unqualified(dst)
	}
	if !src.IsFullyQualified() {
		return model.Unqualified(src)
	}

	if src.Filepath() == dst.Filepath() {
		return nil
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	dstpath := filepath.Join(manifests, dst.Filepath())
	if err := os.MkdirAll(filepath.Dir(dstpath), 0o755); err != nil {
		return err
	}

	srcpath := filepath.Join(manifests, src.Filepath())
	srcfile, err := os.Open(srcpath)
	if err != nil {
		return err
	}
	defer srcfile.Close()

	dstfile, err := os.Create(dstpath)
	if err != nil {
		return err
	}
	defer dstfile.Close()

	_, err = io.Copy(dstfile, srcfile)
	return err
}

func deleteUnusedLayers(deleteMap map[string]struct{}) error {
	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	manifests, err := Manifests(true)
	if err != nil {
		return err
	}

	for _, manifest := range manifests {
		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := GetBlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if err := os.Remove(fp); err != nil {
			slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
			continue
		}
	}

	return nil
}

func PruneLayers() error {
	deleteMap := make(map[string]struct{})
	p, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't read dir '%s': %v", p, err))
		return err
	}

	for _, blob := range blobs {
		name := blob.Name()
		name = strings.ReplaceAll(name, "-", ":")

		_, err := GetBlobsPath(name)
		if err != nil {
			if errors.Is(err, ErrInvalidDigestFormat) {
				// remove invalid blobs (e.g. partial downloads)
				if err := os.Remove(filepath.Join(p, blob.Name())); err != nil {
					slog.Error("couldn't remove blob", "blob", blob.Name(), "error", err)
				}
			}

			continue
		}

		deleteMap[name] = struct{}{}
	}

	slog.Info(fmt.Sprintf("total blobs: %d", len(deleteMap)))

	if err := deleteUnusedLayers(deleteMap); err != nil {
		slog.Error(fmt.Sprintf("couldn't remove unused layers: %v", err))
		return nil
	}

	slog.Info(fmt.Sprintf("total unused blobs removed: %d", len(deleteMap)))

	return nil
}

func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}

func PushModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	manifest, _, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []Layer
	layers = append(layers, manifest.Layers...)
	if manifest.Config.Digest != "" {
		layers = append(layers, manifest.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		// Read raw manifest JSON to preserve tensor metadata fields
		manifestPath, err := mp.GetManifestPath()
		if err != nil {
			return err
		}
		manifestJSON, err := os.ReadFile(manifestPath)
		if err != nil {
			return err
		}
		if err := pushWithTransfer(ctx, mp, layers, manifestJSON, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	for _, layer := range layers {
		if err := uploadBlob(ctx, mp, layer, regOpts, fn); err != nil {
			slog.Info(fmt.Sprintf("error uploading blob: %v", err))
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func PullModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})
	manifest, _, err := GetManifest(mp)
	if errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		slog.Warn("pulling model with bad existing manifest", "name", name, "error", err)
	} else {
		for _, l := range manifest.Layers {
			deleteMap[l.Digest] = struct{}{}
		}
		if manifest.Config.Digest != "" {
			deleteMap[manifest.Config.Digest] = struct{}{}
		}
	}

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err = pullModelManifest(ctx, mp, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []Layer
	layers = append(layers, manifest.Layers...)
	if manifest.Config.Digest != "" {
		layers = append(layers, manifest.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		if err := pullWithTransfer(ctx, mp, layers, manifest, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	skipVerify := make(map[string]bool)
	for _, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			mp:      mp,
			digest:  layer.Digest,
			regOpts: regOpts,
			fn:      fn,
		})
		if err != nil {
			return err
		}
		skipVerify[layer.Digest] = cacheHit
		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if skipVerify[layer.Digest] {
			continue
		}
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				fp, err := GetBlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					slog.Info(fmt.Sprintf("couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
			}
			return err
		}
	}

	for _, layer := range layers {
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, manifest.Config.Digest)

	fn(api.ProgressResponse{Status: "writing manifest"})

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't write to %s", fp))
		return err
	}

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

// hasTensorLayers checks if any layer has tensor media type.
func hasTensorLayers(layers []Layer) bool {
	for _, layer := range layers {
		if layer.MediaType == MediaTypeImageTensor {
			return true
		}
	}
	return false
}

// pullWithTransfer uses the simplified x/transfer package for downloading blobs.
func pullWithTransfer(ctx context.Context, mp ModelPath, layers []Layer, manifest *Manifest, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
		}
	}

	destDir, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	base := mp.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pulling model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	if err := transfer.Download(ctx, transfer.DownloadOptions{
		Blobs:      blobs,
		BaseURL:    baseURL,
		DestDir:    destDir,
		Repository: mp.GetNamespaceRepository(),
		Progress:   progress,
		Token:      regOpts.Token,
		GetToken:   getToken,
		Logger:     slog.Default(),
	}); err != nil {
		return err
	}

	// Write manifest
	fn(api.ProgressResponse{Status: "writing manifest"})
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	return os.WriteFile(fp, manifestJSON, 0o644)
}

// pushWithTransfer uses the simplified x/transfer package for uploading blobs and manifest.
func pushWithTransfer(ctx context.Context, mp ModelPath, layers []Layer, manifestJSON []byte, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
			From:   layer.From,
		}
	}

	srcDir, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	base := mp.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pushing model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	return transfer.Upload(ctx, transfer.UploadOptions{
		Blobs:       blobs,
		BaseURL:     baseURL,
		SrcDir:      srcDir,
		Progress:    progress,
		Token:       regOpts.Token,
		GetToken:    getToken,
		Logger:      slog.Default(),
		Manifest:    manifestJSON,
		ManifestRef: mp.Tag,
		Repository:  mp.GetNamespaceRepository(),
	})
}

func pullModelManifest(ctx context.Context, mp ModelPath, regOpts *registryOptions) (*Manifest, error) {
	requestURL := mp.BaseURL().JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return &m, err
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

var errUnauthorized = errors.New("unauthorized: access denied")

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *registryOptions) (*http.Response, error) {
	for range 2 {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.Info(fmt.Sprintf("request failed: %v", err))
			}

			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			resp.Body.Close()

			// Handle authentication error with one retry
			challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
			token, err := getAuthorizationToken(ctx, challenge, requestURL.Host)
			if err != nil {
				return nil, err
			}
			regOpts.Token = token
			if body != nil {
				_, err = body.Seek(0, io.SeekStart)
				if err != nil {
					return nil, err
				}
			}
		case resp.StatusCode == http.StatusNotFound:
			resp.Body.Close()
			return nil, os.ErrNotExist
		case resp.StatusCode >= http.StatusBadRequest:
			defer resp.Body.Close()
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
			}
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
		default:
			return resp, nil
		}
	}

	return nil, errUnauthorized
}

// testMakeRequestDialContext specifies the dial function for the http client in
// makeRequest. It can be used to resolve hosts in model names to local
// addresses for testing. For example, the model name ("example.com/my/model")
// can be directed to push/pull from "127.0.0.1:1234".
//
// This is not safe to set across goroutines. It should be set in
// the main test goroutine, and not by tests marked to run in parallel with
// t.Parallel().
//
// It should be cleared after use, otherwise it will affect other tests.
//
// Ideally we would have some set this up the stack, but the code is not
// structured in a way that makes this easy, so this will have to do for now.
var testMakeRequestDialContext func(ctx context.Context, network, addr string) (net.Conn, error)

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *registryOptions) (*http.Response, error) {
	if requestURL.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "http"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		} else if regOpts.Username != "" && regOpts.Password != "" {
			req.SetBasicAuth(regOpts.Username, regOpts.Password)
		}
	}

	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	c := &http.Client{
		CheckRedirect: regOpts.CheckRedirect,
	}
	if testMakeRequestDialContext != nil {
		tr := http.DefaultTransport.(*http.Transport).Clone()
		tr.DialContext = testMakeRequestDialContext
		c.Transport = tr
	}
	return c.Do(req)
}

func getValue(header, key string) string {
	startIdx := strings.Index(header, key+"=")
	if startIdx == -1 {
		return ""
	}

	// Move the index to the starting quote after the key.
	startIdx += len(key) + 2
	endIdx := startIdx

	for endIdx < len(header) {
		if header[endIdx] == '"' {
			if endIdx+1 < len(header) && header[endIdx+1] != ',' { // If the next character isn't a comma, continue
				endIdx++
				continue
			}
			break
		}
		endIdx++
	}
	return header[startIdx:endIdx]
}

func parseRegistryChallenge(authStr string) registryChallenge {
	authStr = strings.TrimPrefix(authStr, "Bearer ")

	return registryChallenge{
		Realm:   getValue(authStr, "realm"),
		Service: getValue(authStr, "service"),
		Scope:   getValue(authStr, "scope"),
	}
}

var errDigestMismatch = errors.New("digest mismatch, file must be downloaded again")

func verifyBlob(digest string) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		return fmt.Errorf("%w: want %s, got %s", errDigestMismatch, digest, fileDigest)
	}

	return nil
}
