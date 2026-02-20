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
	"github.com/ollama/ollama/manifest"
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
	Name            string `json:"name"`
	Config          model.ConfigV2
	ShortName       string
	ModelPath       string
	ExtraModelPaths []string
	ParentModel     string
	AdapterPaths    []string
	ProjectorPaths  []string
	System          string
	License         []string
	Digest          string
	Options         map[string]any
	Messages        []api.Message

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

	for _, extraModels := range m.ExtraModelPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "model",
			Args: extraModels,
		})
	}

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

func GetManifest(n model.Name) (*manifest.Manifest, string, error) {
	fp := n.Filepath()

	f, err := os.Open(fp)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()

	sha256sum := sha256.New()

	var manifestFile manifest.Manifest
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&manifestFile); err != nil {
		return nil, "", err
	}

	return &manifestFile, hex.EncodeToString(sha256sum.Sum(nil)), nil
}

func GetModel(name string) (*Model, error) {
	n := model.ParseName(name)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Name:      n.String(),
		ShortName: n.DisplayShortest(),
		Digest:    mf.Digest(),
		Template:  template.DefaultTemplate,
	}

	if mf.Config.Digest != "" {
		filename, err := manifest.BlobsPath(mf.Config.Digest)
		if err != nil {
			return nil, err
		}

		configFile, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer configFile.Close()

		if err := json.NewDecoder(configFile).Decode(&m.Config); err != nil {
			return nil, err
		}
	}

	readMainModelFlag := false

	for _, layer := range mf.Layers {
		filename, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			if !readMainModelFlag {
				m.ModelPath = filename
				m.ParentModel = layer.From
				readMainModelFlag = true
			} else {
				m.ExtraModelPaths = append(m.ExtraModelPaths, filename)
			}
		case "application/vnd.ollama.image.embed":
			// Deprecated in versions  > 0.1.2
			// TODO: remove this warning in a future version
			slog.Info("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			m.AdapterPaths = append(m.AdapterPaths, filename)
		case "application/vnd.ollama.image.projector":
			m.ProjectorPaths = append(m.ProjectorPaths, filename)
		case "application/vnd.ollama.image.prompt",
			"application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.Template, err = template.Parse(string(bts))
			if err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.System = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// parse model options parameters into a map so that we can see which fields have been specified explicitly
			if err = json.NewDecoder(params).Decode(&m.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.messages":
			msgs, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer msgs.Close()

			if err = json.NewDecoder(msgs).Decode(&m.Messages); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			m.License = append(m.License, string(bts))
		}
	}

	return m, nil
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

	manifests, err := manifest.Path()
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
	manifests, err := manifest.Manifests(true)
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
		fp, err := manifest.BlobsPath(k)
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
	p, err := manifest.BlobsPath("")
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

		_, err := manifest.BlobsPath(name)
		if err != nil {
			if errors.Is(err, manifest.ErrInvalidDigestFormat) {
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
	n := model.ParseName(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		// Read raw manifest JSON to preserve tensor metadata fields
		manifestPath, err := manifest.PathForName(n)
		if err != nil {
			return err
		}
		manifestJSON, err := os.ReadFile(manifestPath)
		if err != nil {
			return err
		}
		if err := pushWithTransfer(ctx, n, layers, manifestJSON, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	for _, layer := range layers {
		if err := uploadBlob(ctx, n, layer, regOpts, fn); err != nil {
			slog.Info(fmt.Sprintf("error uploading blob: %v", err))
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := n.BaseURL()
	requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	manifestJSON, err := json.Marshal(mf)
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
	n := model.ParseName(name)

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})
	existingMf, err := manifest.ParseNamedManifest(n)
	if errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		slog.Warn("pulling model with bad existing manifest", "name", name, "error", err)
	} else {
		for _, l := range existingMf.Layers {
			deleteMap[l.Digest] = struct{}{}
		}
		if existingMf.Config.Digest != "" {
			deleteMap[existingMf.Config.Digest] = struct{}{}
		}
	}

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	mf, err := pullModelManifest(ctx, n, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		if err := pullWithTransfer(ctx, n, layers, mf, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	skipVerify := make(map[string]bool)
	isHF := isHuggingFaceRegistry(n.Filepath())

	for i, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			n:       n,
			digest:  layer.Digest,
			regOpts: regOpts,
			fn:      fn,
		})
		if err != nil {
			return err
		}

		// For HuggingFace downloads, replace the HF path with the real digest
		if isHF && strings.HasPrefix(layer.Digest, "hf:") {
			if realDigest, ok := hfDigestMap.Load(layer.Digest); ok {
				layer.Digest = realDigest.(string)
				layers[i].Digest = realDigest.(string)
				// Update the manifest layers
				for j := range mf.Layers {
					if strings.HasPrefix(mf.Layers[j].Digest, "hf:") {
						if rd, ok := hfDigestMap.Load(mf.Layers[j].Digest); ok {
							mf.Layers[j].Digest = rd.(string)
						}
					}
				}
			}
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
				fp, err := manifest.BlobsPath(layer.Digest)
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
	delete(deleteMap, mf.Config.Digest)

	fn(api.ProgressResponse{Status: "writing manifest"})

	manifestJSON, err := json.Marshal(mf)
	if err != nil {
		return err
	}

	fp, err := manifest.PathForName(n)
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
func hasTensorLayers(layers []manifest.Layer) bool {
	for _, layer := range layers {
		if layer.MediaType == manifest.MediaTypeImageTensor {
			return true
		}
	}
	return false
}

// pullWithTransfer uses the simplified x/transfer package for downloading blobs.
func pullWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, mf *manifest.Manifest, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
		}
	}

	destDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
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
		Repository: n.DisplayNamespaceModel(),
		Progress:   progress,
		Token:      regOpts.Token,
		GetToken:   getToken,
		Logger:     slog.Default(),
	}); err != nil {
		return err
	}

	// Write manifest
	fn(api.ProgressResponse{Status: "writing manifest"})
	manifestJSON, err := json.Marshal(mf)
	if err != nil {
		return err
	}

	fp, err := manifest.PathForName(n)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	return os.WriteFile(fp, manifestJSON, 0o644)
}

// pushWithTransfer uses the simplified x/transfer package for uploading blobs and manifest.
func pushWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, manifestJSON []byte, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
			From:   layer.From,
		}
	}

	srcDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
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
		ManifestRef: n.Tag,
		Repository:  n.DisplayNamespaceModel(),
	})
}

func pullModelManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, error) {
	// Check if this is a HuggingFace registry
	if isHuggingFaceRegistry(n.Filepath()) {
		return pullHuggingFaceManifest(ctx, n, regOpts)
	}

	requestURL := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m manifest.Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return &m, err
}

// isHuggingFaceRegistry checks if the registry is HuggingFace
func isHuggingFaceRegistry(registry string) bool {
	return registry == "hf.co" || registry == "huggingface.co"
}

// HFFileInfo represents a file in HuggingFace's file tree
type HFFileInfo struct {
	Type string `json:"type"`
	OID  string `json:"oid"`
	Size int64  `json:"size"`
	Path string `json:"path"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs,omitempty"`
}

// pullHuggingFaceManifest pulls a model manifest from HuggingFace
func pullHuggingFaceManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, error) {
	// For HuggingFace, the tag might be "main" or could include a subdirectory like "BF16"
	// We'll use "main" as the revision and the tag as the subdirectory filter
	revision := "main"
	subdirFilter := n.Tag

	// Query HuggingFace API for file tree (always use main revision, recursive)
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s?recursive=true", n.Namespace, revision)

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating HuggingFace API request: %w", err)
	}

	if regOpts != nil && regOpts.Token != "" {
		req.Header.Set("Authorization", "Bearer "+regOpts.Token)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("querying HuggingFace API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("model not found on HuggingFace")
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HuggingFace API error (%d): %s", resp.StatusCode, string(body))
	}

	var files []HFFileInfo
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, fmt.Errorf("decoding HuggingFace API response: %w", err)
	}

	// Find GGUF files matching the tag/subdirectory
	var ggufFiles []HFFileInfo
	subdirLower := strings.ToLower(subdirFilter)
	for _, file := range files {
		if file.Type == "file" && strings.HasSuffix(file.Path, ".gguf") {
			pathLower := strings.ToLower(file.Path)
			// Match if:
			// 1. Path starts with "tag/" (directory match)
			// 2. Filename is exactly "tag.gguf" or "anything-tag.gguf" or "tag-anything.gguf"
			//    But NOT "tag_anything.gguf" to avoid Q6_K matching Q6_K_XL
			if strings.HasPrefix(pathLower, subdirLower+"/") ||
				strings.Contains(pathLower, "/"+subdirLower+"/") ||
				strings.HasSuffix(pathLower, "-"+subdirLower+".gguf") ||
				strings.Contains(pathLower, "-"+subdirLower+"-") {
				ggufFiles = append(ggufFiles, file)
			}
		}
	}

	if len(ggufFiles) == 0 {
		return nil, fmt.Errorf("no GGUF files found for tag %s", subdirFilter)
	}

	// Check if these are split GGUF files
	shardSets, singles := parser.GroupGGUFShards(extractPaths(ggufFiles))

	var mf manifest.Manifest
	mf.SchemaVersion = 2
	mf.MediaType = "application/vnd.docker.distribution.manifest.v2+json"

	// Handle split GGUF files
	if len(shardSets) > 0 {
		slog.Info("detected split GGUF files", "shards", len(shardSets[0].Shards))

		// Use the first (and should be only) shard set
		for _, shardPath := range shardSets[0].Shards {
			// Find the file info for this shard
			var fileInfo *HFFileInfo
			for i := range ggufFiles {
				if strings.HasSuffix(ggufFiles[i].Path, filepath.Base(shardPath)) {
					fileInfo = &ggufFiles[i]
					break
				}
			}

			if fileInfo == nil {
				return nil, fmt.Errorf("shard file info not found: %s", shardPath)
			}

			// Create a layer for this shard
			layer := manifest.Layer{
				MediaType: "application/vnd.ollama.image.model",
				Size:      fileInfo.Size,
				Digest:    "", // Will be computed during download
			}

			// Store the HuggingFace download URL in the layer
			// We'll use the digest field temporarily to store the download path
			layer.Digest = fmt.Sprintf("hf:%s/%s/%s", n.Namespace, n.Tag, fileInfo.Path)

			mf.Layers = append(mf.Layers, layer)
		}
	} else if len(singles) > 0 {
		// Single GGUF file
		slog.Info("detected single GGUF file", "file", singles[0])

		var fileInfo *HFFileInfo
		for i := range ggufFiles {
			if strings.HasSuffix(ggufFiles[i].Path, filepath.Base(singles[0])) {
				fileInfo = &ggufFiles[i]
				break
			}
		}

		if fileInfo == nil {
			return nil, fmt.Errorf("GGUF file info not found")
		}

		layer := manifest.Layer{
			MediaType: "application/vnd.ollama.image.model",
			Size:      fileInfo.Size,
			Digest:    fmt.Sprintf("hf:%s/%s/%s", n.Namespace, n.Tag, fileInfo.Path),
		}

		mf.Layers = append(mf.Layers, layer)
	}

	return &mf, nil
}

// extractPaths extracts file paths from HFFileInfo slice
func extractPaths(files []HFFileInfo) []string {
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.Path
	}
	return paths
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
	fp, err := manifest.BlobsPath(digest)
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
