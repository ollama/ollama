package server

import (
	"bytes"
	"cmp"
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
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

var (
	errCapabilities         = errors.New("does not support")
	errCapabilityCompletion = errors.New("completion")
	errCapabilityTools      = errors.New("tools")
	errCapabilityInsert     = errors.New("insert")
)

type Capability string

const (
	CapabilityCompletion = Capability("completion")
	CapabilityTools      = Capability("tools")
	CapabilityInsert     = Capability("insert")
)

type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

type Model struct {
	Name               string `json:"name"`
	Config             ConfigV2
	ShortName          string
	ModelPath          string
	ParentModel        string
	AdapterPaths       []string
	ControlVectorPaths []string
	ProjectorPaths     []string
	System             string
	License            []string
	Digest             string
	Options            map[string]interface{}
	Messages           []api.Message

	Template *template.Template
}

// CheckCapabilities checks if the model has the specified capabilities returning an error describing
// any missing or unknown capabilities
func (m *Model) CheckCapabilities(caps ...Capability) error {
	var errs []error
	for _, cap := range caps {
		switch cap {
		case CapabilityCompletion:
			f, err := os.Open(m.ModelPath)
			if err != nil {
				slog.Error("couldn't open model file", "error", err)
				continue
			}
			defer f.Close()

			// TODO(mxyng): decode the GGML into model to avoid doing this multiple times
			ggml, _, err := llm.DecodeGGML(f, 0)
			if err != nil {
				slog.Error("couldn't decode ggml", "error", err)
				continue
			}

			if _, ok := ggml.KV()[fmt.Sprintf("%s.pooling_type", ggml.KV().Architecture())]; ok {
				errs = append(errs, errCapabilityCompletion)
			}
		case CapabilityTools:
			if !slices.Contains(m.Template.Vars(), "tools") {
				errs = append(errs, errCapabilityTools)
			}
		case CapabilityInsert:
			vars := m.Template.Vars()
			if !slices.Contains(vars, "suffix") {
				errs = append(errs, errCapabilityInsert)
			}
		default:
			slog.Error("unknown capability", "capability", cap)
			return fmt.Errorf("unknown capability: %s", cap)
		}
	}

	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))
	}

	return nil
}

func (m *Model) String() string {
	var modelfile parser.File

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

	for _, control := range m.ControlVectorPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "controlvector",
			Args: control,
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

type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"`
	FileType      string   `json:"file_type"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
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
		case "application/vnd.ollama.image.controlvector":
			model.ControlVectorPaths = append(model.ControlVectorPaths, filename)
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

func realpath(rel, from string) string {
	abspath, err := filepath.Abs(from)
	if err != nil {
		return from
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return abspath
	}

	if from == "~" {
		return home
	} else if strings.HasPrefix(from, "~/") {
		return filepath.Join(home, from[2:])
	}

	if _, err := os.Stat(filepath.Join(rel, from)); err == nil {
		// this is a file relative to the Modelfile
		return filepath.Join(rel, from)
	}

	return abspath
}

func CreateModel(ctx context.Context, name model.Name, modelFileDir, quantization string, modelfile *parser.File, fn func(resp api.ProgressResponse)) (err error) {
	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
	}

	var messages []*api.Message
	parameters := make(map[string]any)

	var layers []Layer
	var baseLayers []*layerGGML
	for _, c := range modelfile.Commands {
		mediatype := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)
		command := c.Name

		switch command {
		case "model", "adapter":
			if name := model.ParseName(c.Args); name.IsValid() && command == "model" {
				name, err := getExistingName(name)
				if err != nil {
					return err
				}
				baseLayers, err = parseFromModel(ctx, name, fn)
				if err != nil {
					return err
				}
			} else if strings.HasPrefix(c.Args, "@") {
				digest := strings.TrimPrefix(c.Args, "@")
				if ib, ok := intermediateBlobs[digest]; ok {
					p, err := GetBlobsPath(ib)
					if err != nil {
						return err
					}

					if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
						// pass
					} else if err != nil {
						return err
					} else {
						fn(api.ProgressResponse{Status: fmt.Sprintf("using cached layer %s", ib)})
						digest = ib
					}
				}

				blobpath, err := GetBlobsPath(digest)
				if err != nil {
					return err
				}

				blob, err := os.Open(blobpath)
				if err != nil {
					return err
				}
				defer blob.Close()

				baseLayers, err = parseFromFile(ctx, command, baseLayers, blob, digest, fn)
				if err != nil {
					return err
				}
			} else if file, err := os.Open(realpath(modelFileDir, c.Args)); err == nil {
				defer file.Close()

				baseLayers, err = parseFromFile(ctx, command, baseLayers, file, "", fn)
				if err != nil {
					return err
				}
			} else {
				return fmt.Errorf("invalid model reference: %s", c.Args)
			}

			for _, baseLayer := range baseLayers {
				if quantization != "" &&
					baseLayer.MediaType == "application/vnd.ollama.image.model" &&
					baseLayer.GGML != nil &&
					baseLayer.GGML.Name() == "gguf" {
					want, err := llm.ParseFileType(quantization)
					if err != nil {
						return err
					}

					ft := baseLayer.GGML.KV().FileType()
					if !slices.Contains([]string{"F16", "F32"}, ft.String()) {
						return errors.New("quantization is only supported for F16 and F32 models")
					} else if want != ft {
						fn(api.ProgressResponse{Status: fmt.Sprintf("quantizing %s model to %s", ft, quantization)})

						blob, err := GetBlobsPath(baseLayer.Digest)
						if err != nil {
							return err
						}

						temp, err := os.CreateTemp(filepath.Dir(blob), quantization)
						if err != nil {
							return err
						}
						defer temp.Close()
						defer os.Remove(temp.Name())

						if err := llama.Quantize(blob, temp.Name(), uint32(want)); err != nil {
							return err
						}

						layer, err := NewLayer(temp, baseLayer.MediaType)
						if err != nil {
							return err
						}

						if _, err := temp.Seek(0, io.SeekStart); err != nil {
							return err
						}

						ggml, _, err := llm.DecodeGGML(temp, 0)
						if err != nil {
							return err
						}

						baseLayer.Layer = layer
						baseLayer.GGML = ggml
					}
				}

				if baseLayer.GGML != nil {
					config.ModelFormat = cmp.Or(config.ModelFormat, baseLayer.GGML.Name())
					config.ModelFamily = cmp.Or(config.ModelFamily, baseLayer.GGML.KV().Architecture())
					config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(baseLayer.GGML.KV().ParameterCount()))
					config.FileType = cmp.Or(config.FileType, baseLayer.GGML.KV().FileType().String())
					config.ModelFamilies = append(config.ModelFamilies, baseLayer.GGML.KV().Architecture())
				}

				layers = append(layers, baseLayer.Layer)
			}
		case "controlvector":
			// Save the gguf control vector without any conversion
			path := strings.Trim(c.Args, " \t\r")

			file, err := os.Open(realpath(modelFileDir, path))
			if err != nil {
				return err
			}
			defer file.Close()

			// TODO validate this is a GGUF control vector

			layer, err := NewLayer(file, mediatype)
			if err != nil {
				return err
			}

			layers = append(layers, layer)

		case "license", "template", "system":
			if c.Name == "template" {
				if _, err := template.Parse(c.Args); err != nil {
					return fmt.Errorf("%w: %s", errBadTemplate, err)
				}
			}

			if c.Name != "license" {
				// replace
				layers = slices.DeleteFunc(layers, func(layer Layer) bool {
					if layer.MediaType != mediatype {
						return false
					}

					if err := layer.Remove(); err != nil {
						return false
					}

					return true
				})
			}

			blob := strings.NewReader(c.Args)
			layer, err := NewLayer(blob, mediatype)
			if err != nil {
				return err
			}

			layers = append(layers, layer)
		case "message":
			role, content, ok := strings.Cut(c.Args, ": ")
			if !ok {
				return fmt.Errorf("invalid message: %s", c.Args)
			}

			messages = append(messages, &api.Message{Role: role, Content: content})
		default:
			ps, err := api.FormatParams(map[string][]string{c.Name: {c.Args}})
			if err != nil {
				return err
			}

			for k, v := range ps {
				if ks, ok := parameters[k].([]string); ok {
					parameters[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					parameters[k] = vs
				} else {
					parameters[k] = v
				}
			}
		}
	}

	var err2 error
	layers = slices.DeleteFunc(layers, func(layer Layer) bool {
		switch layer.MediaType {
		case "application/vnd.ollama.image.message":
			// if there are new messages, remove the inherited ones
			if len(messages) > 0 {
				return true
			}

			return false
		case "application/vnd.ollama.image.params":
			// merge inherited parameters with new ones
			r, err := layer.Open()
			if err != nil {
				err2 = err
				return false
			}
			defer r.Close()

			var ps map[string]any
			if err := json.NewDecoder(r).Decode(&ps); err != nil {
				err2 = err
				return false
			}

			for k, v := range ps {
				if _, ok := parameters[k]; !ok {
					parameters[k] = v
				}
			}

			return true
		default:
			return false
		}
	})

	if err2 != nil {
		return err2
	}

	if len(messages) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(messages); err != nil {
			return err
		}

		layer, err := NewLayer(&b, "application/vnd.ollama.image.messages")
		if err != nil {
			return err
		}

		layers = append(layers, layer)
	}

	if len(parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(parameters); err != nil {
			return err
		}

		layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return err
		}

		layers = append(layers, layer)
	}

	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}

	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return err
	}

	configLayer, err := NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return err
	}

	for _, layer := range append(layers, configLayer) {
		if layer.status != "" {
			fn(api.ProgressResponse{Status: layer.status})
		}
	}

	old, _ := ParseNamedManifest(name)

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := WriteManifest(name, configLayer, layers); err != nil {
		return err
	}

	if !envconfig.NoPrune() && old != nil {
		if err := old.RemoveLayers(); err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
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
		return errors.New("insecure protocol http")
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
		return errors.New("insecure protocol http")
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
	delete(deleteMap, manifest.Config.Digest)

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if skipVerify[layer.Digest] {
			continue
		}
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				// something went wrong, delete the blob
				fp, err := GetBlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					// log this, but return the original error
					slog.Info(fmt.Sprintf("couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
			}
			return err
		}
	}

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
			token, err := getAuthorizationToken(ctx, challenge)
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
