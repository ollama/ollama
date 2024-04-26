package server

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/exp/slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string
}

type Model struct {
	Name           string `json:"name"`
	Config         ConfigV2
	ShortName      string
	ModelPath      string
	ParentModel    string
	AdapterPaths   []string
	ProjectorPaths []string
	Template       string
	System         string
	License        []string
	Digest         string
	Size           int64
	Options        map[string]interface{}
	Messages       []Message
}

func (m *Model) IsEmbedding() bool {
	return slices.Contains(m.Config.ModelFamilies, "bert") || slices.Contains(m.Config.ModelFamilies, "nomic-bert")
}

func (m *Model) Commands() (cmds []parser.Command) {
	cmds = append(cmds, parser.Command{Name: "model", Args: m.ModelPath})

	if m.Template != "" {
		cmds = append(cmds, parser.Command{Name: "template", Args: m.Template})
	}

	if m.System != "" {
		cmds = append(cmds, parser.Command{Name: "system", Args: m.System})
	}

	for _, adapter := range m.AdapterPaths {
		cmds = append(cmds, parser.Command{Name: "adapter", Args: adapter})
	}

	for _, projector := range m.ProjectorPaths {
		cmds = append(cmds, parser.Command{Name: "projector", Args: projector})
	}

	for k, v := range m.Options {
		switch v := v.(type) {
		case []any:
			for _, s := range v {
				cmds = append(cmds, parser.Command{Name: k, Args: fmt.Sprintf("%v", s)})
			}
		default:
			cmds = append(cmds, parser.Command{Name: k, Args: fmt.Sprintf("%v", v)})
		}
	}

	for _, license := range m.License {
		cmds = append(cmds, parser.Command{Name: "license", Args: license})
	}

	for _, msg := range m.Messages {
		cmds = append(cmds, parser.Command{Name: "message", Args: fmt.Sprintf("%s %s", msg.Role, msg.Content)})
	}

	return cmds

}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ManifestV2 struct {
	SchemaVersion int      `json:"schemaVersion"`
	MediaType     string   `json:"mediaType"`
	Config        *Layer   `json:"config"`
	Layers        []*Layer `json:"layers"`
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

func (c *ConfigV2) SetModelFormat(format string) {
	if c.ModelFormat == "" {
		c.ModelFormat = format
	}
}

func (c *ConfigV2) SetModelFamily(families ...string) {
	for _, family := range families {
		if c.ModelFamily == "" {
			c.ModelFamily = family
		}

		if !slices.Contains(c.ModelFamilies, family) {
			c.ModelFamilies = append(c.ModelFamilies, family)
		}
	}
}

func (c *ConfigV2) SetModelType(modelType string) {
	if c.ModelType == "" {
		c.ModelType = modelType
	}
}

func (c *ConfigV2) SetFileType(fileType string) {
	if c.FileType == "" {
		c.FileType = fileType
	}
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

func (m *ManifestV2) GetTotalSize() (total int64) {
	for _, layer := range m.Layers {
		total += layer.Size
	}

	total += m.Config.Size
	return total
}

func GetManifest(mp ModelPath) (*ManifestV2, string, error) {
	fp, err := mp.GetManifestPath()
	if err != nil {
		return nil, "", err
	}

	if _, err = os.Stat(fp); err != nil {
		return nil, "", err
	}

	var manifest *ManifestV2

	bts, err := os.ReadFile(fp)
	if err != nil {
		return nil, "", fmt.Errorf("couldn't open file '%s'", fp)
	}

	shaSum := sha256.Sum256(bts)
	shaStr := hex.EncodeToString(shaSum[:])

	if err := json.Unmarshal(bts, &manifest); err != nil {
		return nil, "", err
	}

	return manifest, shaStr, nil
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
		Template:  "{{ .Prompt }}",
		License:   []string{},
		Size:      manifest.GetTotalSize(),
	}

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
		case "application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.System = string(bts)
		case "application/vnd.ollama.image.prompt":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
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

func realpath(mfDir, from string) string {
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

	if _, err := os.Stat(filepath.Join(mfDir, from)); err == nil {
		// this is a file relative to the Modelfile
		return filepath.Join(mfDir, from)
	}

	return abspath
}

func CreateModel(ctx context.Context, name, modelFileDir, quantization string, commands []parser.Command, fn func(resp api.ProgressResponse)) error {
	deleteMap := make(map[string]struct{})
	if manifest, _, err := GetManifest(ParseModelPath(name)); err == nil {
		for _, layer := range append(manifest.Layers, manifest.Config) {
			deleteMap[layer.Digest] = struct{}{}
		}
	}

	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
	}

	var layers Layers
	messages := []string{}

	params := make(map[string][]string)
	fromParams := make(map[string]any)

	for _, c := range commands {
		mediatype := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)

		switch c.Name {
		case "model":
			if strings.HasPrefix(c.Args, "@") {
				blobPath, err := GetBlobsPath(strings.TrimPrefix(c.Args, "@"))
				if err != nil {
					return err
				}

				c.Args = blobPath
			}

			pathName := realpath(modelFileDir, c.Args)

			ggufName, err := convertModel(name, pathName, fn)
			if err != nil {
				var pathErr *fs.PathError
				switch {
				case errors.Is(err, zip.ErrFormat):
					// it's not a safetensor archive
				case errors.As(err, &pathErr):
					// it's not a file on disk, could be a model reference
				default:
					return err
				}
			}

			if ggufName != "" {
				pathName = ggufName
				defer os.RemoveAll(ggufName)

				if quantization != "" {
					quantization = strings.ToUpper(quantization)
					fn(api.ProgressResponse{Status: fmt.Sprintf("quantizing %s model to %s", "F16", quantization)})
					tempfile, err := os.CreateTemp(filepath.Dir(ggufName), quantization)
					if err != nil {
						return err
					}
					defer os.RemoveAll(tempfile.Name())

					if err := llm.Quantize(ggufName, tempfile.Name(), quantization); err != nil {
						return err
					}

					if err := tempfile.Close(); err != nil {
						return err
					}

					pathName = tempfile.Name()
				}
			}

			bin, err := os.Open(pathName)
			if err != nil {
				// not a file on disk so must be a model reference
				modelpath := ParseModelPath(c.Args)
				manifest, _, err := GetManifest(modelpath)
				switch {
				case errors.Is(err, os.ErrNotExist):
					fn(api.ProgressResponse{Status: "pulling model"})
					if err := PullModel(ctx, c.Args, &registryOptions{}, fn); err != nil {
						return err
					}

					manifest, _, err = GetManifest(modelpath)
					if err != nil {
						return err
					}
				case err != nil:
					return err
				}

				fn(api.ProgressResponse{Status: "reading model metadata"})
				fromConfigPath, err := GetBlobsPath(manifest.Config.Digest)
				if err != nil {
					return err
				}

				fromConfigFile, err := os.Open(fromConfigPath)
				if err != nil {
					return err
				}
				defer fromConfigFile.Close()

				var fromConfig ConfigV2
				if err := json.NewDecoder(fromConfigFile).Decode(&fromConfig); err != nil {
					return err
				}

				// if the model is still not in gguf format, error out
				if fromConfig.ModelFormat != "gguf" {
					return fmt.Errorf("%s is not in gguf format, this base model is not compatible with this version of ollama", c.Args)
				}

				config.SetModelFormat(fromConfig.ModelFormat)
				config.SetModelFamily(append(fromConfig.ModelFamilies, fromConfig.ModelFamily)...)
				config.SetModelType(fromConfig.ModelType)
				config.SetFileType(fromConfig.FileType)

				for _, layer := range manifest.Layers {
					deleteMap[layer.Digest] = struct{}{}
					if layer.MediaType == "application/vnd.ollama.image.params" {
						fromParamsPath, err := GetBlobsPath(layer.Digest)
						if err != nil {
							return err
						}

						fromParamsFile, err := os.Open(fromParamsPath)
						if err != nil {
							return err
						}
						defer fromParamsFile.Close()

						if err := json.NewDecoder(fromParamsFile).Decode(&fromParams); err != nil {
							return err
						}
					}

					layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, modelpath.GetShortTagname())
					if err != nil {
						return err
					}

					layers.Add(layer)
				}

				deleteMap[manifest.Config.Digest] = struct{}{}
				continue
			}
			defer bin.Close()

			var offset int64
			for {
				fn(api.ProgressResponse{Status: "creating model layer"})
				if _, err := bin.Seek(offset, io.SeekStart); err != nil {
					return err
				}

				ggml, size, err := llm.DecodeGGML(bin)
				if errors.Is(err, io.EOF) {
					break
				} else if errors.Is(err, llm.ErrUnsupportedFormat) {
					return fmt.Errorf("model binary specified in FROM field is not a valid gguf format model, %w", err)
				} else if err != nil {
					return err
				}

				config.SetModelFormat(ggml.Name())
				config.SetModelFamily(ggml.KV().Architecture())
				config.SetModelType(format.HumanNumber(ggml.KV().ParameterCount()))
				config.SetFileType(ggml.KV().FileType())

				mediatype := mediatype
				if ggml.KV().Architecture() == "clip" {
					mediatype = "application/vnd.ollama.image.projector"
				}

				sr := io.NewSectionReader(bin, offset, size)
				layer, err := NewLayer(sr, mediatype)
				if err != nil {
					return err
				}

				layers.Add(layer)

				offset += size
			}
		case "adapter":
			if strings.HasPrefix(c.Args, "@") {
				blobPath, err := GetBlobsPath(strings.TrimPrefix(c.Args, "@"))
				if err != nil {
					return err
				}

				c.Args = blobPath
			}

			fn(api.ProgressResponse{Status: "creating adapter layer"})
			bin, err := os.Open(realpath(modelFileDir, c.Args))
			if err != nil {
				return err
			}
			defer bin.Close()

			_, size, err := llm.DecodeGGML(bin)
			if err != nil {
				return err
			}

			sr := io.NewSectionReader(bin, 0, size)
			layer, err := NewLayer(sr, mediatype)
			if err != nil {
				return err
			}

			layers.Add(layer)
		case "license":
			fn(api.ProgressResponse{Status: "creating license layer"})

			bin := strings.NewReader(c.Args)
			layer, err := NewLayer(bin, mediatype)
			if err != nil {
				return err
			}

			layers.Add(layer)
		case "template", "system":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating %s layer", c.Name)})

			bin := strings.NewReader(c.Args)
			layer, err := NewLayer(bin, mediatype)
			if err != nil {
				return err
			}

			layers.Replace(layer)
		case "message":
			messages = append(messages, c.Args)
		default:
			params[c.Name] = append(params[c.Name], c.Args)
		}
	}

	if len(messages) > 0 {
		fn(api.ProgressResponse{Status: "creating parameters layer"})

		msgs := make([]api.Message, 0)

		for _, m := range messages {
			// todo: handle images
			msg := strings.SplitN(m, ": ", 2)
			msgs = append(msgs, api.Message{Role: msg[0], Content: msg[1]})
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(msgs); err != nil {
			return err
		}

		layer, err := NewLayer(&b, "application/vnd.ollama.image.messages")
		if err != nil {
			return err
		}

		layers.Replace(layer)
	}

	if len(params) > 0 {
		fn(api.ProgressResponse{Status: "creating parameters layer"})

		formattedParams, err := api.FormatParams(params)
		if err != nil {
			return err
		}

		for k, v := range fromParams {
			if _, ok := formattedParams[k]; !ok {
				formattedParams[k] = v
			}
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(formattedParams); err != nil {
			return err
		}

		fn(api.ProgressResponse{Status: "creating config layer"})
		layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return err
		}

		layers.Replace(layer)
	}

	digests := make([]string, len(layers.items))
	for i, layer := range layers.items {
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

	delete(deleteMap, configLayer.Digest)

	for _, layer := range append(layers.items, configLayer) {
		committed, err := layer.Commit()
		if err != nil {
			return err
		}

		status := "writing layer"
		if !committed {
			status = "using already created layer"
		}

		fn(api.ProgressResponse{Status: fmt.Sprintf("%s %s", status, layer.Digest)})

		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := WriteManifest(name, configLayer, layers.items); err != nil {
		return err
	}

	if noprune := os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		if err := deleteUnusedLayers(nil, deleteMap, false); err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func convertModel(name, path string, fn func(resp api.ProgressResponse)) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()

	tempDir, err := os.MkdirTemp("", "ollama-convert")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tempDir)

	fn(api.ProgressResponse{Status: "unpacking model metadata"})
	for _, f := range r.File {
		fpath := filepath.Join(tempDir, f.Name)
		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return "", err
		}

		rc, err := f.Open()
		if err != nil {
			return "", err
		}

		_, err = io.Copy(outFile, rc)
		if err != nil {
			return "", err
		}

		outFile.Close()
		rc.Close()
	}

	mf, err := convert.GetModelFormat(tempDir)
	if err != nil {
		return "", err
	}

	params, err := mf.GetParams(tempDir)
	if err != nil {
		return "", err
	}

	mArch, err := mf.GetModelArch(name, tempDir, params)
	if err != nil {
		return "", err
	}

	fn(api.ProgressResponse{Status: "processing tensors"})
	if err := mArch.GetTensors(); err != nil {
		return "", err
	}

	if err := mArch.LoadVocab(); err != nil {
		return "", err
	}

	fn(api.ProgressResponse{Status: "converting model"})
	path, err = mArch.WriteGGUF()
	if err != nil {
		return "", err
	}

	return path, nil
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

func deleteUnusedLayers(skipModelPath *ModelPath, deleteMap map[string]struct{}, dryRun bool) error {
	fp, err := GetManifestPath()
	if err != nil {
		return err
	}

	walkFunc := func(path string, info os.FileInfo, _ error) error {
		if info.IsDir() {
			return nil
		}

		dir, file := filepath.Split(path)
		dir = strings.Trim(strings.TrimPrefix(dir, fp), string(os.PathSeparator))
		tag := strings.Join([]string{dir, file}, ":")
		fmp := ParseModelPath(tag)

		// skip the manifest we're trying to delete
		if skipModelPath != nil && skipModelPath.GetFullTagname() == fmp.GetFullTagname() {
			return nil
		}

		// save (i.e. delete from the deleteMap) any files used in other manifests
		manifest, _, err := GetManifest(fmp)
		if err != nil {
			// nolint: nilerr
			return nil
		}

		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
		return nil
	}

	if err := filepath.Walk(fp, walkFunc); err != nil {
		return err
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := GetBlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if !dryRun {
			if err := os.Remove(fp); err != nil {
				slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
				continue
			}
		} else {
			slog.Info(fmt.Sprintf("wanted to remove: %s", fp))
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
		if strings.HasPrefix(name, "sha256:") {
			deleteMap[name] = struct{}{}
		}
	}

	slog.Info(fmt.Sprintf("total blobs: %d", len(deleteMap)))

	err = deleteUnusedLayers(nil, deleteMap, false)
	if err != nil {
		return err
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

func DeleteModel(name string) error {
	mp := ParseModelPath(name)
	manifest, _, err := GetManifest(mp)
	if err != nil {
		return err
	}

	deleteMap := make(map[string]struct{})
	for _, layer := range manifest.Layers {
		deleteMap[layer.Digest] = struct{}{}
	}
	deleteMap[manifest.Config.Digest] = struct{}{}

	err = deleteUnusedLayers(&mp, deleteMap, false)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	err = os.Remove(fp)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't remove manifest file '%s': %v", fp, err))
		return err
	}

	return nil
}

func PushModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	manifest, _, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, manifest.Config)

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

	var manifest *ManifestV2
	var err error
	var noprune string

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})

	if noprune = os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		manifest, _, err = GetManifest(mp)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		if manifest != nil {
			for _, l := range manifest.Layers {
				deleteMap[l.Digest] = struct{}{}
			}
			deleteMap[manifest.Config.Digest] = struct{}{}
		}
	}

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err = pullModelManifest(ctx, mp, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, manifest.Config)

	for _, layer := range layers {
		if err := downloadBlob(
			ctx,
			downloadOpts{
				mp:      mp,
				digest:  layer.Digest,
				regOpts: regOpts,
				fn:      fn,
			}); err != nil {
			return err
		}
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, manifest.Config.Digest)

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
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

	if noprune == "" {
		fn(api.ProgressResponse{Status: "removing any unused layers"})
		err = deleteUnusedLayers(nil, deleteMap, false)
		if err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func pullModelManifest(ctx context.Context, mp ModelPath, regOpts *registryOptions) (*ManifestV2, error) {
	requestURL := mp.BaseURL().JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
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

var errUnauthorized = fmt.Errorf("unauthorized: access denied")

// getTokenSubject returns the subject of a JWT token, it does not validate the token
func getTokenSubject(token string) string {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		slog.Error("jwt token does not contain 3 parts")
		return ""
	}

	payload := parts[1]
	payloadBytes, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		slog.Error(fmt.Sprintf("failed to decode jwt payload: %v", err))
		return ""
	}

	var payloadMap map[string]interface{}
	if err := json.Unmarshal(payloadBytes, &payloadMap); err != nil {
		slog.Error(fmt.Sprintf("failed to unmarshal payload JSON: %v", err))
		return ""
	}

	sub, ok := payloadMap["sub"]
	if !ok {
		slog.Error("jwt does not contain 'sub' field")
		return ""
	}

	return fmt.Sprintf("%s", sub)
}

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *registryOptions) (*http.Response, error) {
	anonymous := true // access will default to anonymous if no user is found associated with the public key
	for i := 0; i < 2; i++ {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.Info(fmt.Sprintf("request failed: %v", err))
			}

			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			// Handle authentication error with one retry
			challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
			token, err := getAuthorizationToken(ctx, challenge)
			if err != nil {
				return nil, err
			}
			anonymous = getTokenSubject(token) == "anonymous"
			regOpts.Token = token
			if body != nil {
				_, err = body.Seek(0, io.SeekStart)
				if err != nil {
					return nil, err
				}
			}
		case resp.StatusCode == http.StatusNotFound:
			return nil, os.ErrNotExist
		case resp.StatusCode >= http.StatusBadRequest:
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
			}
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
		default:
			return resp, nil
		}
	}

	if anonymous {
		// no user is associated with the public key, and the request requires non-anonymous access
		pubKey, nestedErr := auth.GetPublicKey()
		if nestedErr != nil {
			slog.Error(fmt.Sprintf("couldn't get public key: %v", nestedErr))
			return nil, errUnauthorized
		}
		return nil, &errtypes.UnknownOllamaKey{Key: pubKey}
	}
	// user is associated with the public key, but is not authorized to make the request
	return nil, errUnauthorized
}

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

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
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
