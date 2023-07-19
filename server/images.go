package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/parser"
)

type Model struct {
	Name      string `json:"name"`
	ModelPath string
	Prompt    string
	Options   api.Options
}

type ManifestV2 struct {
	SchemaVersion int      `json:"schemaVersion"`
	MediaType     string   `json:"mediaType"`
	Config        Layer    `json:"config"`
	Layers        []*Layer `json:"layers"`
}

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int    `json:"size"`
}

type LayerReader struct {
	Layer
	io.Reader
}

type ConfigV2 struct {
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

func (m *ManifestV2) GetTotalSize() int {
	var total int
	for _, layer := range m.Layers {
		total += layer.Size
	}
	total += m.Config.Size
	return total
}

func GetManifest(mp ModelPath) (*ManifestV2, error) {
	fp, err := mp.GetManifestPath(false)
	if err != nil {
		return nil, err
	}
	if _, err = os.Stat(fp); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("couldn't find model '%s'", mp.GetShortTagname())
	}

	var manifest *ManifestV2

	f, err := os.Open(fp)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file '%s'", fp)
	}

	decoder := json.NewDecoder(f)
	err = decoder.Decode(&manifest)
	if err != nil {
		return nil, err
	}

	return manifest, nil
}

func GetModel(name string) (*Model, error) {
	mp := ParseModelPath(name)

	manifest, err := GetManifest(mp)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name: mp.GetFullTagname(),
	}

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
		case "application/vnd.ollama.image.prompt":
			data, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			model.Prompt = string(data)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			var opts api.Options
			if err = json.NewDecoder(params).Decode(&opts); err != nil {
				return nil, err
			}

			model.Options = opts
		}
	}

	return model, nil
}

func getAbsPath(fp string) (string, error) {
	if strings.HasPrefix(fp, "~/") {
		parts := strings.Split(fp, "/")
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}

		fp = filepath.Join(home, filepath.Join(parts[1:]...))
	}

	return os.ExpandEnv(fp), nil
}

func CreateModel(name string, mf io.Reader, fn func(status string)) error {
	fn("parsing modelfile")
	commands, err := parser.Parse(mf)
	if err != nil {
		fn(fmt.Sprintf("error: %v", err))
		return err
	}

	var layers []*LayerReader
	params := make(map[string]string)

	for _, c := range commands {
		log.Printf("[%s] - %s\n", c.Name, c.Arg)
		switch c.Name {
		case "model":
			fn("looking for model")
			mf, err := GetManifest(ParseModelPath(c.Arg))
			if err != nil {
				// if we couldn't read the manifest, try getting the bin file
				fp, err := getAbsPath(c.Arg)
				if err != nil {
					fn("error determing path. exiting.")
					return err
				}

				fn("creating model layer")
				file, err := os.Open(fp)
				if err != nil {
					fn(fmt.Sprintf("couldn't find model '%s'", c.Arg))
					return fmt.Errorf("failed to open file: %v", err)
				}
				defer file.Close()

				l, err := CreateLayer(file)
				if err != nil {
					fn(fmt.Sprintf("couldn't create model layer: %v", err))
					return fmt.Errorf("failed to create layer: %v", err)
				}
				l.MediaType = "application/vnd.ollama.image.model"
				layers = append(layers, l)
			} else {
				log.Printf("manifest = %#v", mf)
				for _, l := range mf.Layers {
					newLayer, err := GetLayerWithBufferFromLayer(l)
					if err != nil {
						fn(fmt.Sprintf("couldn't read layer: %v", err))
						return err
					}
					layers = append(layers, newLayer)
				}
			}
		case "prompt":
			fn("creating prompt layer")
			// remove the prompt layer if one exists
			layers = removeLayerFromLayers(layers, "application/vnd.ollama.image.prompt")

			prompt := strings.NewReader(c.Arg)
			l, err := CreateLayer(prompt)
			if err != nil {
				fn(fmt.Sprintf("couldn't create prompt layer: %v", err))
				return fmt.Errorf("failed to create layer: %v", err)
			}
			l.MediaType = "application/vnd.ollama.image.prompt"
			layers = append(layers, l)
		default:
			params[c.Name] = c.Arg
		}
	}

	// Create a single layer for the parameters
	if len(params) > 0 {
		fn("creating parameter layer")
		layers = removeLayerFromLayers(layers, "application/vnd.ollama.image.params")
		paramData, err := paramsToReader(params)
		if err != nil {
			return fmt.Errorf("couldn't create params json: %v", err)
		}
		l, err := CreateLayer(paramData)
		if err != nil {
			return fmt.Errorf("failed to create layer: %v", err)
		}
		l.MediaType = "application/vnd.ollama.image.params"
		layers = append(layers, l)
	}

	digests, err := getLayerDigests(layers)
	if err != nil {
		return err
	}

	var manifestLayers []*Layer
	for _, l := range layers {
		manifestLayers = append(manifestLayers, &l.Layer)
	}

	// Create a layer for the config object
	fn("creating config layer")
	cfg, err := createConfigLayer(digests)
	if err != nil {
		return err
	}
	layers = append(layers, cfg)

	err = SaveLayers(layers, fn, false)
	if err != nil {
		fn(fmt.Sprintf("error saving layers: %v", err))
		return err
	}

	// Create the manifest
	fn("writing manifest")
	err = CreateManifest(name, cfg, manifestLayers)
	if err != nil {
		fn(fmt.Sprintf("error creating manifest: %v", err))
		return err
	}

	fn("success")
	return nil
}

func removeLayerFromLayers(layers []*LayerReader, mediaType string) []*LayerReader {
	j := 0
	for _, l := range layers {
		if l.MediaType != mediaType {
			layers[j] = l
			j++
		}
	}
	return layers[:j]
}

func SaveLayers(layers []*LayerReader, fn func(status string), force bool) error {
	// Write each of the layers to disk
	for _, layer := range layers {
		fp, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return err
		}

		_, err = os.Stat(fp)
		if os.IsNotExist(err) || force {
			fn(fmt.Sprintf("writing layer %s", layer.Digest))
			out, err := os.Create(fp)
			if err != nil {
				log.Printf("couldn't create %s", fp)
				return err
			}
			defer out.Close()

			if _, err = io.Copy(out, layer.Reader); err != nil {
				return err
			}

		} else {
			fn(fmt.Sprintf("using already created layer %s", layer.Digest))
		}
	}

	return nil
}

func CreateManifest(name string, cfg *LayerReader, layers []*Layer) error {
	mp := ParseModelPath(name)

	manifest := ManifestV2{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config: Layer{
			MediaType: cfg.MediaType,
			Size:      cfg.Size,
			Digest:    cfg.Digest,
		},
		Layers: layers,
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath(true)
	if err != nil {
		return err
	}
	return os.WriteFile(fp, manifestJSON, 0o644)
}

func GetLayerWithBufferFromLayer(layer *Layer) (*LayerReader, error) {
	fp, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}

	file, err := os.Open(fp)
	if err != nil {
		return nil, fmt.Errorf("could not open blob: %w", err)
	}
	defer file.Close()

	newLayer, err := CreateLayer(file)
	if err != nil {
		return nil, err
	}
	newLayer.MediaType = layer.MediaType
	return newLayer, nil
}

func paramsToReader(params map[string]string) (io.ReadSeeker, error) {
	opts := api.DefaultOptions()
	typeOpts := reflect.TypeOf(opts)

	// build map of json struct tags
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	valueOpts := reflect.ValueOf(&opts).Elem()
	// iterate params and set values based on json struct tags
	for key, val := range params {
		if opt, ok := jsonOpts[key]; ok {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				switch field.Kind() {
				case reflect.Float32:
					floatVal, err := strconv.ParseFloat(val, 32)
					if err != nil {
						return nil, fmt.Errorf("invalid float value %s", val)
					}

					field.SetFloat(floatVal)
				case reflect.Int:
					intVal, err := strconv.ParseInt(val, 10, 0)
					if err != nil {
						return nil, fmt.Errorf("invalid int value %s", val)
					}

					field.SetInt(intVal)
				case reflect.Bool:
					boolVal, err := strconv.ParseBool(val)
					if err != nil {
						return nil, fmt.Errorf("invalid bool value %s", val)
					}

					field.SetBool(boolVal)
				case reflect.String:
					field.SetString(val)
				default:
					return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
				}
			}
		}
	}

	bts, err := json.Marshal(opts)
	if err != nil {
		return nil, err
	}

	return bytes.NewReader(bts), nil
}

func getLayerDigests(layers []*LayerReader) ([]string, error) {
	var digests []string
	for _, l := range layers {
		if l.Digest == "" {
			return nil, fmt.Errorf("layer is missing a digest")
		}
		digests = append(digests, l.Digest)
	}
	return digests, nil
}

// CreateLayer creates a Layer object from a given file
func CreateLayer(f io.ReadSeeker) (*LayerReader, error) {
	digest, size := GetSHA256Digest(f)
	f.Seek(0, 0)

	layer := &LayerReader{
		Layer: Layer{
			MediaType: "application/vnd.docker.image.rootfs.diff.tar",
			Digest:    digest,
			Size:      size,
		},
		Reader: f,
	}

	return layer, nil
}

func PushModel(name, username, password string, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	fn(api.ProgressResponse{Status: "retrieving manifest"})

	manifest, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []*Layer
	var total int
	var completed int
	for _, layer := range manifest.Layers {
		layers = append(layers, layer)
		total += layer.Size
	}
	layers = append(layers, &manifest.Config)
	total += manifest.Config.Size

	for _, layer := range layers {
		exists, err := checkBlobExistence(mp, layer.Digest, username, password)
		if err != nil {
			return err
		}

		if exists {
			completed += layer.Size
			fn(api.ProgressResponse{
				Status:    "using existing layer",
				Digest:    layer.Digest,
				Total:     total,
				Completed: completed,
			})
			continue
		}

		fn(api.ProgressResponse{
			Status:    "starting upload",
			Digest:    layer.Digest,
			Total:     total,
			Completed: completed,
		})

		location, err := startUpload(mp, username, password)
		if err != nil {
			log.Printf("couldn't start upload: %v", err)
			return err
		}

		err = uploadBlob(location, layer, username, password)
		if err != nil {
			log.Printf("error uploading blob: %v", err)
			return err
		}
		completed += layer.Size
		fn(api.ProgressResponse{
			Status:    "upload complete",
			Digest:    layer.Digest,
			Total:     total,
			Completed: completed,
		})
	}

	fn(api.ProgressResponse{
		Status:    "pushing manifest",
		Total:     total,
		Completed: completed,
	})
	url := fmt.Sprintf("%s://%s/v2/%s/manifests/%s", mp.ProtocolScheme, mp.Registry, mp.GetNamespaceRepository(), mp.Tag)
	headers := map[string]string{
		"Content-Type": "application/vnd.docker.distribution.manifest.v2+json",
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	resp, err := makeRequest("PUT", url, headers, bytes.NewReader(manifestJSON), username, password)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	fn(api.ProgressResponse{
		Status:    "success",
		Total:     total,
		Completed: completed,
	})

	return nil
}

func PullModel(name, username, password string, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err := pullModelManifest(mp, username, password)
	if err != nil {
		return fmt.Errorf("pull model manifest: %q", err)
	}

	var layers []*Layer
	var total int
	var completed int
	for _, layer := range manifest.Layers {
		layers = append(layers, layer)
		total += layer.Size
	}
	layers = append(layers, &manifest.Config)
	total += manifest.Config.Size

	for _, layer := range layers {
		if err := downloadBlob(mp, layer.Digest, username, password, fn); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("error downloading: %v", err), Digest: layer.Digest})
			return err
		}

		completed += layer.Size
	}

	fn(api.ProgressResponse{Status: "writing manifest"})

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath(true)
	if err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func pullModelManifest(mp ModelPath, username, password string) (*ManifestV2, error) {
	url := fmt.Sprintf("%s://%s/v2/%s/manifests/%s", mp.ProtocolScheme, mp.Registry, mp.GetNamespaceRepository(), mp.Tag)
	headers := map[string]string{
		"Accept": "application/vnd.docker.distribution.manifest.v2+json",
	}

	resp, err := makeRequest("GET", url, headers, nil, username, password)
	if err != nil {
		log.Printf("couldn't get manifest: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("registry responded with code %d: %s", resp.StatusCode, body)
	}

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

func createConfigLayer(layers []string) (*LayerReader, error) {
	// TODO change architecture and OS
	config := ConfigV2{
		Architecture: "arm64",
		OS:           "linux",
		RootFS: RootFS{
			Type:    "layers",
			DiffIDs: layers,
		},
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(configJSON)
	digest, size := GetSHA256Digest(buf)

	layer := &LayerReader{
		Layer: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    digest,
			Size:      size,
		},
		Reader: buf,
	}
	return layer, nil
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), int(n)
}

func startUpload(mp ModelPath, username string, password string) (string, error) {
	url := fmt.Sprintf("%s://%s/v2/%s/blobs/uploads/", mp.ProtocolScheme, mp.Registry, mp.GetNamespaceRepository())

	resp, err := makeRequest("POST", url, nil, nil, username, password)
	if err != nil {
		log.Printf("couldn't start upload: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	// Check for success
	if resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("registry responded with code %d: %s", resp.StatusCode, body)
	}

	// Extract UUID location from header
	location := resp.Header.Get("Location")
	if location == "" {
		return "", fmt.Errorf("location header is missing in response")
	}

	return location, nil
}

// Function to check if a blob already exists in the Docker registry
func checkBlobExistence(mp ModelPath, digest string, username string, password string) (bool, error) {
	url := fmt.Sprintf("%s://%s/v2/%s/blobs/%s", mp.ProtocolScheme, mp.Registry, mp.GetNamespaceRepository(), digest)

	resp, err := makeRequest("HEAD", url, nil, nil, username, password)
	if err != nil {
		log.Printf("couldn't check for blob: %v", err)
		return false, err
	}
	defer resp.Body.Close()

	// Check for success: If the blob exists, the Docker registry will respond with a 200 OK
	return resp.StatusCode == http.StatusOK, nil
}

func uploadBlob(location string, layer *Layer, username string, password string) error {
	// Create URL
	url := fmt.Sprintf("%s&digest=%s", location, layer.Digest)

	headers := make(map[string]string)
	headers["Content-Length"] = fmt.Sprintf("%d", layer.Size)
	headers["Content-Type"] = "application/octet-stream"

	// TODO change from monolithic uploads to chunked uploads
	// TODO allow resumability
	// TODO allow canceling uploads via DELETE
	// TODO allow cross repo blob mount

	fp, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}

	resp, err := makeRequest("PUT", url, headers, f, username, password)
	if err != nil {
		log.Printf("couldn't upload blob: %v", err)
		return err
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	return nil
}

func downloadBlob(mp ModelPath, digest string, username, password string, fn func(api.ProgressResponse)) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	if fi, _ := os.Stat(fp); fi != nil {
		// we already have the file, so return
		fn(api.ProgressResponse{
			Digest:    digest,
			Total:     int(fi.Size()),
			Completed: int(fi.Size()),
		})

		return nil
	}

	var size int64

	fi, err := os.Stat(fp + "-partial")
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return fmt.Errorf("stat: %w", err)
	default:
		size = fi.Size()
	}

	url := fmt.Sprintf("%s://%s/v2/%s/blobs/%s", mp.ProtocolScheme, mp.Registry, mp.GetNamespaceRepository(), digest)
	headers := map[string]string{
		"Range": fmt.Sprintf("bytes=%d-", size),
	}

	resp, err := makeRequest("GET", url, headers, nil, username, password)
	if err != nil {
		log.Printf("couldn't download blob: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	err = os.MkdirAll(path.Dir(fp), 0o700)
	if err != nil {
		return fmt.Errorf("make blobs directory: %w", err)
	}

	out, err := os.OpenFile(fp+"-partial", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	remaining, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	completed := size
	total := remaining + completed

	for {
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("downloading %s", digest),
			Digest:    digest,
			Total:     int(total),
			Completed: int(completed),
		})

		if completed >= total {
			if err := os.Rename(fp+"-partial", fp); err != nil {
				fn(api.ProgressResponse{
					Status:    fmt.Sprintf("error renaming file: %v", err),
					Digest:    digest,
					Total:     int(total),
					Completed: int(completed),
				})
				return err
			}

			break
		}

		n, err := io.CopyN(out, resp.Body, 8192)
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		completed += n
	}

	log.Printf("success getting %s\n", digest)
	return nil
}

func makeRequest(method, url string, headers map[string]string, body io.Reader, username, password string) (*http.Response, error) {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// TODO: better auth
	if username != "" && password != "" {
		req.SetBasicAuth(username, password)
	}

	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			log.Printf("redirected to: %s\n", req.URL)
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}
