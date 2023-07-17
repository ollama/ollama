package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
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
	"strconv"
	"strings"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/parser"
)

var DefaultRegistry string = "https://registry.ollama.ai"

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

type LayerWithBuffer struct {
	Layer

	Buffer *bytes.Buffer
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

func GetManifest(name string) (*ManifestV2, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	fp := filepath.Join(home, ".ollama/models/manifests", name)
	_, err = os.Stat(fp)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("couldn't find model '%s'", name)
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
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	manifest, err := GetManifest(name)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name: name,
	}

	for _, layer := range manifest.Layers {
		filename := filepath.Join(home, ".ollama/models/blobs", layer.Digest)
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
			/*
				f, err = os.Open(filename)
				if err != nil {
					return nil, err
				}
			*/

			var opts api.Options
			/*
				decoder = json.NewDecoder(f)
				err = decoder.Decode(&opts)
				if err != nil {
					return nil, err
				}
			*/
			model.Options = opts
		}
	}

	return model, nil
}

func getAbsPath(fn string) (string, error) {
	if strings.HasPrefix(fn, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			log.Printf("error getting home directory: %v", err)
			return "", err
		}
		fn = strings.Replace(fn, "~", home, 1)
	}

	return filepath.Abs(fn)
}

func CreateModel(name string, mf io.Reader, fn func(status string)) error {
	fn("parsing modelfile")
	commands, err := parser.Parse(mf)
	if err != nil {
		fn(fmt.Sprintf("error: %v", err))
		return err
	}

	var layers []*LayerWithBuffer
	param := make(map[string]string)

	for _, c := range commands {
		log.Printf("[%s] - %s\n", c.Name, c.Arg)
		switch c.Name {
		case "model":
			fn("looking for model")
			mf, err := GetManifest(c.Arg)
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
			param[c.Name] = c.Arg
		}
	}

	// Create a single layer for the parameters
	fn("creating parameter layer")
	if len(param) > 0 {
		layers = removeLayerFromLayers(layers, "application/vnd.ollama.image.params")
		paramData, err := paramsToReader(param)
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

func removeLayerFromLayers(layers []*LayerWithBuffer, mediaType string) []*LayerWithBuffer {
	j := 0
	for _, l := range layers {
		if l.MediaType != mediaType {
			layers[j] = l
			j++
		}
	}
	return layers[:j]
}

func SaveLayers(layers []*LayerWithBuffer, fn func(status string), force bool) error {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Printf("error getting home directory: %v", err)
		return err
	}

	dir := filepath.Join(home, ".ollama/models/blobs")

	err = os.MkdirAll(dir, 0o700)
	if err != nil {
		return fmt.Errorf("make blobs directory: %w", err)
	}

	// Write each of the layers to disk
	for _, layer := range layers {
		fp := filepath.Join(dir, layer.Digest)

		_, err = os.Stat(fp)
		if os.IsNotExist(err) || force {
			fn(fmt.Sprintf("writing layer %s", layer.Digest))
			out, err := os.Create(fp)
			if err != nil {
				log.Printf("couldn't create %s", fp)
				return err
			}
			defer out.Close()

			_, err = io.Copy(out, layer.Buffer)
			if err != nil {
				return err
			}
		} else {
			fn(fmt.Sprintf("using already created layer %s", layer.Digest))
		}
	}

	return nil
}

func CreateManifest(name string, cfg *LayerWithBuffer, layers []*Layer) error {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Printf("error getting home directory: %v", err)
		return err
	}

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

	fp := filepath.Join(home, ".ollama/models/manifests", name)
	err = os.WriteFile(fp, manifestJSON, 0644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}
	return nil
}

func GetLayerWithBufferFromLayer(layer *Layer) (*LayerWithBuffer, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	fp := filepath.Join(home, ".ollama/models/blobs", layer.Digest)
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

func paramsToReader(m map[string]string) (io.Reader, error) {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}

	return strings.NewReader(string(data)), nil
}

func getLayerDigests(layers []*LayerWithBuffer) ([]string, error) {
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
func CreateLayer(f io.Reader) (*LayerWithBuffer, error) {
	buf := new(bytes.Buffer)
	_, err := io.Copy(buf, f)
	if err != nil {
		return nil, err
	}

	digest, size := GetSHA256Digest(buf)

	layer := &LayerWithBuffer{
		Layer: Layer{
			MediaType: "application/vnd.docker.image.rootfs.diff.tar",
			Digest:    digest,
			Size:      size,
		},
		Buffer: buf,
	}

	return layer, nil
}

func PushModel(name, username, password string, fn func(status, digest string, Total, Completed int, Percent float64)) error {
	fn("retrieving manifest", "", 0, 0, 0)
	manifest, err := GetManifest(name)
	if err != nil {
		fn("couldn't retrieve manifest", "", 0, 0, 0)
		return err
	}

	var repoName string
	var tag string

	comps := strings.Split(name, ":")
	switch {
	case len(comps) < 1 || len(comps) > 2:
		return fmt.Errorf("repository name was invalid")
	case len(comps) == 1:
		repoName = comps[0]
		tag = "latest"
	case len(comps) == 2:
		repoName = comps[0]
		tag = comps[1]
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
		exists, err := checkBlobExistence(DefaultRegistry, repoName, layer.Digest, username, password)
		if err != nil {
			return err
		}

		if exists {
			completed += layer.Size
			fn("using existing layer", layer.Digest, total, completed, float64(completed)/float64(total))
			continue
		}

		fn("starting upload", layer.Digest, total, completed, float64(completed)/float64(total))

		location, err := startUpload(DefaultRegistry, repoName, username, password)
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
		fn("upload complete", layer.Digest, total, completed, float64(completed)/float64(total))
	}

	fn("pushing manifest", "", total, completed, float64(completed/total))
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", DefaultRegistry, repoName, tag)
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

	fn("success", "", total, completed, 1.0)

	return nil
}

func PullModel(name, username, password string, fn func(status, digest string, Total, Completed int, Percent float64)) error {
	var repoName string
	var tag string

	comps := strings.Split(name, ":")
	switch {
	case len(comps) < 1 || len(comps) > 2:
		return fmt.Errorf("repository name was invalid")
	case len(comps) == 1:
		repoName = comps[0]
		tag = "latest"
	case len(comps) == 2:
		repoName = comps[0]
		tag = comps[1]
	}

	fn("pulling manifest", "", 0, 0, 0)

	manifest, err := pullModelManifest(DefaultRegistry, repoName, tag, username, password)
	if err != nil {
		return fmt.Errorf("pull model manifest: %q", err)
	}

	log.Printf("manifest = %#v", manifest)

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
		fn("starting download", layer.Digest, total, completed, float64(completed)/float64(total))
		if err := downloadBlob(DefaultRegistry, repoName, layer.Digest, username, password, fn); err != nil {
			fn(fmt.Sprintf("error downloading: %v", err), layer.Digest, 0, 0, 0)
			return err
		}
		completed += layer.Size
		fn("download complete", layer.Digest, total, completed, float64(completed)/float64(total))
	}

	fn("writing manifest", "", total, completed, 1.0)

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp := filepath.Join(home, ".ollama/models/manifests", name)

	err = os.MkdirAll(path.Dir(fp), 0o700)
	if err != nil {
		return fmt.Errorf("make manifests directory: %w", err)
	}

	err = os.WriteFile(fp, manifestJSON, 0644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}

	fn("success", "", total, completed, 1.0)

	return nil
}

func pullModelManifest(registryURL, repoName, tag, username, password string) (*ManifestV2, error) {
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", registryURL, repoName, tag)
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
		return nil, fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

func createConfigLayer(layers []string) (*LayerWithBuffer, error) {
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

	layer := &LayerWithBuffer{
		Layer: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    digest,
			Size:      size,
		},
		Buffer: buf,
	}
	return layer, nil
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(data *bytes.Buffer) (string, int) {
	layerBytes := data.Bytes()
	hash := sha256.Sum256(layerBytes)
	return "sha256:" + hex.EncodeToString(hash[:]), len(layerBytes)
}

func startUpload(registryURL string, repositoryName string, username string, password string) (string, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/uploads/", registryURL, repositoryName)

	resp, err := makeRequest("POST", url, nil, nil, username, password)
	if err != nil {
		log.Printf("couldn't start upload: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	// Check for success
	if resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	// Extract UUID location from header
	location := resp.Header.Get("Location")
	if location == "" {
		return "", fmt.Errorf("location header is missing in response")
	}

	return location, nil
}

// Function to check if a blob already exists in the Docker registry
func checkBlobExistence(registryURL string, repositoryName string, digest string, username string, password string) (bool, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/%s", registryURL, repositoryName, digest)

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
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	// Create URL
	url := fmt.Sprintf("%s&digest=%s", location, layer.Digest)

	headers := make(map[string]string)
	headers["Content-Length"] = fmt.Sprintf("%d", layer.Size)
	headers["Content-Type"] = "application/octet-stream"

	// TODO change from monolithic uploads to chunked uploads
	// TODO allow resumability
	// TODO allow canceling uploads via DELETE
	// TODO allow cross repo blob mount

	fp := filepath.Join(home, ".ollama/models/blobs", layer.Digest)
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

func downloadBlob(registryURL, repoName, digest string, username, password string, fn func(status, digest string, Total, Completed int, Percent float64)) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	fp := filepath.Join(home, ".ollama/models/blobs", digest)

	_, err = os.Stat(fp)
	if !os.IsNotExist(err) {
		// we already have the file, so return
		log.Printf("already have %s\n", digest)
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

	url := fmt.Sprintf("%s/v2/%s/blobs/%s", registryURL, repoName, digest)
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
		fn(fmt.Sprintf("Downloading %s", digest), digest, int(total), int(completed), float64(completed)/float64(total))
		if completed >= total {
			fmt.Printf("finished downloading\n")
			err = os.Rename(fp+"-partial", fp)
			if err != nil {
				fmt.Printf("error: %v\n", err)
				fn(fmt.Sprintf("error renaming file: %v", err), digest, int(total), int(completed), 1)
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
