package client

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/safetensors"
)

const maxUploadRetries = 6

// backoffDuration calculates the backoff sleep for a retry attempt.
// Overridable in tests to avoid real sleeps.
var backoffDuration = func(attempt int) time.Duration {
	return time.Duration(math.Pow(2, float64(attempt-1))) * time.Second
}

// RemoteCreateOptions holds options for remote safetensors model creation.
type RemoteCreateOptions struct {
	ModelName string
	ModelDir  string
	Quantize  string
	Modelfile *ModelfileConfig
}

// CreateModelRemote creates a model on a remote server by uploading blobs and
// sending a create request. It reuses the same x/create traversal as local
// creation; only the layer/manifest callbacks differ.
func CreateModelRemote(ctx context.Context, client *api.Client, opts RemoteCreateOptions, p *progress.Progress) error {
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)

	if !isSafetensors && !isImageGen {
		return fmt.Errorf("%s is not a supported model directory", opts.ModelDir)
	}

	var modelType, spinnerKey string
	var parserName, rendererName string
	if isSafetensors {
		modelType = "safetensors model"
		spinnerKey = "create"
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
	} else {
		modelType = "image generation model"
		spinnerKey = "imagegen"
	}

	session := newRemoteCreateSession(client, p, opts, parserName, rendererName, modelType)

	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)
	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	var err error
	if isSafetensors {
		err = create.CreateSafetensorsModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			session.layerCreator(ctx), session.tensorLayerCreator(ctx),
			session.manifestWriter(ctx),
			progressFn,
			session.packedTensorLayerCreator(ctx),
		)
	} else {
		err = create.CreateImageGenModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			session.layerCreator(ctx), session.tensorLayerCreator(ctx),
			session.manifestWriter(ctx),
			progressFn,
		)
	}

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

type remoteCreateSession struct {
	client         *api.Client
	progress       *progress.Progress
	opts           RemoteCreateOptions
	parserName     string
	rendererName   string
	modelType      string
	transferBar    *progress.Bar
	transferred    atomic.Int64
	totalSize      atomic.Int64
	fileType       string
	clientQuantize bool
}

func newRemoteCreateSession(client *api.Client, p *progress.Progress, opts RemoteCreateOptions, parserName, rendererName, modelType string) *remoteCreateSession {
	bar := progress.NewBar("transferring model", 0, 0)
	p.Add("transfer", bar)
	return &remoteCreateSession{
		client:       client,
		progress:     p,
		opts:         opts,
		parserName:   parserName,
		rendererName: rendererName,
		modelType:    modelType,
		transferBar:  bar,
		fileType:     strings.ToLower(strings.TrimSpace(opts.Quantize)),
		clientQuantize: func() bool {
			return opts.Quantize == "" || (!forceServerQuantize() && remoteQuantizeSupported())
		}(),
	}
}

var remoteQuantizeSupported = QuantizeSupported

var forceServerQuantize = func() bool {
	return boolEnv("OLLAMA_CREATE_SERVER_QUANTIZE")
}

func boolEnv(name string) bool {
	if s := strings.TrimSpace(os.Getenv(name)); s != "" {
		b, err := strconv.ParseBool(s)
		if err != nil {
			return true
		}
		return b
	}
	return false
}

func (s *remoteCreateSession) layerCreator(ctx context.Context) create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		return s.uploadLayer(ctx, r, mediaType, name)
	}
}

func (s *remoteCreateSession) tensorLayerCreator(ctx context.Context) create.QuantizingTensorLayerCreator {
	return func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
		if quantize != "" {
			if !s.clientQuantize {
				layer, err := s.uploadLayer(ctx, r, manifest.MediaTypeImageTensor, name)
				if err != nil {
					return nil, err
				}
				return []create.LayerInfo{layer}, nil
			}

			if s.fileType == "" {
				s.fileType = quantize
			}
			blobData, err := QuantizeTensor(r, name, dtype, shape, quantize)
			if err != nil {
				return nil, fmt.Errorf("failed to quantize %s: %w", name, err)
			}
			layer, err := s.uploadLayer(ctx, bytes.NewReader(blobData), manifest.MediaTypeImageTensor, name)
			if err != nil {
				return nil, err
			}
			return []create.LayerInfo{layer}, nil
		}

		layer, err := s.uploadLayer(ctx, r, manifest.MediaTypeImageTensor, name)
		if err != nil {
			return nil, err
		}
		return []create.LayerInfo{layer}, nil
	}
}

func (s *remoteCreateSession) packedTensorLayerCreator(ctx context.Context) create.PackedTensorLayerCreator {
	return func(groupName string, tensors []create.PackedTensorInput) (create.LayerInfo, error) {
		hasQuantize := false
		for _, t := range tensors {
			if t.Quantize != "" {
				hasQuantize = true
				if s.fileType == "" {
					s.fileType = t.Quantize
				}
			}
		}

		var blobReader io.Reader
		if hasQuantize {
			if !s.clientQuantize {
				packedReader, cleanup, err := buildPackedReaderFromInputs(tensors)
				if err != nil {
					return create.LayerInfo{}, err
				}
				defer cleanup()

				return s.uploadLayer(ctx, packedReader, manifest.MediaTypeImageTensor, groupName)
			}

			blobData, err := QuantizePackedGroup(groupName, tensors)
			if err != nil {
				return create.LayerInfo{}, fmt.Errorf("failed to quantize packed group %s: %w", groupName, err)
			}
			blobReader = bytes.NewReader(blobData)
		} else {
			tds := make([]*safetensors.TensorData, 0, len(tensors))
			for _, t := range tensors {
				if t.TensorData == nil {
					return create.LayerInfo{}, fmt.Errorf("packed tensor %s is not file-backed", t.Name)
				}
				tds = append(tds, t.TensorData.WithName(t.Name))
			}
			blobReader = safetensors.BuildPackedSafetensorsReader(tds)
		}

		return s.uploadLayer(ctx, blobReader, manifest.MediaTypeImageTensor, groupName)
	}
}

func (s *remoteCreateSession) manifestWriter(ctx context.Context) create.ManifestWriter {
	return func(modelName string, _ create.LayerInfo, layers []create.LayerInfo) error {
		files := make(map[string]string, len(layers))
		for _, layer := range layers {
			files[layer.Name] = layer.Digest
		}

		req := &api.CreateRequest{
			Model:       modelName,
			ModelFormat: "safetensors",
			Files:       files,
			Renderer:    resolveRendererName(s.opts.Modelfile, s.rendererName),
			Parser:      resolveParserName(s.opts.Modelfile, s.parserName),
			Requires:    MinOllamaVersion,
		}
		if s.clientQuantize {
			req.ClientQuantized = s.fileType
		} else if s.opts.Quantize != "" {
			req.Quantize = s.opts.Quantize
		}

		if s.opts.Modelfile != nil {
			req.Template = s.opts.Modelfile.Template
			req.System = s.opts.Modelfile.System
			req.Parameters = s.opts.Modelfile.Parameters
			if s.opts.Modelfile.License != "" {
				req.License = s.opts.Modelfile.License
			}
		}

		createSpinner := progress.NewSpinner("creating " + s.modelType)
		s.progress.Add("create", createSpinner)
		stream := false
		req.Stream = &stream
		err := s.client.Create(ctx, req, func(resp api.ProgressResponse) error {
			if resp.Status != "" {
				createSpinner.Stop()
				createSpinner = progress.NewSpinner(resp.Status)
				s.progress.Add("create", createSpinner)
			}
			return nil
		})
		createSpinner.Stop()
		if err != nil {
			return fmt.Errorf("server create failed: %w", err)
		}

		return nil
	}
}

func buildPackedReaderFromInputs(tensors []create.PackedTensorInput) (io.Reader, func(), error) {
	var (
		tds        []*safetensors.TensorData
		extractors []*safetensors.TensorExtractor
		tmpPaths   []string
	)

	cleanup := func() {
		for _, ext := range extractors {
			ext.Close()
		}
		for _, p := range tmpPaths {
			os.Remove(p)
		}
	}

	for _, t := range tensors {
		tmp, err := os.CreateTemp("", "ollama-packed-input-*")
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		tmpPath := tmp.Name()
		tmpPaths = append(tmpPaths, tmpPath)

		_, copyErr := io.Copy(tmp, t.Reader)
		closeErr := tmp.Close()
		if copyErr != nil {
			cleanup()
			return nil, nil, fmt.Errorf("stage packed tensor %s: %w", t.Name, copyErr)
		}
		if closeErr != nil {
			cleanup()
			return nil, nil, closeErr
		}

		ext, err := safetensors.OpenForExtraction(tmpPath)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		extractors = append(extractors, ext)

		inputTensors, err := ext.ExtractAll()
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		tds = append(tds, inputTensors...)
	}

	return safetensors.BuildPackedSafetensorsReader(tds), cleanup, nil
}

func (s *remoteCreateSession) uploadLayer(ctx context.Context, r io.Reader, mediaType, name string) (create.LayerInfo, error) {
	tmp, err := os.CreateTemp("", "ollama-create-blob-*")
	if err != nil {
		return create.LayerInfo{}, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	h := sha256.New()
	size, err := io.Copy(io.MultiWriter(tmp, h), r)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return create.LayerInfo{}, fmt.Errorf("failed to stage %s: %w", name, err)
	}

	digest := fmt.Sprintf("sha256:%x", h.Sum(nil))
	total := s.totalSize.Add(size)
	s.transferBar.SetTotal(total)

	b := blob{
		name:   name,
		digest: digest,
		size:   size,
		reader: func() io.ReadCloser {
			f, err := os.Open(tmpPath)
			if err != nil {
				return nil
			}
			return f
		},
	}

	if err := uploadBlob(ctx, s.client, b, &s.transferred, s.transferBar, func() {}); err != nil {
		return create.LayerInfo{}, err
	}

	return create.LayerInfo{
		Digest:    digest,
		Size:      size,
		MediaType: mediaType,
		Name:      name,
	}, nil
}

// blob represents a single content-addressed blob to upload.
type blob struct {
	name   string               // path-style name (e.g., "model.embed_tokens.weight" or "config.json")
	digest string               // sha256:hex digest
	size   int64                // blob size in bytes
	reader func() io.ReadCloser // factory to get a fresh reader for upload
}

// uploadBlob performs a HEAD check and uploads a blob if missing, with
// exponential backoff retry on transient failures (matching push/pull patterns).
// On success, calls onSuccess to record the blob in the files map.
func uploadBlob(ctx context.Context, client *api.Client, b blob, transferred *atomic.Int64, bar *progress.Bar, onSuccess func()) error {
	var lastErr error
	for attempt := range maxUploadRetries {
		if attempt > 0 {
			sleep := backoffDuration(attempt)
			slog.Info("retrying blob upload", "blob", b.name, "attempt", attempt+1, "backoff", sleep, "error", lastErr)
			select {
			case <-time.After(sleep):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		err := uploadBlobOnce(ctx, client, b, transferred, bar)
		if err == nil {
			onSuccess()
			return nil
		}

		// Context cancellation - abort immediately, no retry.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}

		lastErr = err
	}
	return fmt.Errorf("upload failed for %s after %d attempts: %w", b.name, maxUploadRetries, lastErr)
}

// uploadBlobOnce performs a single HEAD check + upload attempt for a blob.
// Progress bytes are tracked so the caller can roll back on retry.
func uploadBlobOnce(ctx context.Context, client *api.Client, b blob, transferred *atomic.Int64, bar *progress.Bar) error {
	exists, err := client.HeadBlob(ctx, b.digest)
	if err != nil {
		return fmt.Errorf("HEAD check: %w", err)
	}

	if exists {
		bar.Set(transferred.Add(b.size))
		return nil
	}

	rc := b.reader()
	if rc == nil {
		return fmt.Errorf("open blob %s for upload", b.name)
	}
	defer rc.Close()

	var blobTransferred atomic.Int64
	pr := &progressReader{
		r: rc,
		onRead: func(n int) {
			blobTransferred.Add(int64(n))
			bar.Set(transferred.Add(int64(n)))
		},
	}

	if err := client.CreateBlob(ctx, b.digest, pr); err != nil {
		// Roll back progress for this blob so the bar stays accurate on retry.
		bar.Set(transferred.Add(-blobTransferred.Load()))
		return fmt.Errorf("upload: %w", err)
	}

	return nil
}

// progressReader wraps an io.Reader and calls onRead for each read.
type progressReader struct {
	r      io.Reader
	onRead func(n int)
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.r.Read(p)
	if n > 0 {
		pr.onRead(n)
	}
	return n, err
}
