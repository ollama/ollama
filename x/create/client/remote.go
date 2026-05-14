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
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/safetensors"
	"golang.org/x/sync/errgroup"
)

const maxUploadRetries = 6

// backoffDuration calculates the backoff sleep for a retry attempt.
// Overridable in tests to avoid real sleeps.
var backoffDuration = func(attempt int) time.Duration {
	return time.Duration(math.Pow(2, float64(attempt-1))) * time.Second
}

// RemoteCreateOptions holds options for remote safetensors model creation.
type RemoteCreateOptions struct {
	ModelName     string
	ModelDir      string
	Quantize      string
	DraftQuantize string
	Modelfile     *ModelfileConfig
}

// CreateModelRemote creates a model on a remote server by uploading blobs and
// sending a create request. It reuses the same x/create traversal as local
// creation; only the layer/manifest callbacks differ.
func CreateModelRemote(ctx context.Context, client *api.Client, opts RemoteCreateOptions, p *progress.Progress) error {
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	isBaseModelWithDraft := hasDraft && !isSafetensors && create.IsSafetensorsLLMModel(opts.ModelDir)

	if !isSafetensors && !isImageGen && !isBaseModelWithDraft {
		return fmt.Errorf("%s is not a supported model directory", opts.ModelDir)
	}
	if opts.DraftQuantize != "" && !hasDraft {
		return fmt.Errorf("--draft-quantize requires a DRAFT model")
	}
	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}
	if hasDraft && isImageGen {
		return fmt.Errorf("draft models are only supported for safetensors LLM models")
	}

	var modelType, spinnerKey string
	var parserName, rendererName string
	if isSafetensors {
		modelType = "safetensors model"
		spinnerKey = "create"
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
	} else if isBaseModelWithDraft {
		modelType = "safetensors model"
		spinnerKey = "create"
	} else {
		modelType = "image generation model"
		spinnerKey = "imagegen"
	}

	session := newRemoteCreateSession(ctx, client, p, opts, parserName, rendererName, modelType)

	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)
	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	var draftLayers []create.LayerInfo
	if hasDraft {
		var err error
		draftLayers, err = create.CreateDraftSafetensorsLayers(
			opts.Modelfile.Draft,
			"draft.",
			"draft",
			opts.DraftQuantize,
			session.layerCreator(ctx), session.draftTensorLayerCreator(ctx),
			progressFn,
		)
		if err != nil {
			spinner.Stop()
			session.cancel()
			_ = session.waitUploads()
			return err
		}
	}

	var err error
	if isBaseModelWithDraft {
		err = session.createSafetensorsModelFromBase(ctx, opts.ModelName, opts.ModelDir, draftLayers)
	} else if isSafetensors {
		writer := session.manifestWriter(ctx)
		if len(draftLayers) > 0 {
			writer = appendLayersManifestWriter(writer, draftLayers)
		}
		err = create.CreateSafetensorsModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			session.layerCreator(ctx), session.tensorLayerCreator(ctx),
			writer,
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
		session.cancel()
		_ = session.waitUploads()
		return err
	}
	session.cancel()

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
	cancel         context.CancelFunc
	uploadDone     <-chan struct{}
	uploads        *errgroup.Group
	uploadBlobFn   func(blob) error
	uploadSlots    chan struct{}
	waitUploadsErr error
	waitUploadsMu  sync.Mutex
	waitUploadsRan bool
}

func newRemoteCreateSession(ctx context.Context, client *api.Client, p *progress.Progress, opts RemoteCreateOptions, parserName, rendererName, modelType string) *remoteCreateSession {
	baseCtx, cancel := context.WithCancel(ctx)
	uploads, uploadCtx := errgroup.WithContext(baseCtx)
	bar := progress.NewBar("transferring model", 0, 0)
	p.Add("transfer", bar)
	concurrency := remoteUploadConcurrency()
	s := &remoteCreateSession{
		client:         client,
		progress:       p,
		opts:           opts,
		parserName:     parserName,
		rendererName:   rendererName,
		modelType:      modelType,
		transferBar:    bar,
		fileType:       strings.ToLower(strings.TrimSpace(opts.Quantize)),
		cancel:         cancel,
		uploadDone:     uploadCtx.Done(),
		uploads:        uploads,
		uploadSlots:    make(chan struct{}, concurrency),
		clientQuantize: remoteCreateCanClientQuantize(opts.Quantize),
	}
	s.uploadBlobFn = func(b blob) error {
		return uploadBlob(uploadCtx, client, b, &s.transferred, bar, func() {})
	}
	return s
}

var (
	remoteQuantizeSupported   = QuantizeSupported
	remoteQuantizeTensor      = QuantizeTensor
	remoteQuantizePackedGroup = QuantizePackedGroup
)

var forceServerQuantize = func() bool {
	return boolEnv("OLLAMA_CREATE_SERVER_QUANTIZE")
}

func remoteCreateCanClientQuantize(quantize string) bool {
	return quantize == "" || (!forceServerQuantize() && remoteQuantizeSupported())
}

var remoteUploadConcurrency = func() int {
	n := intEnv("OLLAMA_MAX_TRANSFERS", 4)
	if n < 1 {
		return 1
	}
	return n
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

func intEnv(name string, defaultValue int) int {
	if s := strings.TrimSpace(os.Getenv(name)); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			return defaultValue
		}
		return n
	}
	return defaultValue
}

func (s *remoteCreateSession) waitUploads() error {
	s.waitUploadsMu.Lock()
	if s.waitUploadsRan {
		err := s.waitUploadsErr
		s.waitUploadsMu.Unlock()
		return err
	}
	s.waitUploadsRan = true
	s.waitUploadsMu.Unlock()

	err := s.uploads.Wait()

	s.waitUploadsMu.Lock()
	s.waitUploadsErr = err
	s.waitUploadsMu.Unlock()
	return err
}

func (s *remoteCreateSession) layerCreator(ctx context.Context) create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		return s.uploadLayer(ctx, r, mediaType, name)
	}
}

func (s *remoteCreateSession) tensorLayerCreator(ctx context.Context) create.QuantizingTensorLayerCreator {
	return s.quantizingTensorLayerCreator(ctx, true)
}

func (s *remoteCreateSession) draftTensorLayerCreator(ctx context.Context) create.QuantizingTensorLayerCreator {
	return s.quantizingTensorLayerCreator(ctx, false)
}

func (s *remoteCreateSession) quantizingTensorLayerCreator(ctx context.Context, allowServerQuantize bool) create.QuantizingTensorLayerCreator {
	return func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
		if quantize == "" {
			layer, err := s.uploadLayer(ctx, r, manifest.MediaTypeImageTensor, name)
			if err != nil {
				return nil, err
			}
			return []create.LayerInfo{layer}, nil
		}

		if allowServerQuantize && !s.clientQuantize {
			layer, err := s.uploadLayer(ctx, r, manifest.MediaTypeImageTensor, name)
			if err != nil {
				return nil, err
			}
			return []create.LayerInfo{layer}, nil
		}

		if allowServerQuantize && s.fileType == "" {
			s.fileType = quantize
		}

		blobData, err := remoteQuantizeTensor(r, name, dtype, shape, quantize)
		if err != nil {
			return nil, fmt.Errorf("failed to quantize %s: %w", name, err)
		}
		layer, err := s.uploadLayer(ctx, bytes.NewReader(blobData), manifest.MediaTypeImageTensor, name)
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

			blobData, err := remoteQuantizePackedGroup(groupName, tensors)
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
		if err := s.waitUploads(); err != nil {
			return fmt.Errorf("upload blobs: %w", err)
		}

		req := s.newCreateRequest(modelName, "", layers, resolveParserName(s.opts.Modelfile, s.parserName), resolveRendererName(s.opts.Modelfile, s.rendererName))
		if s.clientQuantize {
			req.ClientQuantized = s.fileType
		} else if s.opts.Quantize != "" {
			req.Quantize = s.opts.Quantize
		}

		if err := s.runCreateRequest(ctx, req); err != nil {
			return fmt.Errorf("server create failed: %w", err)
		}
		return nil
	}
}

// createSafetensorsModelFromBase is the remote/API counterpart to the
// local base-overlay flow in create.go. Keep it aligned with
// server/create.go:createSafetensorsModel: both paths preserve the stored base
// model and only upload/append draft overlay layers.
func (s *remoteCreateSession) createSafetensorsModelFromBase(ctx context.Context, modelName, baseModel string, layers []create.LayerInfo) error {
	baseConfig, err := create.LoadStoredSafetensorsLLMBaseModel(baseModel)
	if err != nil {
		return err
	}
	if _, err := create.ResolveBaseModelQuantization(baseConfig.Config, s.opts.Quantize); err != nil {
		return err
	}
	if err := s.waitUploads(); err != nil {
		return fmt.Errorf("upload blobs: %w", err)
	}

	req := s.newCreateRequest(modelName, baseModel, layers, resolveParserName(s.opts.Modelfile, baseConfig.Config.Parser), resolveRendererName(s.opts.Modelfile, baseConfig.Config.Renderer))
	if err := s.runCreateRequest(ctx, req); err != nil {
		return fmt.Errorf("server create failed: %w", err)
	}
	return nil
}

func (s *remoteCreateSession) newCreateRequest(modelName, from string, layers []create.LayerInfo, parserName, rendererName string) *api.CreateRequest {
	files := make(map[string]string, len(layers))
	for _, layer := range layers {
		files[layer.Name] = layer.Digest
	}

	req := &api.CreateRequest{
		Model:       modelName,
		ModelFormat: "safetensors",
		From:        from,
		Files:       files,
		Renderer:    rendererName,
		Parser:      parserName,
		Requires:    MinOllamaVersion,
	}
	if s.opts.Modelfile != nil {
		req.Template = s.opts.Modelfile.Template
		req.System = s.opts.Modelfile.System
		req.Parameters = s.opts.Modelfile.Parameters
		if s.opts.Modelfile.License != "" {
			req.License = s.opts.Modelfile.License
		}
	}
	return req
}

func (s *remoteCreateSession) runCreateRequest(ctx context.Context, req *api.CreateRequest) error {
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
	return err
}

func buildPackedReaderFromInputs(tensors []create.PackedTensorInput) (io.Reader, func(), error) {
	allFileBacked := true
	tds := make([]*safetensors.TensorData, 0, len(tensors))
	for _, t := range tensors {
		if t.TensorData == nil {
			allFileBacked = false
			break
		}
		tds = append(tds, t.TensorData.WithName(t.Name))
	}
	if allFileBacked {
		return safetensors.BuildPackedSafetensorsReader(tds), func() {}, nil
	}

	var (
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
	select {
	case <-s.uploadDone:
		return create.LayerInfo{}, context.Canceled
	case s.uploadSlots <- struct{}{}:
	}

	tmp, err := os.CreateTemp("", "ollama-create-blob-*")
	if err != nil {
		<-s.uploadSlots
		return create.LayerInfo{}, err
	}
	tmpPath := tmp.Name()

	h := sha256.New()
	size, err := io.Copy(io.MultiWriter(tmp, h), r)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		<-s.uploadSlots
		os.Remove(tmpPath)
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

	s.uploads.Go(func() error {
		defer func() {
			os.Remove(tmpPath)
			<-s.uploadSlots
		}()
		return s.uploadBlobFn(b)
	})

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
