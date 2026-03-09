package client

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
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

// CreateModelRemote creates a safetensors model on a remote server by uploading
// tensor blobs and sending a CreateRequest with model_format="safetensors".
//
// It mimics the push flow: for each tensor, hash it to get the digest, HEAD
// check whether the blob exists on the server, and upload if missing. Blobs
// flow through a bounded channel so memory stays proportional to the upload
// concurrency, not the total model size.
func CreateModelRemote(ctx context.Context, client *api.Client, opts RemoteCreateOptions, p *progress.Progress) error {
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)

	if !isSafetensors && !isImageGen {
		return fmt.Errorf("%s is not a supported model directory", opts.ModelDir)
	}

	var modelType string
	var capabilities []string
	var parserName, rendererName string
	if isSafetensors {
		modelType = "safetensors model"
		capabilities = []string{"completion"}
		if supportsThinking(opts.ModelDir) {
			capabilities = append(capabilities, "thinking")
		}
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
	} else {
		modelType = "image generation model"
		capabilities = []string{"image"}
	}

	// Check for Flux2KleinPipeline vision capability
	modelIndex := filepath.Join(opts.ModelDir, "model_index.json")
	if data, err := os.ReadFile(modelIndex); err == nil {
		var cfg struct {
			ClassName string `json:"_class_name"`
		}
		if json.Unmarshal(data, &cfg) == nil && cfg.ClassName == "Flux2KleinPipeline" {
			capabilities = append(capabilities, "vision")
		}
	}

	// Pipeline: producer hashes blobs → consumers HEAD check + upload.
	// Channel cap bounds memory: at most 8 blobs in flight between
	// hashing and uploading.
	blobCh := make(chan blob, 8)

	// Collect name→digest results for the CreateRequest.
	var (
		files   = make(map[string]string)
		filesMu sync.Mutex
	)

	// Upload errgroup — cancels gctx on first error, which also stops
	// the producer via the shared context.
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(17) // 16 upload workers + 1 slot for the range loop to enqueue

	// Progress: show bytes transferred.
	var transferred atomic.Int64
	bar := progress.NewBar("transferring model", 0, 0)
	p.Add("transfer", bar)

	// Producer goroutine: hash blobs and send to channel.
	var produceErr error
	go func() {
		defer close(blobCh)
		produceErr = produceBlobs(gctx, opts, blobCh)
	}()

	// Consumer: drain channel, HEAD check, upload if missing.
	// Retries with exponential backoff on transient failures (matching push/pull).
	var totalSize atomic.Int64
	for b := range blobCh {
		totalSize.Add(b.size)
		bar.SetTotal(totalSize.Load())
		b := b
		g.Go(func() error {
			return uploadBlob(gctx, client, b, &transferred, bar, func() {
				filesMu.Lock()
				files[b.name] = b.digest
				filesMu.Unlock()
			})
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}
	if produceErr != nil {
		return produceErr
	}

	bar.Set(totalSize.Load())

	// Build the CreateRequest. Quantization is always done server-side —
	// the client uploads unquantized tensors and the server quantizes them.
	req := &api.CreateRequest{
		Model:        opts.ModelName,
		ModelFormat:  "safetensors",
		Files:        files,
		Capabilities: capabilities,
		Renderer:     resolveRendererName(opts.Modelfile, rendererName),
		Parser:       resolveParserName(opts.Modelfile, parserName),
		Requires:     MinOllamaVersion,
		Quantize:     opts.Quantize,
	}

	if opts.Modelfile != nil {
		req.Template = opts.Modelfile.Template
		req.System = opts.Modelfile.System
		if opts.Modelfile.License != "" {
			req.License = opts.Modelfile.License
		}
	}

	// Send the create request
	createSpinner := progress.NewSpinner("creating " + modelType)
	p.Add("create", createSpinner)

	stream := false
	req.Stream = &stream
	err := client.Create(ctx, req, func(resp api.ProgressResponse) error {
		if resp.Status != "" {
			createSpinner.Stop()
			createSpinner = progress.NewSpinner(resp.Status)
			p.Add("create", createSpinner)
		}
		return nil
	})
	createSpinner.Stop()
	if err != nil {
		return fmt.Errorf("server create failed: %w", err)
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

// blob represents a single content-addressed blob to upload.
type blob struct {
	name   string               // path-style name (e.g., "model.embed_tokens.weight" or "config.json")
	digest string               // sha256:hex digest
	size   int64                // blob size in bytes
	reader func() io.ReadCloser // factory to get a fresh reader for upload
}

// produceBlobs walks the model directory, hashes each blob, and sends it to ch.
// It returns when all blobs have been sent or ctx is cancelled.
func produceBlobs(ctx context.Context, opts RemoteCreateOptions, ch chan<- blob) error {
	entries, err := os.ReadDir(opts.ModelDir)
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)

	if isSafetensors {
		if err := produceSafetensorsBlobs(ctx, opts, entries, ch); err != nil {
			return err
		}
	} else {
		if err := produceImageGenBlobs(ctx, opts, entries, ch); err != nil {
			return err
		}
	}

	// JSON config files
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if entry.Name() == "model.safetensors.index.json" {
			continue
		}

		fullPath := filepath.Join(opts.ModelDir, entry.Name())
		digest, size, err := hashFile(fullPath)
		if err != nil {
			return fmt.Errorf("failed to hash %s: %w", entry.Name(), err)
		}

		path := fullPath
		if err := sendBlob(ctx, ch, blob{
			name:   entry.Name(),
			digest: digest,
			size:   size,
			reader: func() io.ReadCloser {
				f, _ := os.Open(path)
				return f
			},
		}); err != nil {
			return err
		}
	}

	return nil
}

// produceSafetensorsBlobs extracts individual tensors from safetensors files,
// hashes each one, and sends it to ch. Files are processed in parallel to
// saturate disk I/O and use multiple CPU cores for SHA256 hashing.
// Expert tensors are buffered until all files are processed, then packed and sent.
func produceSafetensorsBlobs(ctx context.Context, opts RemoteCreateOptions, entries []os.DirEntry, ch chan<- blob) error {
	// Collect safetensors file paths
	var stFiles []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}
		stFiles = append(stFiles, filepath.Join(opts.ModelDir, entry.Name()))
	}

	// Expert group collection (populated in sequential phase 1)
	expertGroups := make(map[string][]pendingTensor)

	// Phase 1: Process each file, send non-expert blobs, collect expert tensors.
	for _, stPath := range stFiles {
		extractor, err := safetensors.OpenForExtraction(stPath)
		if err != nil {
			return fmt.Errorf("failed to open %s: %w", stPath, err)
		}

		tensorNames := extractor.ListTensors()
		for _, tensorName := range tensorNames {
			td, err := extractor.GetTensor(tensorName)
			if err != nil {
				extractor.Close()
				return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
			}

			groupPrefix := create.ExpertGroupPrefix(tensorName)
			if groupPrefix != "" {
				expertGroups[groupPrefix] = append(expertGroups[groupPrefix], pendingTensor{
					name:   tensorName,
					dtype:  td.Dtype,
					shape:  td.Shape,
					stPath: stPath,
				})
				continue
			}

			b, err := safetensorsBlobFromTensor(td, stPath)
			if err != nil {
				extractor.Close()
				return err
			}

			if err := sendBlob(ctx, ch, b); err != nil {
				extractor.Close()
				return err
			}
		}
		extractor.Close()
	}

	// Phase 2: Pack and hash expert groups in parallel — each group is
	// independent and can be hashed on a separate core.
	groupNames := make([]string, 0, len(expertGroups))
	for name := range expertGroups {
		groupNames = append(groupNames, name)
	}
	sort.Strings(groupNames)

	// Use a goroutine pool to hash expert groups, feeding results into
	// an intermediate channel that we forward to the main blob channel.
	expertCh := make(chan blob, 4)
	var expertErr error
	go func() {
		defer close(expertCh)
		eg, egctx := errgroup.WithContext(ctx)
		eg.SetLimit(4)
		for _, groupName := range groupNames {
			groupName := groupName
			tensors := expertGroups[groupName]
			eg.Go(func() error {
				b, err := packedExpertBlob(groupName, tensors)
				if err != nil {
					return err
				}
				select {
				case expertCh <- b:
					return nil
				case <-egctx.Done():
					return egctx.Err()
				}
			})
		}
		expertErr = eg.Wait()
	}()

	for b := range expertCh {
		if err := sendBlob(ctx, ch, b); err != nil {
			return err
		}
	}
	if expertErr != nil {
		return expertErr
	}

	return nil
}

type pendingTensor struct {
	name   string
	dtype  string
	shape  []int32
	stPath string // path to source safetensors file
}

// produceImageGenBlobs handles image generation models.
// Each safetensors file in subdirectories is uploaded as a whole blob with
// component-scoped names (e.g., "text_encoder/model.safetensors").
func produceImageGenBlobs(ctx context.Context, opts RemoteCreateOptions, entries []os.DirEntry, ch chan<- blob) error {
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		componentDir := filepath.Join(opts.ModelDir, entry.Name())
		componentEntries, err := os.ReadDir(componentDir)
		if err != nil {
			continue
		}

		for _, ce := range componentEntries {
			ext := filepath.Ext(ce.Name())
			if ce.IsDir() || (ext != ".safetensors" && ext != ".json") {
				continue
			}

			fullPath := filepath.Join(componentDir, ce.Name())
			name := entry.Name() + "/" + ce.Name()

			digest, size, err := hashFile(fullPath)
			if err != nil {
				return fmt.Errorf("failed to hash %s: %w", name, err)
			}

			path := fullPath
			if err := sendBlob(ctx, ch, blob{
				name:   name,
				digest: digest,
				size:   size,
				reader: func() io.ReadCloser {
					f, _ := os.Open(path)
					return f
				},
			}); err != nil {
				return err
			}
		}
	}

	return nil
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

		// Context cancellation — abort immediately, no retry.
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

// sendBlob sends a blob to the channel, respecting context cancellation.
func sendBlob(ctx context.Context, ch chan<- blob, b blob) error {
	select {
	case ch <- b:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// safetensorsBlobFromTensor hashes a safetensors-wrapped tensor and returns a
// lightweight blob descriptor. No tensor data is held in memory — the reader
// factory re-extracts from the source file on demand.
func safetensorsBlobFromTensor(td *safetensors.TensorData, stPath string) (blob, error) {
	r := td.SafetensorsReader()
	h := sha256.New()
	size, err := io.Copy(h, r)
	if err != nil {
		return blob{}, fmt.Errorf("failed to hash tensor %s: %w", td.Name, err)
	}
	digest := fmt.Sprintf("sha256:%x", h.Sum(nil))

	name := td.Name
	srcPath := stPath
	tensorName := td.Name
	return blob{
		name:   name,
		digest: digest,
		size:   size,
		reader: func() io.ReadCloser {
			ext, err := safetensors.OpenForExtraction(srcPath)
			if err != nil {
				return io.NopCloser(strings.NewReader(""))
			}
			td2, err := ext.GetTensor(tensorName)
			if err != nil {
				ext.Close()
				return io.NopCloser(strings.NewReader(""))
			}
			r := td2.SafetensorsReader()
			return &extractorReadCloser{Reader: r, ext: ext}
		},
	}, nil
}

// packedExpertBlob creates a packed blob from a group of expert tensors.
func packedExpertBlob(groupName string, tensors []pendingTensor) (blob, error) {
	// Open all extractors and get TensorData backed by file SectionReaders.
	// Stream through BuildPackedSafetensorsReader for hashing — no tensor
	// data is copied into memory.
	tds, extractors, err := openPackedTensors(tensors)
	if err != nil {
		return blob{}, err
	}

	packedReader := safetensors.BuildPackedSafetensorsReader(tds)
	h := sha256.New()
	size, err := io.Copy(h, packedReader)
	closeAll(extractors)
	if err != nil {
		return blob{}, fmt.Errorf("failed to hash packed blob for %s: %w", groupName, err)
	}

	digest := fmt.Sprintf("sha256:%x", h.Sum(nil))

	// Capture tensor metadata for re-read at upload time.
	tensorSpecs := make([]pendingTensor, len(tensors))
	copy(tensorSpecs, tensors)

	return blob{
		name:   groupName,
		digest: digest,
		size:   size,
		reader: func() io.ReadCloser {
			tds2, exts2, err := openPackedTensors(tensorSpecs)
			if err != nil {
				return io.NopCloser(strings.NewReader(""))
			}
			r := safetensors.BuildPackedSafetensorsReader(tds2)
			return &multiExtractorReadCloser{Reader: r, extractors: exts2}
		},
	}, nil
}

// openPackedTensors opens extractors and returns TensorData for a group of
// tensors. The caller must close all extractors when done.
func openPackedTensors(tensors []pendingTensor) ([]*safetensors.TensorData, []*safetensors.TensorExtractor, error) {
	var tds []*safetensors.TensorData
	var extractors []*safetensors.TensorExtractor

	for _, t := range tensors {
		ext, err := safetensors.OpenForExtraction(t.stPath)
		if err != nil {
			closeAll(extractors)
			return nil, nil, err
		}
		extractors = append(extractors, ext)

		td, err := ext.GetTensor(t.name)
		if err != nil {
			closeAll(extractors)
			return nil, nil, err
		}
		tds = append(tds, td)
	}

	return tds, extractors, nil
}

// hashFile computes the sha256 digest of a file.
func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n, nil
}

func closeAll(extractors []*safetensors.TensorExtractor) {
	for _, e := range extractors {
		e.Close()
	}
}

// extractorReadCloser wraps a reader and closes the extractor when done.
type extractorReadCloser struct {
	io.Reader
	ext *safetensors.TensorExtractor
}

func (r *extractorReadCloser) Close() error {
	r.ext.Close()
	return nil
}

// multiExtractorReadCloser wraps a reader and closes multiple extractors when done.
type multiExtractorReadCloser struct {
	io.Reader
	extractors []*safetensors.TensorExtractor
}

func (r *multiExtractorReadCloser) Close() error {
	closeAll(r.extractors)
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
