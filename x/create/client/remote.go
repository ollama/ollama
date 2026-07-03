package client

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/quant"
)

const maxUploadRetries = 6

var backoffDuration = func(attempt int) time.Duration {
	return time.Duration(math.Pow(2, float64(attempt-1))) * time.Second
}

// CreateModelRemote uploads raw safetensors source files and asks the server to
// run the x/create import pipeline. The server performs planning, transforms,
// and MLX quantization against its own hardware.
func CreateModelRemote(ctx context.Context, client *api.Client, opts CreateOptions, p *progress.Progress) error {
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	if opts.DraftQuantize != "" && !hasDraft {
		return fmt.Errorf("--draft-quantize requires a DRAFT model")
	}
	if opts.Quantize != "" && quant.Canonical(opts.Quantize) == "" {
		return fmt.Errorf("unsupported --quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.Quantize)
	}
	if opts.DraftQuantize != "" && quant.Canonical(opts.DraftQuantize) == "" {
		return fmt.Errorf("unsupported --draft-quantize %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", opts.DraftQuantize)
	}
	if !isSafetensors {
		return fmt.Errorf("%s is not a supported safetensors model directory (needs config.json + *.safetensors)", opts.ModelDir)
	}
	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}

	parserName := getParserName(opts.ModelDir)
	rendererName := getRendererName(opts.ModelDir)
	capabilities := inferSafetensorsCapabilities(opts.ModelDir, resolveParserName(opts.Modelfile, parserName))

	files, err := prepareRemoteSourceFiles(opts.ModelDir, false)
	if err != nil {
		return err
	}
	if hasDraft {
		draftFiles, err := prepareRemoteSourceFiles(opts.Modelfile.Draft, true)
		if err != nil {
			return err
		}
		files = append(files, draftFiles...)
	}

	if err := uploadRemoteSourceFiles(ctx, client, files, p); err != nil {
		return err
	}

	req := newRemoteCreateRequest(opts, files, capabilities, parserName, rendererName)
	if err := runRemoteCreateRequest(ctx, client, req, p); err != nil {
		return err
	}

	fmt.Printf("Created safetensors model '%s'\n", opts.ModelName)
	return nil
}

type remoteSourceFile struct {
	logical string
	path    string
	digest  string
	size    int64
	draft   bool
}

func prepareRemoteSourceFiles(dir string, draft bool) ([]remoteSourceFile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var files []remoteSourceFile
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") && !(strings.HasSuffix(name, ".safetensors") && strings.HasPrefix(name, "model")) {
			continue
		}

		abs := filepath.Join(dir, name)
		digest, size, err := digestFile(abs)
		if err != nil {
			return nil, err
		}
		files = append(files, remoteSourceFile{
			logical: path.Clean(filepath.ToSlash(name)),
			path:    abs,
			digest:  digest,
			size:    size,
			draft:   draft,
		})
	}
	return files, nil
}

func digestFile(name string) (string, int64, error) {
	f, err := os.Open(name)
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

func uploadRemoteSourceFiles(ctx context.Context, client *api.Client, files []remoteSourceFile, p *progress.Progress) error {
	var total int64
	for _, f := range files {
		total += f.size
	}

	var bar *progress.Bar
	if p != nil {
		bar = progress.NewBar("transferring model", total, 0)
		p.Add("transfer", bar)
	}

	var transferred atomic.Int64
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(remoteUploadConcurrency())
	for _, f := range files {
		g.Go(func() error {
			return uploadRemoteSourceFile(ctx, client, f, &transferred, bar)
		})
	}
	return g.Wait()
}

var remoteUploadConcurrency = func() int {
	n := envconfig.MaxTransferStreams()
	if n == 0 {
		return 1
	}
	return int(n)
}

func uploadRemoteSourceFile(ctx context.Context, client *api.Client, f remoteSourceFile, transferred *atomic.Int64, bar *progress.Bar) error {
	var lastErr error
	for attempt := range maxUploadRetries {
		if attempt > 0 {
			sleep := backoffDuration(attempt)
			slog.Info("retrying blob upload", "blob", f.logical, "attempt", attempt+1, "backoff", sleep, "error", lastErr)
			select {
			case <-time.After(sleep):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		err := uploadRemoteSourceFileOnce(ctx, client, f, transferred, bar)
		if err == nil {
			return nil
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		lastErr = err
	}
	return fmt.Errorf("upload failed for %s after %d attempts: %w", f.logical, maxUploadRetries, lastErr)
}

func uploadRemoteSourceFileOnce(ctx context.Context, client *api.Client, f remoteSourceFile, transferred *atomic.Int64, bar *progress.Bar) error {
	exists, err := client.HeadBlob(ctx, f.digest)
	if err != nil {
		return fmt.Errorf("HEAD check %s: %w", f.logical, err)
	}
	if exists {
		if bar != nil {
			bar.Set(transferred.Add(f.size))
		}
		return nil
	}

	rc, err := os.Open(f.path)
	if err != nil {
		return err
	}
	defer rc.Close()

	var blobTransferred atomic.Int64
	pr := &progressReader{
		r: rc,
		onRead: func(n int) {
			blobTransferred.Add(int64(n))
			if bar != nil {
				bar.Set(transferred.Add(int64(n)))
			}
		},
	}
	if err := client.CreateBlob(ctx, f.digest, pr); err != nil {
		if bar != nil {
			bar.Set(transferred.Add(-blobTransferred.Load()))
		}
		return fmt.Errorf("upload %s: %w", f.logical, err)
	}
	return nil
}

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

func newRemoteCreateRequest(opts CreateOptions, files []remoteSourceFile, capabilities []string, parserName, rendererName string) *api.CreateRequest {
	req := &api.CreateRequest{
		Model:         opts.ModelName,
		Files:         make(map[string]string),
		Quantize:      opts.Quantize,
		DraftQuantize: opts.DraftQuantize,
		Renderer:      resolveRendererName(opts.Modelfile, rendererName),
		Parser:        resolveParserName(opts.Modelfile, parserName),
		Requires:      MinOllamaVersion,
		Info: map[string]any{
			"capabilities": capabilities,
		},
	}
	for _, f := range files {
		if f.draft {
			if req.DraftFiles == nil {
				req.DraftFiles = make(map[string]string)
			}
			req.DraftFiles[f.logical] = f.digest
		} else {
			req.Files[f.logical] = f.digest
		}
	}
	if opts.Modelfile != nil {
		req.Template = opts.Modelfile.Template
		req.System = opts.Modelfile.System
		req.Parameters = opts.Modelfile.Parameters
		if opts.Modelfile.License != "" {
			req.License = opts.Modelfile.License
		}
		if opts.Modelfile.Requires != "" {
			req.Requires = opts.Modelfile.Requires
		}
	}
	return req
}

func runRemoteCreateRequest(ctx context.Context, client *api.Client, req *api.CreateRequest, p *progress.Progress) error {
	status := "creating safetensors model"
	var spinner *progress.Spinner
	if p != nil {
		spinner = progress.NewSpinner(status)
		p.Add("create", spinner)
	}
	err := client.Create(ctx, req, func(resp api.ProgressResponse) error {
		if resp.Status == "" || resp.Status == status || spinner == nil || p == nil {
			return nil
		}
		spinner.Stop()
		status = resp.Status
		spinner = progress.NewSpinner(status)
		p.Add("create", spinner)
		return nil
	})
	if spinner != nil {
		spinner.Stop()
	}
	if err != nil {
		return fmt.Errorf("server create failed: %w", err)
	}
	return nil
}
