package client

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/x/create"
)

// Six attempts produce at most 31 seconds of exponential backoff per blob.
const maxUploadRetries = 6

// CreateModelRemote uploads raw safetensors source files and asks the server to
// run the x/create import pipeline. The server performs planning, transforms,
// and MLX quantization against its own hardware.
func CreateModelRemote(ctx context.Context, client *api.Client, opts CreateOptions, p *progress.Progress) error {
	if opts.Force {
		return errors.New("--force is only supported for local MLX safetensors imports")
	}
	if opts.Modelfile != nil && len(opts.Modelfile.Adapters) > 0 {
		return errSafetensorsAdapters
	}
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	hasDraft := opts.Modelfile != nil && opts.Modelfile.Draft != ""
	if err := validateSafetensorsQuantization(opts); err != nil {
		return err
	}
	if !isSafetensors {
		return fmt.Errorf("%s is not a supported safetensors model directory (needs config.json + *.safetensors)", opts.ModelDir)
	}
	if hasDraft && !create.IsSafetensorsModelDir(opts.Modelfile.Draft) {
		return fmt.Errorf("draft %s is not a supported safetensors model directory", opts.Modelfile.Draft)
	}
	if err := validateDistinctSafetensorsSources(opts.ModelDir, opts.Modelfile); err != nil {
		return err
	}

	files, err := prepareRemoteSourceFiles(ctx, opts.ModelDir, false)
	if err != nil {
		return err
	}
	if hasDraft {
		draftFiles, err := prepareRemoteSourceFiles(ctx, opts.Modelfile.Draft, true)
		if err != nil {
			return err
		}
		files = append(files, draftFiles...)
	}

	if err := uploadRemoteSourceFiles(ctx, client, files, p); err != nil {
		return err
	}

	req := newRemoteCreateRequest(opts, files)
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

func prepareRemoteSourceFiles(ctx context.Context, dir string, draft bool) ([]remoteSourceFile, error) {
	weightFiles, err := create.SafetensorsWeightFiles(dir)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	names := append([]string(nil), weightFiles...)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if filepath.Ext(name) == ".json" || name == "chat_template.jinja" {
			names = append(names, name)
		}
	}
	slices.Sort(names)

	files := make([]remoteSourceFile, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		abs := filepath.Join(dir, name)
		digest, size, err := digestFile(ctx, abs)
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

func digestFile(ctx context.Context, name string) (string, int64, error) {
	f, err := os.Open(name)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	h := sha256.New()
	n, err := io.Copy(h, readerWithContext(ctx, f))
	if err != nil {
		return "", 0, err
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n, nil
}

type readFunc func([]byte) (int, error)

func (f readFunc) Read(p []byte) (int, error) { return f(p) }

func readerWithContext(ctx context.Context, r io.Reader) io.Reader {
	return readFunc(func(p []byte) (int, error) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		return r.Read(p)
	})
}

func uploadRemoteSourceFiles(ctx context.Context, client *api.Client, files []remoteSourceFile, p *progress.Progress) error {
	var total int64
	uploads := make([]remoteSourceFile, 0, len(files))
	digests := make(map[string]struct{}, len(files))
	for _, f := range files {
		if _, ok := digests[f.digest]; ok {
			continue
		}
		digests[f.digest] = struct{}{}
		uploads = append(uploads, f)
		total += f.size
	}

	var bar *progress.Bar
	if p != nil {
		bar = progress.NewBar("transferring model", total, 0)
		p.Add("transfer", bar)
	}

	var transferred atomic.Int64
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(max(1, int(envconfig.MaxTransferStreams())))
	for _, f := range uploads {
		g.Go(func() error {
			return uploadRemoteSourceFile(ctx, client, f, &transferred, bar)
		})
	}
	return g.Wait()
}

func uploadRemoteSourceFile(ctx context.Context, client *api.Client, f remoteSourceFile, transferred *atomic.Int64, bar *progress.Bar) error {
	var lastErr error
	for attempt := range maxUploadRetries {
		if attempt > 0 {
			sleep := time.Second << (attempt - 1)
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
		if !shouldRetryUpload(err) {
			return err
		}
		lastErr = err
	}
	return fmt.Errorf("upload failed for %s after %d attempts: %w", f.logical, maxUploadRetries, lastErr)
}

func shouldRetryUpload(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var authorizationError api.AuthorizationError
	if errors.As(err, &authorizationError) {
		return false
	}
	var statusError api.StatusError
	if errors.As(err, &statusError) {
		return statusError.StatusCode == http.StatusRequestTimeout ||
			statusError.StatusCode == http.StatusTooManyRequests ||
			statusError.StatusCode >= http.StatusInternalServerError
	}
	var pathError *os.PathError
	return !errors.As(err, &pathError)
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

func newRemoteCreateRequest(opts CreateOptions, files []remoteSourceFile) *api.CreateRequest {
	req := &api.CreateRequest{
		Model:         opts.ModelName,
		Files:         make(map[string]string),
		Quantize:      opts.Quantize,
		DraftQuantize: opts.DraftQuantize,
		Requires:      create.SafetensorsMinOllamaVersion,
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
		req.Renderer = opts.Modelfile.Renderer
		req.Parser = opts.Modelfile.Parser
		req.Template = opts.Modelfile.Template
		req.System = opts.Modelfile.System
		req.Parameters = opts.Modelfile.Parameters
		req.Messages = opts.Modelfile.Messages
		if len(opts.Modelfile.Licenses) > 0 {
			req.License = opts.Modelfile.Licenses
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
