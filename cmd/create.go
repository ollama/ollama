package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

func expandPath(path, dir string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}

	path, found := strings.CutPrefix(path, "~")
	if !found {
		// make path relative to dir
		if !filepath.IsAbs(dir) {
			// if dir is relative, make it absolute relative to cwd
			cwd, err := os.Getwd()
			if err != nil {
				return "", err
			}
			dir = filepath.Join(cwd, dir)
		}
		path = filepath.Join(dir, path)
	} else if filepath.IsLocal(path) {
		// ~<user>/...
		// make path relative to specified user's home
		split := strings.SplitN(path, "/", 2)
		u, err := user.Lookup(split[0])
		if err != nil {
			return "", err
		}
		split[0] = u.HomeDir
		path = filepath.Join(split...)
	} else {
		// ~ or ~/...
		// make path relative to current user's home
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, path)
	}

	return filepath.Clean(path), nil
}

func detectContentType(fsys fs.FS, path string) (string, error) {
	f, err := fsys.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	b := make([]byte, 512)
	if _, err := f.Read(b); err != nil && err != io.EOF {
		return "", err
	}

	contentType, _, _ := strings.Cut(http.DetectContentType(b), ";")
	return contentType, nil
}

// glob returns an iterator that yields files matching the given patterns and content types.
// The patterns and content types are provided as pairs of strings.
// If a content type is an empty string, all files matching the pattern are yielded.
// The iterator stops after the first pattern that matches any files.
func glob(fsys fs.FS, patternOrContentType ...string) iter.Seq2[string, error] {
	if len(patternOrContentType)%2 != 0 {
		panic("glob: patternOrContentType must have an even number of elements")
	}

	return func(yield func(string, error) bool) {
		for i := 0; i < len(patternOrContentType); i += 2 {
			pattern := patternOrContentType[i]
			contentType := patternOrContentType[i+1]

			matches, err := fs.Glob(fsys, pattern)
			if err != nil {
				yield("", err)
				return
			}

			if len(matches) > 0 {
				for _, match := range matches {
					if contentType == "" {
						if !yield(match, nil) {
							return
						}

						continue
					}

					ct, err := detectContentType(fsys, match)
					if err != nil {
						yield("", err)
						return
					}

					if ct == contentType {
						if !yield(match, nil) {
							return
						}
					}
				}

				return
			}
		}
	}
}

func filesSeq(fsys fs.FS) iter.Seq[string] {
	return func(yield func(string) bool) {
		for match := range glob(fsys,
			"*.safetensors", "",
			"*.bin", "application/zip",
			"*.pth", "application/zip",
			"*.gguf", "application/octet-stream",
			"*.bin", "application/octet-stream") {
			if !yield(match) {
				return
			}
		}

		for match := range glob(fsys,
			"tokenizer.json", "application/json",
			"tokenizer.model", "application/octet-stream",
		) {
			if !yield(match) {
				return
			}
		}

		for match := range glob(fsys, "*.json", "") {
			if !yield(match) {
				return
			}
		}

		for match := range glob(fsys, "**/*.json", "") {
			if !yield(match) {
				return
			}
		}
	}
}

func get[T any](m map[string]any, key string) (t T) {
	if v, ok := m[key].(T); ok {
		t = v
	}
	return
}

var deprecatedParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

func createRequest(modelfile *parser.Modelfile, dir string) (*api.CreateRequest, error) {
	m := make(map[string]any)
	parameters := make(map[string]any)
	var files, adapters []api.File

	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, cmd := range modelfile.Commands {
		switch cmd.Name {
		case "model", "adapter":
			path, err := expandPath(cmd.Args, dir)
			if err != nil {
				return nil, err
			}

			fsys := os.DirFS(path)
			seq := filesSeq(fsys)
			if fi, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
				m["from"] = cmd.Args
				break
			} else if err != nil {
				return nil, err
			} else if !fi.IsDir() {
				base := filepath.Base(path)
				path = filepath.Dir(path)
				seq = func(yield func(string) bool) {
					yield(base)
				}
			}

			var mu sync.Mutex
			for file := range seq {
				g.Go(func() error {
					f, err := os.Open(filepath.Join(path, file))
					if err != nil {
						return err
					}
					defer f.Close()

					sha256sum := sha256.New()
					if _, err := io.Copy(sha256sum, f); err != nil {
						return err
					}

					file := api.File{
						Name:   file,
						Path:   filepath.Join(path, file),
						Digest: "sha256:" + hex.EncodeToString(sha256sum.Sum(nil)),
					}

					mu.Lock()
					defer mu.Unlock()
					switch cmd.Name {
					case "model":
						files = append(files, file)
					case "adapter":
						adapters = append(adapters, file)
					}

					return nil
				})
			}

		case "template", "system", "renderer", "parser":
			m[cmd.Name] = cmd.Args
		case "license":
			m[cmd.Name] = append(get[[]string](m, cmd.Name), cmd.Args)
		case "message":
			role, msg, found := strings.Cut(cmd.Args, ": ")
			if !found {
				return nil, fmt.Errorf("invalid message command: %s", cmd.Args)
			}

			m[cmd.Name] = append(get[[]api.Message](m, cmd.Name), api.Message{
				Role:    role,
				Content: msg,
			})
		default:
			if slices.Contains(deprecatedParameters, cmd.Name) {
				slog.Warn("parameter is deprecated", "name", cmd.Name)
				break
			}

			ps, err := api.FormatParameters(map[string][]string{cmd.Name: {cmd.Args}})
			if err != nil {
				return nil, err
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

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return &api.CreateRequest{
		From:       get[string](m, "from"),
		Files:      files,
		Adapters:   adapters,
		License:    get[[]string](m, "license"),
		Messages:   get[[]api.Message](m, "message"),
		Parameters: parameters,
		Parser:     get[string](m, "parser"),
		Renderer:   get[string](m, "renderer"),
		System:     get[string](m, "system"),
		Template:   get[string](m, "template"),
	}, nil
}

func CreateHandler(cmd *cobra.Command, args []string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	var reader io.Reader

	filename, err := getModelfileName(cmd)
	if os.IsNotExist(err) {
		if filename == "" {
			reader = strings.NewReader("FROM .\n")
		} else {
			return errModelfileNotFound
		}
	} else if err != nil {
		return err
	} else {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}

		reader = f
		defer f.Close()
	}

	modelfile, err := parser.ParseFile(reader)
	if err != nil {
		return err
	}

	status := "gathering model components"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	req, err := createRequest(modelfile, filepath.Dir(filename))
	if err != nil {
		return err
	}
	spinner.Stop()

	req.Model = args[0]
	quantize, _ := cmd.Flags().GetString("quantize")
	if quantize != "" {
		req.Quantize = quantize
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(0))
	for _, file := range req.Files {
		g.Go(func() error {
			_, err := createBlob(cmd, client, file.Path, file.Digest, p)
			return err
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	bars := make(map[string]*progress.Bar)
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("pulling %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	if err := client.Create(cmd.Context(), req, fn); err != nil {
		if strings.Contains(err.Error(), "path or Modelfile are required") {
			return fmt.Errorf("the ollama server must be updated to use `ollama create` with this client")
		}
		return err
	}

	return nil
}

func createBlob(cmd *cobra.Command, client *api.Client, path string, digest string, p *progress.Progress) (string, error) {
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}

	bin, err := os.Open(realPath)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	// Get file info to retrieve the size
	fileInfo, err := bin.Stat()
	if err != nil {
		return "", err
	}
	fileSize := fileInfo.Size()

	var pw progressWriter
	status := fmt.Sprintf("copying file %s 0%%", digest)
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)
	defer spinner.Stop()

	done := make(chan struct{})
	defer close(done)

	go func() {
		ticker := time.NewTicker(60 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				spinner.SetMessage(fmt.Sprintf("copying file %s %d%%", digest, int(100*pw.n.Load()/fileSize)))
			case <-done:
				spinner.SetMessage(fmt.Sprintf("copying file %s 100%%", digest))
				return
			}
		}
	}()

	if err := client.CreateBlob(cmd.Context(), digest, io.TeeReader(bin, &pw)); err != nil {
		return "", err
	}
	return digest, nil
}
