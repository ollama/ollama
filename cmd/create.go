package cmd

import (
	"crypto/sha256"
	"fmt"
	"io"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
)

type blob struct {
	Rel, Abs, Digest string
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

func createBlobs(mappings ...map[string]string) iter.Seq2[blob, error] {
	return func(yield func(blob, error) bool) {
		for _, mapping := range mappings {
			for rel, abs := range mapping {
				if abs, ok := strings.CutPrefix(abs, "abs:"); ok {
					f, err := os.Open(abs)
					if err != nil {
						yield(blob{}, err)
						return
					}

					h := sha256.New()
					if _, err := io.Copy(h, f); err != nil {
						yield(blob{}, err)
						return
					}

					if err := f.Close(); err != nil {
						yield(blob{}, err)
						return
					}

					if !yield(blob{
						Rel:    rel,
						Abs:    abs,
						Digest: fmt.Sprintf("sha256:%x", h.Sum(nil)),
					}, nil) {
						return
					}
				}
			}
		}
	}
}
