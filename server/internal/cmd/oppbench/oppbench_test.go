package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/server/internal/chunks"
	"golang.org/x/sync/errgroup"
)

func BenchmarkDownload(b *testing.B) {
	run := func(fileSize, chunkSize int64) {
		name := fmt.Sprintf("size=%d/chunksize=%d", fileSize, chunkSize)
		b.Run(name, func(b *testing.B) { benchmarkDownload(b, fileSize, chunkSize) })
	}

	run(100<<20, 8<<20)
	run(100<<20, 16<<20)
	run(100<<20, 32<<20)
	run(100<<20, 64<<20)
	run(100<<20, 128<<20) // 1 chunk
}

func run(ctx context.Context, c *http.Client, chunk chunks.Chunk) error {
	const blobURL = "https://ollama.com/v2/x/x/blobs/sha256-4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d"
	req, err := http.NewRequestWithContext(ctx, "GET", blobURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%s", chunk))
	res, err := c.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	_, err = io.CopyN(io.Discard, res.Body, chunk.Size()) // will io.EOF on short read
	return err
}

var sleepTime atomic.Int64

func benchmarkDownload(b *testing.B, fileSize, chunkSize int64) {
	client := &http.Client{
		Transport: func() http.RoundTripper {
			tr := http.DefaultTransport.(*http.Transport).Clone()
			tr.DisableKeepAlives = true
			return tr
		}(),
	}
	defer client.CloseIdleConnections()

	// warm up the client
	run(context.Background(), client, chunks.New(0, 1<<20))

	b.SetBytes(fileSize)
	b.ReportAllocs()

	// Give our CDN a min to breathe between benchmarks.
	time.Sleep(time.Duration(sleepTime.Swap(3)))

	for b.Loop() {
		g, ctx := errgroup.WithContext(b.Context())
		g.SetLimit(runtime.GOMAXPROCS(0))
		for chunk := range chunks.Of(fileSize, chunkSize) {
			g.Go(func() error { return run(ctx, client, chunk) })
		}
		if err := g.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWrite(b *testing.B) {
	b.Run("chunksize=1MiB", func(b *testing.B) { benchmarkWrite(b, 1<<20) })
}

func benchmarkWrite(b *testing.B, chunkSize int) {
	b.ReportAllocs()

	dir := b.TempDir()
	f, err := os.Create(filepath.Join(dir, "write-single"))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	data := make([]byte, chunkSize)
	b.SetBytes(int64(chunkSize))
	r := bytes.NewReader(data)
	for b.Loop() {
		r.Reset(data)
		_, err := io.Copy(f, r)
		if err != nil {
			b.Fatal(err)
		}
	}
}
