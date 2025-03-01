package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/cmd/opp/internal/safetensors"
	"golang.org/x/sync/errgroup"
)

var stdout io.Writer = os.Stdout

const usage = `Opp is a tool for pushing and pulling Ollama models.

Usage:

    opp [flags] <push|pull|import>

Commands:

    push    Upload a model to the Ollama server.
    pull    Download a model from the Ollama server.
    import  Import a model from a local safetensor directory.

Examples:

    # Pull a model from the Ollama server.
    opp pull library/llama3.2:latest

    # Push a model to the Ollama server.
    opp push username/my_model:8b 

    # Import a model from a local safetensor directory.
    opp import /path/to/safetensor

Envionment Variables:

    OLLAMA_MODELS
        The directory where models are pushed and pulled from
	(default ~/.ollama/models).
`

func main() {
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usage)
	}
	flag.Parse()

	c, err := ollama.DefaultCache()
	if err != nil {
		log.Fatal(err)
	}

	rc, err := ollama.DefaultRegistry()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	err = func() error {
		switch cmd := flag.Arg(0); cmd {
		case "pull":
			return cmdPull(ctx, rc, c)
		case "push":
			return cmdPush(ctx, rc, c)
		case "import":
			return cmdImport(ctx, c)
		default:
			if cmd == "" {
				flag.Usage()
			} else {
				fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
			}
			os.Exit(2)
			return errors.New("unreachable")
		}
	}()
	if err != nil {
		fmt.Fprintf(os.Stderr, "opp: %v\n", err)
		os.Exit(1)
	}
}

func cmdPull(ctx context.Context, rc *ollama.Registry, c *blob.DiskCache) error {
	model := flag.Arg(1)
	if model == "" {
		flag.Usage()
		os.Exit(1)
	}

	tr := http.DefaultTransport.(*http.Transport).Clone()
	// TODO(bmizerany): configure transport?
	rc.HTTPClient = &http.Client{Transport: tr}

	var mu sync.Mutex
	p := make(map[blob.Digest][2]int64) // digest -> [total, downloaded]

	var pb bytes.Buffer
	printProgress := func() {
		pb.Reset()
		mu.Lock()
		for d, s := range p {
			// Write progress to a buffer first to avoid blocking
			// on stdout while holding the lock.
			stamp := time.Now().Format("2006/01/02 15:04:05")
			fmt.Fprintf(&pb, "%s %s pulling %d/%d (%.1f%%)\n", stamp, d.Short(), s[1], s[0], 100*float64(s[1])/float64(s[0]))
			if s[0] == s[1] {
				delete(p, d)
			}
		}
		mu.Unlock()
		io.Copy(stdout, &pb)
	}

	ctx = ollama.WithTrace(ctx, &ollama.Trace{
		Update: func(l *ollama.Layer, n int64, err error) {
			if err != nil && !errors.Is(err, ollama.ErrCached) {
				fmt.Fprintf(stdout, "opp: pull %s ! %v\n", l.Digest.Short(), err)
				return
			}

			mu.Lock()
			p[l.Digest] = [2]int64{l.Size, n}
			mu.Unlock()
		},
	})

	errc := make(chan error)
	go func() {
		errc <- rc.Pull(ctx, c, model)
	}()

	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			printProgress()
		case err := <-errc:
			printProgress()
			return err
		}
	}
}

func cmdPush(ctx context.Context, rc *ollama.Registry, c *blob.DiskCache) error {
	args := flag.Args()[1:]
	flag := flag.NewFlagSet("push", flag.ExitOnError)
	flagFrom := flag.String("from", "", "Use the manifest from a model by another name.")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: opp push <model>\n")
		flag.PrintDefaults()
	}
	flag.Parse(args)

	model := flag.Arg(0)
	if model == "" {
		return fmt.Errorf("missing model argument")
	}

	from := cmp.Or(*flagFrom, model)
	m, err := rc.ResolveLocal(c, from)
	if err != nil {
		return err
	}

	ctx = ollama.WithTrace(ctx, &ollama.Trace{
		Update: func(l *ollama.Layer, n int64, err error) {
			switch {
			case errors.Is(err, ollama.ErrCached):
				fmt.Fprintf(stdout, "opp: uploading %s %d (existed)", l.Digest.Short(), n)
			case err != nil:
				fmt.Fprintf(stdout, "opp: uploading %s %d ! %v\n", l.Digest.Short(), n, err)
			case n == 0:
				l := m.Layer(l.Digest)
				mt, p, _ := mime.ParseMediaType(l.MediaType)
				mt, _ = strings.CutPrefix(mt, "application/vnd.ollama.image.")
				switch mt {
				case "tensor":
					fmt.Fprintf(stdout, "opp: uploading tensor %s %s\n", l.Digest.Short(), p["name"])
				default:
					fmt.Fprintf(stdout, "opp: uploading %s %s\n", l.Digest.Short(), l.MediaType)
				}
			}
		},
	})

	return rc.Push(ctx, c, model, &ollama.PushParams{
		From: from,
	})
}

type trackingReader struct {
	io.Reader
	n *atomic.Int64
}

func (r *trackingReader) Read(p []byte) (n int, err error) {
	n, err = r.Reader.Read(p)
	r.n.Add(int64(n))
	return n, err
}

func cmdImport(ctx context.Context, c *blob.DiskCache) error {
	args := flag.Args()[1:]
	flag := flag.NewFlagSet("import", flag.ExitOnError)
	flagAs := flag.String("as", "", "Import using the provided name.")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: opp import <SafetensorDir>\n")
		flag.PrintDefaults()
	}
	flag.Parse(args)
	if *flagAs == "" {
		return fmt.Errorf("missing -as flag")
	}
	as := ollama.CompleteName(*flagAs)

	dir := cmp.Or(flag.Arg(0), ".")
	fmt.Fprintf(os.Stderr, "Reading %s\n", dir)

	m, err := safetensors.Read(os.DirFS(dir))
	if err != nil {
		return err
	}

	var total int64
	var tt []*safetensors.Tensor
	for t, err := range m.Tensors() {
		if err != nil {
			return err
		}
		tt = append(tt, t)
		total += t.Size()
	}

	var n atomic.Int64
	done := make(chan error)
	go func() {
		layers := make([]*ollama.Layer, len(tt))
		var g errgroup.Group
		g.SetLimit(runtime.GOMAXPROCS(0))
		var ctxErr error
		for i, t := range tt {
			if ctx.Err() != nil {
				// The context may cancel AFTER we exit the
				// loop, and so if we use ctx.Err() after the
				// loop we may report it as the error that
				// broke the loop, when it was not. This can
				// manifest as a false-negative, leading the
				// user to think their import failed when it
				// did not, so capture it if and only if we
				// exit the loop because of a ctx.Err() and
				// report it.
				ctxErr = ctx.Err()
				break
			}
			g.Go(func() (err error) {
				rc, err := t.Reader()
				if err != nil {
					return err
				}
				defer rc.Close()
				tr := &trackingReader{rc, &n}
				d, err := c.Import(tr, t.Size())
				if err != nil {
					return err
				}
				if err := rc.Close(); err != nil {
					return err
				}

				layers[i] = &ollama.Layer{
					Digest: d,
					Size:   t.Size(),
					MediaType: mime.FormatMediaType("application/vnd.ollama.image.tensor", map[string]string{
						"name":  t.Name(),
						"dtype": t.DataType(),
						"shape": t.Shape().String(),
					}),
				}

				return nil
			})
		}

		done <- func() error {
			if err := errors.Join(g.Wait(), ctxErr); err != nil {
				return err
			}
			m := &ollama.Manifest{Layers: layers}
			data, err := json.MarshalIndent(m, "", "  ")
			if err != nil {
				return err
			}
			d := blob.DigestFromBytes(data)
			err = blob.PutBytes(c, d, data)
			if err != nil {
				return err
			}
			return c.Link(as, d)
		}()
	}()

	fmt.Fprintf(stdout, "Importing %d tensors from %s\n", len(tt), dir)

	csiHideCursor(stdout)
	defer csiShowCursor(stdout)

	csiSavePos(stdout)
	writeProgress := func() {
		csiRestorePos(stdout)
		nn := n.Load()
		fmt.Fprintf(stdout, "Imported %s/%s bytes (%d%%)%s\n",
			formatNatural(nn),
			formatNatural(total),
			nn*100/total,
			ansiClearToEndOfLine,
		)
	}

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			writeProgress()
		case err := <-done:
			writeProgress()
			fmt.Println()
			fmt.Println("Successfully imported", as)
			return err
		}
	}
}

func formatNatural(n int64) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%d B", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1f KB", float64(n)/1024)
	case n < 1024*1024*1024:
		return fmt.Sprintf("%.1f MB", float64(n)/(1024*1024))
	default:
		return fmt.Sprintf("%.1f GB", float64(n)/(1024*1024*1024))
	}
}

const ansiClearToEndOfLine = "\033[K"

func csiSavePos(w io.Writer)    { fmt.Fprint(w, "\033[s") }
func csiRestorePos(w io.Writer) { fmt.Fprint(w, "\033[u") }
func csiHideCursor(w io.Writer) { fmt.Fprint(w, "\033[?25l") }
func csiShowCursor(w io.Writer) { fmt.Fprint(w, "\033[?25h") }
