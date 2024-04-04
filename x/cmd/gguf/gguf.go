// Gguf is a tool for learning about GGUF files.
//
// Usage:
//
//	gguf [flags] <file>
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"text/tabwriter"

	"github.com/ollama/ollama/x/encoding/gguf"
)

func main() {
	if err := Main(os.Stdout, os.Args[1:]...); err != nil {
		log.Fatal(err)
	}
}

func Main(stdout io.Writer, args ...string) error {
	fs := flag.NewFlagSet("gguf", flag.ExitOnError)
	flagGPU := fs.Uint64("gpu", 0, "use N bytes of GPU memory (default is 0)")

	fs.Usage = func() {
		io.WriteString(stdout, "Gguf is a tool for learning about GGUF files.\n")
		io.WriteString(stdout, "\n")
		io.WriteString(stdout, "Usage:\n")
		io.WriteString(stdout, "\n")
		io.WriteString(stdout, "\tgguf [flags] <file>\n")
		io.WriteString(stdout, "\n")
		var numFlags int
		fs.VisitAll(func(*flag.Flag) { numFlags++ })
		if numFlags > 0 {
			io.WriteString(stdout, "Flags:\n")
			fs.PrintDefaults()
		}
	}
	fs.Parse(args)

	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}

	file := fs.Arg(0)
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	g, err := gguf.ReadFile(f)
	if err != nil {
		log.Fatal(err)
	}

	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	defer tw.Flush()

	fmt.Fprintf(tw, "version:\t%d\n", g.Version())

	for m, err := range g.Metadata {
		if err != nil {
			log.Fatal(err)
		}
		if len(m.Values) > 5 {
			fmt.Fprintf(tw, "meta:\t%q: ... (%d values)\n", m.Key, len(m.Values))
		} else {
			fmt.Fprintf(tw, "meta:\t%q: %v\n", m.Key, m.Values)
		}
	}

	var i int
	var totalLayerBytes uint64
	var offGPU bool
	for t, err := range g.Tensors {
		if err != nil {
			log.Fatal(err)
		}

		totalLayerBytes += t.Size
		if totalLayerBytes > *flagGPU {
			offGPU = true
		}

		const msg = "tensor (layer %000d):\t%q\t%s\tdims=%v\toffset=%d\tsize=%d\tonGPU=%v\n"
		fmt.Fprintf(tw, msg, i, t.Name, t.Type, t.Dimensions, t.Offset, t.Size, !offGPU)

		i++
	}
	return nil
}
