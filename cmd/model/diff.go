// model-diff performs offline comparisons of local MLX/safetensors and GGUF models.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	os.Exit(run(ctx, os.Args[1:], os.Stdout, os.Stderr))
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("model-diff", flag.ContinueOnError)
	flags.SetOutput(stderr)
	var opts Options
	var all, summary bool
	var limit int
	flags.BoolVar(&all, "all", false, "show unchanged entries, locations, and unabridged values")
	flags.BoolVar(&summary, "summary", false, "show only summary counts and verification coverage")
	flags.IntVar(&limit, "limit", 40, "maximum changed entries displayed (does not limit comparison)")
	flags.BoolVar(&opts.MetadataOnly, "metadata-only", false, "compare metadata and tensor descriptors without hashing payloads")
	flags.BoolVar(&opts.Stats, "stats", false, "compute quantization saturation and dequantized NMSE")
	flags.StringVar(&opts.Tensor, "tensor", "", "select tensor names by regular expression; includes weight companions")
	flags.StringVar(&opts.LeftStore, "left-store", "", "left Ollama model store (defaults to OLLAMA_MODELS)")
	flags.StringVar(&opts.RightStore, "right-store", "", "right Ollama model store (defaults to OLLAMA_MODELS)")
	flags.Usage = func() {
		fmt.Fprintln(stderr, "Usage: model-diff [flags] LEFT RIGHT\n\nInputs: Ollama names, manifest files, safetensors directories/files, or GGUF files.\nExit: 0 equal in requested scope, 1 different, 2 error/incomplete.\nMatching store blob hashes are trusted; distinct blobs are hashed per tensor.")
		flags.PrintDefaults()
	}
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if flags.NArg() != 2 || limit < 1 || all && summary || opts.Stats && opts.MetadataOnly {
		flags.Usage()
		return 2
	}
	prepared, err := prepareComparison(ctx, flags.Arg(0), flags.Arg(1), opts)
	if err != nil {
		fmt.Fprintf(stderr, "model-diff: %s\n", err)
		return 2
	}
	if err := writeTextHeader(stdout, prepared.left.source, prepared.right.source); err != nil {
		fmt.Fprintf(stderr, "model-diff: write input header: %s\n", err)
		return 2
	}
	report, err := prepared.compare(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "model-diff: %s\n", err)
		return 2
	}
	err = writeTextBody(stdout, report, all, summary, limit)
	if err != nil {
		fmt.Fprintf(stderr, "model-diff: write report: %s\n", err)
		return 2
	}
	if !report.Equal {
		return 1
	}
	return 0
}
