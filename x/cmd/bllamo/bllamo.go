// Bllamo is a (new) tool for managing Ollama models.
//
// Usage:
//
//	bllamo <command> [arguments]
//
// The commands are:
//
//	build     build a model from a Modelfile
//	list      list all models
//	push      push a model from an ollama registry
//	pull      pull a model from an ollama registry
//	delete    delete a model from an ollama registry
//	help      display help for a command
package main

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"

	"bllamo.com/api"
	"bllamo.com/build"
	"bllamo.com/client/ollama"
	"bllamo.com/registry"
)

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "bllamo: no command provided")
		os.Exit(2)
	}
	if err := Main(flag.Args()...); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

var TODOUsage = fmt.Errorf("TODO: usage")

var commands = map[string]func(ctx context.Context, args ...string) error{
	"build":    cmdBuild,
	"push":     cmdPush,
	"serve":    cmdServe,
	"registry": cmdRegistry,
}

// Main is the entry point for the blammo command.
func Main(args ...string) error {
	cmd := args[0]
	args = args[1:]
	if f, ok := commands[cmd]; ok {
		ctx := context.TODO()
		return f(ctx, args...)
	}
	return fmt.Errorf("blammo: unknown command %q", cmd)
}

func cmdBuild(ctx context.Context, args ...string) error {
	var v struct {
		Modelfile string `flag:"f,the Modelfile to use"`
	}

	fs := readFlags("build", args, &v)
	if fs.NArg() != 1 {
		return TODOUsage
	}

	modelfile, err := os.ReadFile(cmp.Or(v.Modelfile, "Modelfile"))
	if err != nil {
		return err
	}
	return ollama.Default().Build(ctx, args[0], modelfile, os.DirFS("."))
}

func cmdRegistry(_ context.Context, _ ...string) error {
	var s registry.Server
	return http.ListenAndServe(":8888", &s)
}

func cmdServe(ctx context.Context, args ...string) error {
	bs, err := build.Open("")
	if err != nil {
		return err
	}
	return http.ListenAndServe(":11434", &api.Server{Build: bs})
}

func cmdPush(ctx context.Context, args ...string) error {
	fs := readFlags("push", args, nil)
	if fs.NArg() != 1 {
		return TODOUsage
	}
	return ollama.Default().Push(ctx, fs.Arg(0))
}
