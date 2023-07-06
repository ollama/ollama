package cmd

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
)

func cacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama")
}

func RunRun(cmd *cobra.Command, args []string) error {
	if err := pull(args[0]); err != nil {
		return err
	}

	fmt.Println("Up to date.")
	return RunGenerate(cmd, args)
}

func pull(model string) error {
	client, err := NewAPIClient()
	if err != nil {
		return err
	}

	var bar *progressbar.ProgressBar
	return client.Pull(
		context.Background(),
		&api.PullRequest{Model: model},
		func(progress api.PullProgress) error {
			if bar == nil {
				bar = progressbar.DefaultBytes(progress.Total)
			}

			return bar.Set64(progress.Completed)
		},
	)
}

func RunGenerate(_ *cobra.Command, args []string) error {
	if len(args) > 1 {
		return generate(args[0], args[1:]...)
	}

	if term.IsTerminal(int(os.Stdin.Fd())) {
		return generateInteractive(args[0])
	}

	return generateBatch(args[0])
}

func generate(model string, prompts ...string) error {
	client, err := NewAPIClient()
	if err != nil {
		return err
	}

	for _, prompt := range prompts {
		client.Generate(context.Background(), &api.GenerateRequest{Model: model, Prompt: prompt}, func(resp api.GenerateResponse) error {
			fmt.Print(resp.Response)
			return nil
		})
	}

	fmt.Println()
	fmt.Println()
	return nil
}

func generateInteractive(model string) error {
	fmt.Print(">>> ")
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		if err := generate(model, scanner.Text()); err != nil {
			return err
		}

		fmt.Print(">>> ")
	}

	return nil
}

func generateBatch(model string) error {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		prompt := scanner.Text()
		fmt.Printf(">>> %s\n", prompt)
		if err := generate(model, prompt); err != nil {
			return err
		}
	}

	return nil
}

func RunServer(_ *cobra.Command, _ []string) error {
	ln, err := net.Listen("tcp", "127.0.0.1:11434")
	if err != nil {
		return err
	}

	return server.Serve(ln)
}

func NewAPIClient() (*api.Client, error) {
	return &api.Client{
		URL: "http://localhost:11434",
	}, nil
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	rootCmd := &cobra.Command{
		Use:          "ollama",
		Short:        "Large language model runner",
		SilenceUsage: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		PersistentPreRunE: func(_ *cobra.Command, args []string) error {
			// create the models directory and it's parent
			return os.MkdirAll(path.Join(cacheDir(), "models"), 0o700)
		},
	}

	cobra.EnableCommandSorting = false

	runCmd := &cobra.Command{
		Use:   "run MODEL [PROMPT]",
		Short: "Run a model",
		Args:  cobra.MinimumNArgs(1),
		RunE:  RunRun,
	}

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE:    RunServer,
	}

	rootCmd.AddCommand(
		serveCmd,
		runCmd,
	)

	return rootCmd
}
