package cmd

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"
	"sync"

	"github.com/gosuri/uiprogress"
	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
	"github.com/spf13/cobra"
)

func cacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama")
}

func bytesToGB(bytes int) float64 {
	return float64(bytes) / float64(1<<30)
}

func run(model string) error {
	client, err := NewAPIClient()
	if err != nil {
		return err
	}
	pr := api.PullRequest{
		Model: model,
	}
	var bar *uiprogress.Bar
	mutex := &sync.Mutex{}
	var progressData api.PullProgress

	pullCallback := func(progress api.PullProgress) {
		mutex.Lock()
		progressData = progress
		if bar == nil {
			uiprogress.Start()                           // start rendering
			bar = uiprogress.AddBar(int(progress.Total)) // Add a new bar

			// display the total file size and how much has downloaded so far
			bar.PrependFunc(func(b *uiprogress.Bar) string {
				return fmt.Sprintf("Downloading: %.2f GB / %.2f GB", bytesToGB(progressData.Completed), bytesToGB(progressData.Total))
			})

			// display completion percentage
			bar.AppendFunc(func(b *uiprogress.Bar) string {
				return fmt.Sprintf(" %d%%", int((float64(progressData.Completed)/float64(progressData.Total))*100))
			})
		}
		bar.Set(int(progress.Completed))
		mutex.Unlock()
	}
	if err := client.Pull(context.Background(), &pr, pullCallback); err != nil {
		return err
	}
	fmt.Println("Up to date.")
	return nil
}

func serve() error {
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
		Use:   "ollama",
		Short: "Large language model runner",
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// Disable usage printing on errors
			cmd.SilenceUsage = true
			// create the models directory and it's parent
			if err := os.MkdirAll(path.Join(cacheDir(), "models"), 0o700); err != nil {
				panic(err)
			}
		},
	}

	cobra.EnableCommandSorting = false

	runCmd := &cobra.Command{
		Use:   "run MODEL",
		Short: "Run a model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(args[0])
		},
	}

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE: func(cmd *cobra.Command, args []string) error {
			return serve()
		},
	}

	rootCmd.AddCommand(
		serveCmd,
		runCmd,
	)

	return rootCmd
}
