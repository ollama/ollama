package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"path"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"golang.org/x/term"
	"gonum.org/v1/gonum/mat"

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
	_, err := os.Stat(args[0])
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := pull(args[0]); err != nil {
			return err
		}
	case err != nil:
		return err
	}

	return RunGenerate(cmd, args)
}

func pull(model string) error {
	client := api.NewClient()
	var bar *progressbar.ProgressBar
	return client.Pull(
		context.Background(),
		&api.PullRequest{Model: model},
		func(progress api.PullProgress) error {
			if bar == nil && progress.Percent == 100 {
				// already downloaded
				return nil
			}
			if bar == nil {
				bar = progressbar.DefaultBytes(progress.Total)
			}

			return bar.Set64(progress.Completed)
		},
	)
}

func RunGenerate(_ *cobra.Command, args []string) error {
	// join all args into a single prompt
	prompt := strings.Join(args[1:], " ")
	if len(args) > 1 {
		_, err := generate(args[0], prompt)
		return err
	}

	if term.IsTerminal(int(os.Stdin.Fd())) {
		return generateInteractive(args[0])
	}

	return generateBatch(args[0])
}

// TODO: rather than setting this, add an endpoint to the server to tokenize the prompt so we can get the real length of the prompt
const maxChars = 250 // currently the max default context in the server is 512, this sets characters to a range that hopefully won't exceed the token length

// stuffPrompt adds adds more context to the prompt from the current session
func stuffPrompt(prompt string, similar VectorSlice) string {
	if len(prompt) >= maxChars {
		return prompt
	}
	for _, s := range similar {
		userInput := fmt.Sprintf(". I previously stated %q", s.UserInput)
		if len(prompt)+len(userInput) < maxChars {
			prompt += userInput
		}
		modelResponse := fmt.Sprintf(". You previously responded %q", s.ModelResponse)
		if len(prompt)+len(modelResponse) < maxChars {
			prompt += modelResponse
		}
	}
	return prompt
}

// generateWithEmbeddings adds additional context to the prompt from the current session
func generateWithEmbeddings(model string, embeddings *VectorSlice, prompt string) error {
	input := prompt
	client := api.NewClient()
	// get the embedding of the current prompt to find similar prompts stored in memory
	e, err := client.Embedding(context.Background(), api.EmbeddingRequest{Model: model, Input: prompt})
	if err != nil {
		return err
	}
	embedding := mat.NewVecDense(len(e.Embedding), e.Embedding)
	similar := embeddings.NearestNeighbors(embedding, 2)

	prompt = stuffPrompt(input, similar)

	generated, err := generate(model, prompt)
	if err != nil {
		return err
	}

	go func() {
		fullText := fmt.Sprintf("%s %s", prompt, generated)
		// if the prompt got stuffed, only add the original input to avoid nesting user inputs
		if prompt != input {
			fullText = fmt.Sprintf("%s %s", input, generated)
		}
		e, err = client.Embedding(context.Background(), api.EmbeddingRequest{Model: model, Input: fullText})
		if err != nil {
			return
		}
		embeddings.Add(Vector{
			UserInput:     prompt,
			ModelResponse: generated,
			Data:          mat.NewVecDense(len(e.Embedding), e.Embedding),
		})
	}()
	return nil
}

func generate(model, prompt string) (string, error) {
	result := ""
	if len(strings.TrimSpace(prompt)) > 0 {
		client := api.NewClient()

		spinner := progressbar.NewOptions(-1,
			progressbar.OptionSetWriter(os.Stderr),
			progressbar.OptionThrottle(60*time.Millisecond),
			progressbar.OptionSpinnerType(14),
			progressbar.OptionSetRenderBlankState(true),
			progressbar.OptionSetElapsedTime(false),
			progressbar.OptionClearOnFinish(),
		)

		go func() {
			for range time.Tick(60 * time.Millisecond) {
				if spinner.IsFinished() {
					break
				}

				spinner.Add(1)
			}
		}()

		request := api.GenerateRequest{Model: model, Prompt: prompt}
		fn := func(resp api.GenerateResponse) error {
			if !spinner.IsFinished() {
				spinner.Finish()
			}

			fmt.Print(resp.Response)
			result += resp.Response
			return nil
		}

		if err := client.Generate(context.Background(), &request, fn); err != nil {
			return "", err
		}

		fmt.Println()
		fmt.Println()
	}

	return result, nil
}

// generateInteractive runs the generator in interactive mode which has a memory of previous prompts
func generateInteractive(model string) error {
	fmt.Print(">>> ")
	scanner := bufio.NewScanner(os.Stdin)
	embeddings := &VectorSlice{}
	for scanner.Scan() {
		if err := generateWithEmbeddings(model, embeddings, scanner.Text()); err != nil {
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
		if _, err := generate(model, prompt); err != nil {
			return err
		}
	}

	return nil
}

func RunServer(_ *cobra.Command, _ []string) error {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "127.0.0.1"
	}

	port := os.Getenv("OLLAMA_PORT")
	if port == "" {
		port = "11434"
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%s", host, port))
	if err != nil {
		return err
	}

	return server.Serve(ln)
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
