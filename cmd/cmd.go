package cmd

import (
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/containerd/console"
	"github.com/ollama/ollama/envconfig"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cobra.EnableCommandSorting = false

	if runtime.GOOS == "windows" && term.IsTerminal(int(os.Stdout.Fd())) {
		console.ConsoleFromFile(os.Stdin) //nolint:errcheck
	}

	rootCmd := NewOllamaCmd()
	createCmd := NewCreateCmd()
	showCmd := NewShowCmd()
	runCmd := NewRunCmd()
	stopCmd := NewStopCmd()
	serveCmd := NewServeCmd()
	pullCmd := NewPullCmd()
	pushCmd := NewPushCmd()
	listCmd := NewListCmd()
	psCmd := NewPsCmd()
	copyCmd := NewCopyCmd()
	deleteCmd := NewDeleteCmd()
	runnerCmd := NewRunnerCmd()

	envVars := envconfig.AsMap()

	envs := []envconfig.EnvVar{envVars["OLLAMA_HOST"]}

	for _, cmd := range []*cobra.Command{
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		serveCmd,
	} {
		switch cmd {
		case runCmd:
			appendEnvDocs(cmd, []envconfig.EnvVar{envVars["OLLAMA_HOST"], envVars["OLLAMA_NOHISTORY"]})
		case serveCmd:
			appendEnvDocs(cmd, []envconfig.EnvVar{
				envVars["OLLAMA_DEBUG"],
				envVars["OLLAMA_HOST"],
				envVars["OLLAMA_KEEP_ALIVE"],
				envVars["OLLAMA_MAX_LOADED_MODELS"],
				envVars["OLLAMA_MAX_QUEUE"],
				envVars["OLLAMA_MODELS"],
				envVars["OLLAMA_NUM_PARALLEL"],
				envVars["OLLAMA_NOPRUNE"],
				envVars["OLLAMA_ORIGINS"],
				envVars["OLLAMA_SCHED_SPREAD"],
				envVars["OLLAMA_TMPDIR"],
				envVars["OLLAMA_FLASH_ATTENTION"],
				envVars["OLLAMA_KV_CACHE_TYPE"],
				envVars["OLLAMA_LLM_LIBRARY"],
				envVars["OLLAMA_GPU_OVERHEAD"],
				envVars["OLLAMA_LOAD_TIMEOUT"],
			})
		default:
			appendEnvDocs(cmd, envs)
		}
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		runnerCmd,
	)

	return rootCmd
}

func appendEnvDocs(cmd *cobra.Command, envs []envconfig.EnvVar) {
	if len(envs) == 0 {
		return
	}

	envUsage := `
Environment Variables:
`
	for _, e := range envs {
		envUsage += fmt.Sprintf("      %-24s   %s\n", e.Name, e.Description)
	}

	cmd.SetUsageTemplate(cmd.UsageTemplate() + envUsage)
}
