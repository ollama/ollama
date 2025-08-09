package cmd

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

const completionDesc = `
Generate autocompletion scripts for ollama for the specified shell.
`
const bashCompDesc = `
Generate the autocompletion script for ollama for the bash shell.

To load completions in your current shell session:

    source <(ollama completion bash)

To load completions for every new session, execute once:
- Linux:

      ollama completion bash > /etc/bash_completion.d/ollama

- MacOS:

      ollama completion bash > /usr/local/etc/bash_completion.d/ollama
`

const zshCompDesc = `
Generate the autocompletion script for ollama for the zsh shell.

To load completions in your current shell session:

    source <(ollama completion zsh)

To load completions for every new session, execute once:

    ollama completion zsh > "${fpath[1]}/_ollama"
`

const fishCompDesc = `
Generate the autocompletion script for ollama for the fish shell.

To load completions in your current shell session:

    ollama completion fish | source

To load completions for every new session, execute once:

    ollama completion fish > ~/.config/fish/completions/ollama.fish

You will need to start a new shell for this setup to take effect.
`

const powershellCompDesc = `
Generate the autocompletion script for powershell.

To load completions in your current shell session:
PS C:\> ollama completion powershell | Out-String | Invoke-Expression

To load completions for every new session, add the output of the above command
to your powershell profile.
`

var disableCompDescriptions bool

const (
	noDescFlagName = "no-descriptions"
	noDescFlagText = "disable completion descriptions"
)

func generateCompletionCommand(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "completion",
		Short: "Generate autocompletion scripts for the specified shell",
		Long:  completionDesc,
		Args:  cobra.NoArgs,
	}

	bash := &cobra.Command{
		Use:   "bash",
		Short: "Generate autocompletion script for bash",
		Long:  bashCompDesc,
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return bashCompletionHandler(out, cmd)
		},
	}
	bash.Flags().BoolVar(&disableCompDescriptions, noDescFlagName, false, noDescFlagText)

	zsh := &cobra.Command{
		Use:   "zsh",
		Short: "Generate autocompletion script for zsh",
		Long:  zshCompDesc,
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return zshCompletionHandler(out, cmd)
		},
	}
	zsh.Flags().BoolVar(&disableCompDescriptions, noDescFlagName, false, noDescFlagText)

	fish := &cobra.Command{
		Use:   "fish",
		Short: "Generate autocompletion script for fish",
		Long:  fishCompDesc,
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fishCompletionHandler(out, cmd)
		},
	}
	fish.Flags().BoolVar(&disableCompDescriptions, noDescFlagName, false, noDescFlagText)

	powershell := &cobra.Command{
		Use:   "powershell",
		Short: "Generate autocompletion script for powershell",
		Long:  powershellCompDesc,
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return powershellCompletionHandler(out, cmd)
		},
	}
	powershell.Flags().BoolVar(&disableCompDescriptions, noDescFlagName, false, noDescFlagText)

	cmd.AddCommand(bash, zsh, fish, powershell)

	return cmd
}

func bashCompletionHandler(out io.Writer, cmd *cobra.Command) error {
	err := cmd.Root().GenBashCompletionV2(out, !disableCompDescriptions)

	if binary := filepath.Base(os.Args[0]); binary != "ollama" {
		renamedBinaryHook := `
# Hook the command used to generate the completion script
# to the ollama completion function to handle the case where
# the user renamed the ollama binary
if [[ $(type -t compopt) = "builtin" ]]; then
    complete -o default -F __start_ollama %[1]s
else
    complete -o default -o nospace -F __start_ollama %[1]s
fi
`
		fmt.Fprintf(out, renamedBinaryHook, binary)
	}
	return err
}

func zshCompletionHandler(out io.Writer, cmd *cobra.Command) error {
	var err error
	if disableCompDescriptions {
		err = cmd.Root().GenZshCompletionNoDesc(out)
	} else {
		err = cmd.Root().GenZshCompletion(out)
	}

	if binary := filepath.Base(os.Args[0]); binary != "ollama" {
		renamedBinaryHook := `
# Hook the command used to generate the completion script
# to the ollama completion function to handle the case where
# the user renamed the ollama binary
if [[ $(type -t compopt) = "builtin" ]]; then
    complete -o default -F __start_ollama %[1]s
else
    complete -o default -o nospace -F __start_ollama %[1]s
fi
`
		fmt.Fprintf(out, renamedBinaryHook, binary)
	}

	// Cobra doesn't source zsh completion file, explicitly doing it here
	fmt.Fprintf(out, "compdef _ollama ollama")

	return err
}

func fishCompletionHandler(out io.Writer, cmd *cobra.Command) error {
	return cmd.Root().GenFishCompletion(out, !disableCompDescriptions)
}

func powershellCompletionHandler(out io.Writer, cmd *cobra.Command) error {
	if disableCompDescriptions {
		return cmd.Root().GenPowerShellCompletion(out)
	}
	return cmd.Root().GenPowerShellCompletionWithDesc(out)
}

func runningModelSuggestions(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	if len(args) > 0 {
		return nil, cobra.ShellCompDirectiveNoFileComp
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}
	models, err := client.ListRunning(cmd.Context())
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}
	suggestions := make([]string, 0, len(models.Models))
	for _, m := range models.Models {
		if toComplete == "" || strings.HasPrefix(m.Name, toComplete) {
			suggestions = append(suggestions, m.Name)
		}
	}
	return suggestions, cobra.ShellCompDirectiveNoFileComp
}

func multiModelSuggestions(maxArgs int, cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	if maxArgs != -1 && len(args) > maxArgs {
		return nil, cobra.ShellCompDirectiveNoFileComp
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}
	models, err := client.List(cmd.Context())
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}
	suggestions := make([]string, 0, len(models.Models))
	for _, m := range models.Models {
		if (toComplete == "" || strings.HasPrefix(m.Name, toComplete)) && !slices.Contains(args, m.Name) {
			suggestions = append(suggestions, m.Name)
		}
	}
	return suggestions, cobra.ShellCompDirectiveNoFileComp
}
