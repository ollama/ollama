package cmd

import (
	"errors"
	"fmt"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

var showCmd = &cobra.Command{
	Use:     "show MODEL",
	Short:   "Show information for a model",
	Args:    cobra.ExactArgs(1),
	PreRunE: checkServerHeartbeat,
	RunE:    ShowHandler,
}

func init() {
	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system message of a model")
	appendHostEnvDocs(showCmd)
}

func ShowHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if len(args) != 1 {
		return errors.New("missing model name")
	}

	license, errLicense := cmd.Flags().GetBool("license")
	modelfile, errModelfile := cmd.Flags().GetBool("modelfile")
	parameters, errParams := cmd.Flags().GetBool("parameters")
	system, errSystem := cmd.Flags().GetBool("system")
	template, errTemplate := cmd.Flags().GetBool("template")

	for _, boolErr := range []error{errLicense, errModelfile, errParams, errSystem, errTemplate} {
		if boolErr != nil {
			return errors.New("error retrieving flags")
		}
	}

	flagsSet := 0
	showType := ""

	if license {
		flagsSet++
		showType = "license"
	}

	if modelfile {
		flagsSet++
		showType = "modelfile"
	}

	if parameters {
		flagsSet++
		showType = "parameters"
	}

	if system {
		flagsSet++
		showType = "system"
	}

	if template {
		flagsSet++
		showType = "template"
	}

	if flagsSet > 1 {
		return errors.New("only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
	} else if flagsSet == 0 {
		return errors.New("one of '--license', '--modelfile', '--parameters', '--system', or '--template' must be specified")
	}

	req := api.ShowRequest{Name: args[0]}
	resp, err := client.Show(cmd.Context(), &req)
	if err != nil {
		return err
	}

	switch showType {
	case "license":
		fmt.Println(resp.License)
	case "modelfile":
		fmt.Println(resp.Modelfile)
	case "parameters":
		fmt.Println(resp.Parameters)
	case "system":
		fmt.Println(resp.System)
	case "template":
		fmt.Println(resp.Template)
	}

	return nil
}
