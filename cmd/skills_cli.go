package cmd

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/skills"
)

func SkillsListHandler(cmd *cobra.Command, args []string) error {
	all, err := skills.List()
	if err != nil {
		return err
	}

	if len(all) == 0 {
		fmt.Fprintln(os.Stdout, "No skills installed. Use `ollama skills install <path>` to add one.")
		return nil
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ENABLED", "INPUTS", "OUTPUTS", "PERMISSIONS", "DESCRIPTION"})
	table.SetAutoWrapText(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	for _, skill := range all {
		table.Append([]string{
			skill.Spec.Name,
			boolToYesNo(skill.Enabled),
			joinOrDash(skill.Spec.IO.Inputs),
			joinOrDash(skill.Spec.IO.Outputs),
			joinOrDash(skill.Spec.Permissions.Required),
			valueOrDash(skill.Spec.Description),
		})
	}
	table.Render()

	return nil
}

func SkillsInstallHandler(cmd *cobra.Command, args []string) error {
	installed, err := skills.Install(args[0])
	if err != nil {
		return err
	}

	autoEnable, _ := cmd.Flags().GetBool("enable")
	if autoEnable {
		if err := skills.Enable(installed.Spec.Name); err != nil {
			return err
		}
	}

	fmt.Fprintf(os.Stdout, "Installed skill '%s'.\n", installed.Spec.Name)
	if autoEnable {
		fmt.Fprintf(os.Stdout, "Enabled skill '%s'.\n", installed.Spec.Name)
	}

	return nil
}

func SkillsEnableHandler(cmd *cobra.Command, args []string) error {
	if err := skills.Enable(args[0]); err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Enabled skill '%s'.\n", args[0])
	return nil
}

func SkillsDisableHandler(cmd *cobra.Command, args []string) error {
	if err := skills.Disable(args[0]); err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Disabled skill '%s'.\n", args[0])
	return nil
}

func SkillsRunHandler(cmd *cobra.Command, args []string) error {
	name := args[0]
	runArgs := args[1:]
	allow, err := cmd.Flags().GetStringSlice("allow")
	if err != nil {
		return err
	}

	return skills.RunWithOptions(cmd.Context(), name, runArgs, os.Stdin, os.Stdout, os.Stderr, skills.RunOptions{
		GrantedPermissions: allow,
	})
}

func printActiveSkills(w io.Writer, showEmpty bool) error {
	active, err := skills.Enabled()
	if err != nil {
		return err
	}

	if len(active) == 0 {
		if showEmpty {
			fmt.Fprintln(w, "No active skills. Use `ollama skills enable <name>` to activate one.")
		}
		return nil
	}

	fmt.Fprintln(w, "Active skills in this chat:")
	for _, skill := range active {
		fmt.Fprintf(w, "  - %s\n", describeSkill(skill))
	}
	fmt.Fprintln(w)

	return nil
}

func describeSkill(skill skills.Skill) string {
	return fmt.Sprintf("%s: %s (inputs: %s; outputs: %s; permissions: %s)",
		skill.Spec.Name,
		valueOrDash(skill.Spec.Description),
		joinOrDash(skill.Spec.IO.Inputs),
		joinOrDash(skill.Spec.IO.Outputs),
		joinOrDash(skill.Spec.Permissions.Required),
	)
}

func boolToYesNo(v bool) string {
	if v {
		return "yes"
	}
	return "no"
}

func joinOrDash(items []string) string {
	if len(items) == 0 {
		return "-"
	}
	return strings.Join(items, ", ")
}

func valueOrDash(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return "-"
	}
	return v
}
