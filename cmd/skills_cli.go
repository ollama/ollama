package cmd

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

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
		fmt.Fprintln(os.Stdout, "No skills installed. Use `ollama skills install <path|repo@ref>` to add one.")
		return nil
	}

	printSkillsTable(all)
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
	if installed.Metadata.Commit != "" {
		fmt.Fprintf(os.Stdout, "Pinned commit: %s\n", installed.Metadata.Commit)
	}
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
	dryRun, err := cmd.Flags().GetBool("dry-run")
	if err != nil {
		return err
	}
	timeoutSeconds, err := cmd.Flags().GetInt("timeout")
	if err != nil {
		return err
	}
	maxOutputBytes, err := cmd.Flags().GetInt64("max-output-bytes")
	if err != nil {
		return err
	}

	opts := skills.RunOptions{
		GrantedPermissions: allow,
		DryRun:             dryRun,
		Timeout:            time.Duration(timeoutSeconds) * time.Second,
		MaxOutputBytes:     maxOutputBytes,
	}

	start := time.Now()
	if err := skills.RunWithOptions(cmd.Context(), name, runArgs, os.Stdin, os.Stdout, os.Stderr, opts); err != nil {
		return err
	}

	duration := time.Since(start).Round(time.Millisecond)
	if dryRun {
		fmt.Fprintf(os.Stdout, "Trace: skill=%s dry_run=yes duration=%s\n", name, duration)
		return nil
	}

	fmt.Fprintf(os.Stdout, "Trace: skill=%s status=ok duration=%s\n", name, duration)
	return nil
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
	return fmt.Sprintf("%s: %s (inputs: %s; outputs: %s; required: %s)",
		skill.Spec.Name,
		valueOrDash(skill.Spec.Description),
		joinOrDash(skill.Spec.IO.Inputs),
		joinOrDash(skill.Spec.IO.Outputs),
		joinOrDash(skill.Spec.Permissions.Required),
	)
}

func printSkillsTable(all []skills.Skill) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ENABLED", "TAGS", "INPUTS", "OUTPUTS", "REQUIRED", "DESCRIPTION", "SOURCE"})
	table.SetAutoWrapText(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	for _, skill := range all {
		table.Append([]string{
			skill.Spec.Name,
			boolToYesNo(skill.Enabled),
			joinOrDash(skill.Spec.Tags),
			joinOrDash(skill.Spec.IO.Inputs),
			joinOrDash(skill.Spec.IO.Outputs),
			joinOrDash(skill.Spec.Permissions.Required),
			valueOrDash(skill.Spec.Description),
			valueOrDash(skill.Metadata.Source),
		})
	}
	table.Render()
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
