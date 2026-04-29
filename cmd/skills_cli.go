package cmd

import (
	"bufio"
	"errors"
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

func SkillsSearchHandler(cmd *cobra.Command, args []string) error {
	query := strings.Join(args, " ")

	includeInstalled, err := cmd.Flags().GetBool("installed")
	if err != nil {
		return err
	}
	includeCatalog, err := cmd.Flags().GetBool("catalog")
	if err != nil {
		return err
	}
	if !includeInstalled && !includeCatalog {
		return errors.New("at least one of --installed or --catalog must be enabled")
	}

	tags, err := cmd.Flags().GetStringSlice("tag")
	if err != nil {
		return err
	}
	permissions, err := cmd.Flags().GetStringSlice("permission")
	if err != nil {
		return err
	}
	verifiedOnly, err := cmd.Flags().GetBool("verified")
	if err != nil {
		return err
	}

	var localMatches []skills.Skill
	if includeInstalled {
		localMatches, err = skills.Search(query)
		if err != nil {
			return err
		}
		localMatches = filterLocalSkills(localMatches, tags, permissions, verifiedOnly)
	}

	var catalogMatches []skills.CatalogEntry
	if includeCatalog {
		catalogMatches, err = skills.SearchCatalog(skills.CatalogFilter{
			Query:        query,
			Tags:         tags,
			Permissions:  permissions,
			VerifiedOnly: verifiedOnly,
		})
		if err != nil {
			return err
		}
	}

	if len(localMatches) == 0 && len(catalogMatches) == 0 {
		fmt.Fprintf(os.Stdout, "No skills matched %q.\n", strings.TrimSpace(query))
		return nil
	}

	if len(localMatches) > 0 {
		fmt.Fprintln(os.Stdout, "Installed Skills")
		printSkillsTable(localMatches)
	}
	if len(catalogMatches) > 0 {
		if len(localMatches) > 0 {
			fmt.Fprintln(os.Stdout)
		}
		fmt.Fprintln(os.Stdout, "Catalog Skills")
		printCatalogTable(catalogMatches)
	}

	return nil
}

func SkillsInfoHandler(cmd *cobra.Command, args []string) error {
	skill, err := skills.Get(args[0])
	if err != nil {
		return err
	}

	backupCount, err := skills.BackupCount(skill.Spec.Name)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Name: %s\n", skill.Spec.Name)
	fmt.Fprintf(os.Stdout, "Description: %s\n", valueOrDash(skill.Spec.Description))
	fmt.Fprintf(os.Stdout, "Version: %s\n", valueOrDash(skill.Spec.Version))
	fmt.Fprintf(os.Stdout, "Tags: %s\n", joinOrDash(skill.Spec.Tags))
	fmt.Fprintf(os.Stdout, "Enabled: %s\n", boolToYesNo(skill.Enabled))
	fmt.Fprintf(os.Stdout, "Inputs: %s\n", joinOrDash(skill.Spec.IO.Inputs))
	fmt.Fprintf(os.Stdout, "Outputs: %s\n", joinOrDash(skill.Spec.IO.Outputs))
	fmt.Fprintf(os.Stdout, "Input schema: %s\n", formatFieldSchema(skill.Spec.IO.InputSchema))
	fmt.Fprintf(os.Stdout, "Output schema: %s\n", formatFieldSchema(skill.Spec.IO.OutputSchema))
	fmt.Fprintf(os.Stdout, "Required permissions: %s\n", joinOrDash(skill.Spec.Permissions.Required))
	fmt.Fprintf(os.Stdout, "Granted permissions: %s\n", joinOrDash(skill.GrantedPermissions))
	sessionGrants, err := skills.SessionGranted(skill.Spec.Name)
	if err == nil {
		fmt.Fprintf(os.Stdout, "Session permissions: %s\n", joinOrDash(sessionGrants))
	}
	fmt.Fprintf(os.Stdout, "Source: %s\n", valueOrDash(skill.Metadata.Source))
	fmt.Fprintf(os.Stdout, "Repository: %s\n", valueOrDash(skill.Metadata.Repository))
	fmt.Fprintf(os.Stdout, "Pinned ref: %s\n", valueOrDash(skill.Metadata.Ref))
	fmt.Fprintf(os.Stdout, "Commit: %s\n", valueOrDash(skill.Metadata.Commit))
	fmt.Fprintf(os.Stdout, "Digest: %s\n", valueOrDash(skill.Metadata.Digest))
	fmt.Fprintf(os.Stdout, "Verified: %s\n", boolToYesNo(skill.Metadata.Verified))
	fmt.Fprintf(os.Stdout, "Signed: %s\n", boolToYesNo(skill.Metadata.Signed))
	fmt.Fprintf(os.Stdout, "Verification key: %s\n", valueOrDash(skill.Metadata.PublicKey))
	fmt.Fprintf(os.Stdout, "Verified at: %s\n", valueOrDash(skill.Metadata.VerifiedAt))
	fmt.Fprintf(os.Stdout, "Sandbox timeout: %ds\n", skill.Spec.Sandbox.TimeoutSeconds)
	fmt.Fprintf(os.Stdout, "Sandbox max output bytes: %d\n", skill.Spec.Sandbox.MaxOutputBytes)
	fmt.Fprintf(os.Stdout, "Sandbox network: %s\n", boolToYesNo(derefBool(skill.Spec.Sandbox.AllowNetwork, true)))
	fmt.Fprintf(os.Stdout, "Sandbox filesystem: %s\n", boolToYesNo(derefBool(skill.Spec.Sandbox.AllowFilesystem, true)))
	fmt.Fprintf(os.Stdout, "Sandbox allowed paths: %s\n", joinOrDash(skill.Spec.Sandbox.AllowedPaths))
	if len(skill.Spec.Examples) > 0 {
		fmt.Fprintln(os.Stdout, "Examples:")
		for i, example := range skill.Spec.Examples {
			fmt.Fprintf(os.Stdout, "  %d. prompt=%q args=[%s] output=%q\n", i+1, example.Prompt, strings.Join(example.Args, " "), example.Output)
		}
	}
	fmt.Fprintf(os.Stdout, "Backups: %d\n", backupCount)

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

func SkillsUpdateHandler(cmd *cobra.Command, args []string) error {
	source, err := cmd.Flags().GetString("source")
	if err != nil {
		return err
	}
	toRef, err := cmd.Flags().GetString("to")
	if err != nil {
		return err
	}

	updated, err := skills.Update(args[0], source, toRef)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Updated skill '%s'.\n", updated.Spec.Name)
	if updated.Metadata.Commit != "" {
		fmt.Fprintf(os.Stdout, "Pinned commit: %s\n", updated.Metadata.Commit)
	}

	return nil
}

func SkillsVerifyHandler(cmd *cobra.Command, args []string) error {
	verified, err := skills.Verify(args[0])
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Skill: %s\n", verified.Skill)
	fmt.Fprintf(os.Stdout, "Digest: %s\n", valueOrDash(verified.Digest))
	fmt.Fprintf(os.Stdout, "Verified: %s\n", boolToYesNo(verified.Verified))
	fmt.Fprintf(os.Stdout, "Signed: %s\n", boolToYesNo(verified.Signed))
	fmt.Fprintf(os.Stdout, "Public key: %s\n", valueOrDash(verified.PublicKey))
	fmt.Fprintf(os.Stdout, "Declared sha256: %s\n", valueOrDash(verified.Provenance.SHA256))
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

func SkillsUninstallHandler(cmd *cobra.Command, args []string) error {
	if err := skills.Uninstall(args[0]); err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Uninstalled skill '%s'.\n", args[0])
	return nil
}

func SkillsRollbackHandler(cmd *cobra.Command, args []string) error {
	rolledBack, err := skills.Rollback(args[0])
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Rolled back skill '%s'.\n", rolledBack.Spec.Name)
	if rolledBack.Metadata.Commit != "" {
		fmt.Fprintf(os.Stdout, "Pinned commit: %s\n", rolledBack.Metadata.Commit)
	}
	return nil
}

func SkillsAllowHandler(cmd *cobra.Command, args []string) error {
	grants, err := skills.Allow(args[0], args[1:])
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Granted permissions for '%s': %s\n", args[0], joinOrDash(grants))
	return nil
}

func SkillsRevokeHandler(cmd *cobra.Command, args []string) error {
	grants, err := skills.Revoke(args[0], args[1:])
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Remaining permissions for '%s': %s\n", args[0], joinOrDash(grants))
	return nil
}

func SkillsLogsHandler(cmd *cobra.Command, args []string) error {
	last, err := cmd.Flags().GetInt("last")
	if err != nil {
		return err
	}

	lines, err := skills.ReadLogs(args[0], last)
	if err != nil {
		return err
	}

	if len(lines) == 0 {
		fmt.Fprintf(os.Stdout, "No logs for '%s'.\n", args[0])
		return nil
	}

	for _, line := range lines {
		fmt.Fprintln(os.Stdout, line)
	}
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
	scope, err := cmd.Flags().GetString("scope")
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
	yesToPrompt, err := cmd.Flags().GetBool("yes")
	if err != nil {
		return err
	}

	if err := applyScope(name, allow, scope); err != nil {
		return err
	}

	opts := skills.RunOptions{
		GrantedPermissions: allow,
		DryRun:             dryRun,
		Timeout:            time.Duration(timeoutSeconds) * time.Second,
		MaxOutputBytes:     maxOutputBytes,
	}

	start := time.Now()
	err = skills.RunWithOptions(cmd.Context(), name, runArgs, os.Stdin, os.Stdout, os.Stderr, opts)
	if err != nil && !dryRun {
		additional, decisionErr := maybePromptForPermissions(name, err, yesToPrompt)
		if decisionErr == nil {
			opts.GrantedPermissions = append(opts.GrantedPermissions, additional...)
			err = skills.RunWithOptions(cmd.Context(), name, runArgs, os.Stdin, os.Stdout, os.Stderr, opts)
		}
	}
	duration := time.Since(start).Round(time.Millisecond)

	if dryRun && err == nil {
		fmt.Fprintf(os.Stdout, "Trace: skill=%s dry_run=yes duration=%s\n", name, duration)
		return nil
	}
	if err != nil {
		return err
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
	return fmt.Sprintf("%s: %s (inputs: %s; outputs: %s; required: %s; granted: %s)",
		skill.Spec.Name,
		valueOrDash(skill.Spec.Description),
		joinOrDash(skill.Spec.IO.Inputs),
		joinOrDash(skill.Spec.IO.Outputs),
		joinOrDash(skill.Spec.Permissions.Required),
		joinOrDash(skill.GrantedPermissions),
	)
}

func printSkillsTable(all []skills.Skill) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ENABLED", "TAGS", "INPUTS", "OUTPUTS", "REQUIRED", "GRANTED", "DESCRIPTION", "SOURCE"})
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
			joinOrDash(skill.GrantedPermissions),
			valueOrDash(skill.Spec.Description),
			valueOrDash(skill.Metadata.Source),
		})
	}
	table.Render()
}

func printCatalogTable(all []skills.CatalogEntry) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "VERIFIED", "TAGS", "PERMISSIONS", "DESCRIPTION", "SOURCE", "UPDATED"})
	table.SetAutoWrapText(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	for _, entry := range all {
		table.Append([]string{
			entry.Name,
			boolToYesNo(entry.Verified),
			joinOrDash(entry.Tags),
			joinOrDash(entry.Permissions),
			valueOrDash(entry.Description),
			valueOrDash(entry.Source),
			valueOrDash(entry.UpdatedAt),
		})
	}
	table.Render()
}

func filterLocalSkills(all []skills.Skill, tags []string, permissions []string, verifiedOnly bool) []skills.Skill {
	tagSet := make(map[string]bool)
	for _, tag := range tags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag != "" {
			tagSet[tag] = true
		}
	}
	permSet := make(map[string]bool)
	for _, permission := range permissions {
		permission = strings.TrimSpace(permission)
		if permission != "" {
			permSet[permission] = true
		}
	}

	filtered := make([]skills.Skill, 0, len(all))
	for _, skill := range all {
		if verifiedOnly && !localSkillVerified(skill) {
			continue
		}
		if len(tagSet) > 0 {
			skillTags := map[string]bool{}
			for _, tag := range skill.Spec.Tags {
				skillTags[strings.ToLower(tag)] = true
			}
			match := true
			for tag := range tagSet {
				if !skillTags[tag] {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}
		if len(permSet) > 0 {
			skillPerms := map[string]bool{}
			for _, permission := range skill.Spec.Permissions.Required {
				skillPerms[permission] = true
			}
			match := true
			for permission := range permSet {
				if !skillPerms[permission] {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}
		filtered = append(filtered, skill)
	}
	return filtered
}

func localSkillVerified(skill skills.Skill) bool {
	return skill.Metadata.Verified
}

func applyScope(name string, permissions []string, scope string) error {
	if len(permissions) == 0 {
		return nil
	}

	scope = strings.ToLower(strings.TrimSpace(scope))
	if scope == "" {
		scope = "once"
	}
	switch scope {
	case "once":
		return nil
	case "chat":
		_, err := skills.AllowSession(name, permissions)
		return err
	case "always":
		_, err := skills.Allow(name, permissions)
		return err
	default:
		return fmt.Errorf("invalid --scope %q, expected once|chat|always", scope)
	}
}

func maybePromptForPermissions(name string, runErr error, autoYes bool) ([]string, error) {
	missing := skills.MissingPermissions(runErr)
	if len(missing) == 0 {
		return nil, runErr
	}

	if autoYes {
		_, err := skills.Allow(name, missing)
		return nil, err
	}

	fmt.Fprintf(os.Stderr, "Skill '%s' requires additional permissions: %s\n", name, strings.Join(missing, ", "))
	fmt.Fprint(os.Stderr, "Grant scope? [once/chat/always/deny] (default: deny): ")

	reader := bufio.NewReader(os.Stdin)
	choice, err := reader.ReadString('\n')
	if err != nil {
		return nil, runErr
	}
	choice = strings.ToLower(strings.TrimSpace(choice))
	if choice == "" || choice == "deny" {
		return nil, runErr
	}

	switch choice {
	case "once":
		// Caller retries with explicit --allow values.
		return missing, nil
	case "chat":
		_, err := skills.AllowSession(name, missing)
		return nil, err
	case "always":
		_, err := skills.Allow(name, missing)
		return nil, err
	default:
		return nil, runErr
	}
}

func formatFieldSchema(fields []skills.FieldSpec) string {
	if len(fields) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(fields))
	for _, field := range fields {
		required := ""
		if field.Required {
			required = "!"
		}
		parts = append(parts, fmt.Sprintf("%s:%s%s", field.Name, field.Type, required))
	}
	return strings.Join(parts, ", ")
}

func derefBool(v *bool, fallback bool) bool {
	if v == nil {
		return fallback
	}
	return *v
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
