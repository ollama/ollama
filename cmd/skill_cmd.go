package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
)

// SkillPushHandler handles the skill push command.
func SkillPushHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: ollama skill push NAME[:TAG] PATH")
	}

	name := args[0]
	path := args[1]

	// Expand path
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("expanding home directory: %w", err)
		}
		path = filepath.Join(home, path[1:])
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolving path: %w", err)
	}

	// Validate skill directory
	skillMdPath := filepath.Join(absPath, "SKILL.md")
	if _, err := os.Stat(skillMdPath); err != nil {
		return fmt.Errorf("skill directory must contain SKILL.md: %w", err)
	}

	// Parse skill name (will set Kind="skill")
	n := server.ParseSkillName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid skill name: %s", name)
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	// Create skill layer
	displayName := n.DisplayShortest()
	status := fmt.Sprintf("Creating skill layer for %s", displayName)
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	layer, err := server.CreateSkillLayer(absPath)
	if err != nil {
		return fmt.Errorf("creating skill layer: %w", err)
	}

	spinner.Stop()

	// Create skill manifest
	manifest, configLayer, err := createSkillManifest(absPath, layer)
	if err != nil {
		return fmt.Errorf("creating skill manifest: %w", err)
	}

	// Write manifest locally
	manifestPath, err := server.GetSkillManifestPath(n)
	if err != nil {
		return fmt.Errorf("getting manifest path: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return fmt.Errorf("creating manifest directory: %w", err)
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshaling manifest: %w", err)
	}

	if err := os.WriteFile(manifestPath, manifestJSON, 0o644); err != nil {
		return fmt.Errorf("writing manifest: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Skill %s created locally\n", displayName)
	fmt.Fprintf(os.Stderr, "  Config: %s (%s)\n", configLayer.Digest, format.HumanBytes(configLayer.Size))
	fmt.Fprintf(os.Stderr, "  Layer:  %s (%s)\n", layer.Digest, format.HumanBytes(layer.Size))

	// Push to registry
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("creating client: %w", err)
	}

	insecure, _ := cmd.Flags().GetBool("insecure")

	// For now, we'll use the existing push mechanism
	fmt.Fprintf(os.Stderr, "\nPushing to registry...\n")

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar := progress.NewBar(resp.Status, resp.Total, resp.Completed)
			p.Add(resp.Digest, bar)
		} else if resp.Status != "" {
			spinner := progress.NewSpinner(resp.Status)
			p.Add(resp.Status, spinner)
		}
		return nil
	}

	req := &api.PushRequest{
		Model:    displayName,
		Insecure: insecure,
	}

	if err := client.Push(context.Background(), req, fn); err != nil {
		// If push fails, still show success for local creation
		fmt.Fprintf(os.Stderr, "\nNote: Local skill created but push failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "You can try pushing later with: ollama skill push %s\n", name)
		return nil
	}

	fmt.Fprintf(os.Stderr, "Successfully pushed %s\n", displayName)
	return nil
}

// SkillPullHandler handles the skill pull command.
func SkillPullHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: ollama skill pull NAME[:TAG]")
	}

	name := args[0]
	n := server.ParseSkillName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid skill name: %s", name)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("creating client: %w", err)
	}

	insecure, _ := cmd.Flags().GetBool("insecure")

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar := progress.NewBar(resp.Status, resp.Total, resp.Completed)
			p.Add(resp.Digest, bar)
		} else if resp.Status != "" {
			spinner := progress.NewSpinner(resp.Status)
			p.Add(resp.Status, spinner)
		}
		return nil
	}

	displayName := n.DisplayShortest()
	req := &api.PullRequest{
		Model:    displayName,
		Insecure: insecure,
	}

	if err := client.Pull(context.Background(), req, fn); err != nil {
		return fmt.Errorf("pulling skill: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Successfully pulled %s\n", displayName)
	return nil
}

// SkillListHandler handles the skill list command.
func SkillListHandler(cmd *cobra.Command, args []string) error {
	skills, err := listLocalSkills()
	if err != nil {
		return fmt.Errorf("listing skills: %w", err)
	}

	if len(skills) == 0 {
		fmt.Println("No skills installed")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tTAG\tSIZE\tMODIFIED")

	for _, skill := range skills {
		fmt.Fprintf(w, "%s/%s\t%s\t%s\t%s\n",
			skill.Namespace,
			skill.Name,
			skill.Tag,
			format.HumanBytes(skill.Size),
			format.HumanTime(skill.ModifiedAt, "Never"),
		)
	}

	return w.Flush()
}

// SkillRemoveHandler handles the skill rm command.
func SkillRemoveHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: ollama skill rm NAME[:TAG] [NAME[:TAG]...]")
	}

	for _, name := range args {
		n := server.ParseSkillName(name)
		if n.Model == "" {
			fmt.Fprintf(os.Stderr, "Invalid skill name: %s\n", name)
			continue
		}

		displayName := n.DisplayShortest()
		manifestPath, err := server.GetSkillManifestPath(n)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting manifest path for %s: %v\n", name, err)
			continue
		}

		if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Skill not found: %s\n", displayName)
			continue
		}

		if err := os.Remove(manifestPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error removing %s: %v\n", displayName, err)
			continue
		}

		// Clean up empty parent directories
		dir := filepath.Dir(manifestPath)
		for dir != filepath.Join(os.Getenv("HOME"), ".ollama", "models", "manifests") {
			entries, _ := os.ReadDir(dir)
			if len(entries) == 0 {
				os.Remove(dir)
				dir = filepath.Dir(dir)
			} else {
				break
			}
		}

		fmt.Fprintf(os.Stderr, "Deleted '%s'\n", displayName)
	}

	return nil
}

// SkillShowHandler handles the skill show command.
func SkillShowHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: ollama skill show NAME[:TAG]")
	}

	name := args[0]
	n := server.ParseSkillName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid skill name: %s", name)
	}

	displayName := n.DisplayShortest()
	manifestPath, err := server.GetSkillManifestPath(n)
	if err != nil {
		return fmt.Errorf("getting manifest path: %w", err)
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("skill not found: %s", displayName)
		}
		return fmt.Errorf("reading manifest: %w", err)
	}

	var manifest server.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("parsing manifest: %w", err)
	}

	fmt.Printf("Skill: %s\n\n", displayName)

	fmt.Println("Layers:")
	for _, layer := range manifest.Layers {
		fmt.Printf("  %s  %s  %s\n", layer.MediaType, layer.Digest[:19], format.HumanBytes(layer.Size))
	}

	// Try to read and display SKILL.md content
	if len(manifest.Layers) > 0 {
		for _, layer := range manifest.Layers {
			if layer.MediaType == server.MediaTypeSkill {
				skillPath, err := server.GetSkillsPath(layer.Digest)
				if err == nil {
					skillMdPath := filepath.Join(skillPath, "SKILL.md")
					if content, err := os.ReadFile(skillMdPath); err == nil {
						fmt.Println("\nContent:")
						fmt.Println(string(content))
					}
				}
			}
		}
	}

	return nil
}

// SkillInfo represents information about an installed skill.
type SkillInfo struct {
	Namespace  string
	Name       string
	Tag        string
	Size       int64
	ModifiedAt time.Time
}

// listLocalSkills returns a list of locally installed skills.
// Skills are stored with 5-part paths: host/namespace/kind/model/tag
// where kind is "skill".
func listLocalSkills() ([]SkillInfo, error) {
	manifestsPath := filepath.Join(os.Getenv("HOME"), ".ollama", "models", "manifests")

	var skills []SkillInfo

	// Walk through all registries
	registries, err := os.ReadDir(manifestsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return skills, nil
		}
		return nil, err
	}

	for _, registry := range registries {
		if !registry.IsDir() {
			continue
		}

		// Walk namespaces
		namespaces, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name()))
		if err != nil {
			continue
		}

		for _, namespace := range namespaces {
			if !namespace.IsDir() {
				continue
			}

			// Walk kinds looking for "skill"
			kinds, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name()))
			if err != nil {
				continue
			}

			for _, kind := range kinds {
				if !kind.IsDir() {
					continue
				}

				// Only process skill kind
				if kind.Name() != server.SkillNamespace {
					continue
				}

				// Walk skill names (model names)
				skillNames, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name()))
				if err != nil {
					continue
				}

				for _, skillName := range skillNames {
					if !skillName.IsDir() {
						continue
					}

					// Walk tags
					tags, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name(), skillName.Name()))
					if err != nil {
						continue
					}

					for _, tag := range tags {
						manifestPath := filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name(), skillName.Name(), tag.Name())
						fi, err := os.Stat(manifestPath)
						if err != nil || fi.IsDir() {
							continue
						}

						// Read manifest to get size
						data, err := os.ReadFile(manifestPath)
						if err != nil {
							continue
						}

						var manifest server.Manifest
						if err := json.Unmarshal(data, &manifest); err != nil {
							continue
						}

						var totalSize int64
						for _, layer := range manifest.Layers {
							totalSize += layer.Size
						}

						// Build display name using model.Name
						n := model.Name{
							Host:      registry.Name(),
							Namespace: namespace.Name(),
							Kind:      kind.Name(),
							Model:     skillName.Name(),
							Tag:       tag.Name(),
						}

						skills = append(skills, SkillInfo{
							Namespace:  n.Namespace + "/" + n.Kind,
							Name:       n.Model,
							Tag:        n.Tag,
							Size:       totalSize,
							ModifiedAt: fi.ModTime(),
						})
					}
				}
			}
		}
	}

	return skills, nil
}

// createSkillManifest creates a manifest for a standalone skill.
func createSkillManifest(skillDir string, layer server.Layer) (*server.Manifest, *server.Layer, error) {
	// Read SKILL.md to extract metadata
	skillMdPath := filepath.Join(skillDir, "SKILL.md")
	content, err := os.ReadFile(skillMdPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading SKILL.md: %w", err)
	}

	// Extract name and description from frontmatter
	name, description := extractSkillMetadata(string(content))
	if name == "" {
		return nil, nil, errors.New("skill name not found in SKILL.md frontmatter")
	}

	// Create config
	config := map[string]any{
		"name":         name,
		"description":  description,
		"architecture": "amd64",
		"os":           "linux",
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, nil, fmt.Errorf("marshaling config: %w", err)
	}

	// Create config layer
	configLayer, err := server.NewLayer(strings.NewReader(string(configJSON)), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, nil, fmt.Errorf("creating config layer: %w", err)
	}

	manifest := &server.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        configLayer,
		Layers:        []server.Layer{layer},
	}

	return manifest, &configLayer, nil
}

// extractSkillMetadata extracts name and description from SKILL.md frontmatter.
func extractSkillMetadata(content string) (name, description string) {
	lines := strings.Split(content, "\n")

	inFrontmatter := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		if trimmed == "---" {
			if !inFrontmatter {
				inFrontmatter = true
				continue
			} else {
				break // End of frontmatter
			}
		}

		if inFrontmatter {
			if strings.HasPrefix(trimmed, "name:") {
				name = strings.TrimSpace(strings.TrimPrefix(trimmed, "name:"))
			} else if strings.HasPrefix(trimmed, "description:") {
				description = strings.TrimSpace(strings.TrimPrefix(trimmed, "description:"))
			}
		}
	}

	return name, description
}

// NewSkillCommand creates the skill parent command with subcommands.
func NewSkillCommand() *cobra.Command {
	skillCmd := &cobra.Command{
		Use:   "skill",
		Short: "Manage skills",
		Long:  "Commands for managing agent skills (push, pull, list, rm, show)",
	}

	pushCmd := &cobra.Command{
		Use:     "push NAME[:TAG] PATH",
		Short:   "Push a skill to a registry",
		Long:    "Package a local skill directory and push it to a registry",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    SkillPushHandler,
	}
	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pullCmd := &cobra.Command{
		Use:     "pull NAME[:TAG]",
		Short:   "Pull a skill from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    SkillPullHandler,
	}
	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List installed skills",
		Args:    cobra.NoArgs,
		RunE:    SkillListHandler,
	}

	rmCmd := &cobra.Command{
		Use:     "rm NAME[:TAG] [NAME[:TAG]...]",
		Aliases: []string{"remove", "delete"},
		Short:   "Remove a skill",
		Args:    cobra.MinimumNArgs(1),
		RunE:    SkillRemoveHandler,
	}

	showCmd := &cobra.Command{
		Use:   "show NAME[:TAG]",
		Short: "Show skill details",
		Args:  cobra.ExactArgs(1),
		RunE:  SkillShowHandler,
	}

	skillCmd.AddCommand(pushCmd, pullCmd, listCmd, rmCmd, showCmd)

	return skillCmd
}
