package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/secrets"
	"github.com/ollama/ollama/internal/secrets/github"
)

// Example program demonstrating GSM and GitHub integration with Ollama.
// This is a reference implementation showing how to use the secrets package.
//
// Usage:
//   OLLAMA_GSM_ENABLED=true OLLAMA_GSM_PROJECT_ID=my-project go run examples/secrets_integration.go
//   OLLAMA_GITHUB_TOKEN=ghp_xxx go run examples/secrets_integration.go

func main() {
	ctx := context.Background()

	// Parse command-line flags
	useGSM := flag.Bool("gsm", envconfig.GSMEnabled(), "Use Google Secret Manager")
	gsmProjectID := flag.String("gsm-project", envconfig.GSMProjectID(), "Google Cloud project ID")
	testGitHub := flag.Bool("test-github", false, "Test GitHub integration")
	flag.Parse()

	// Example 1: Google Secret Manager Integration
	if *useGSM && *gsmProjectID != "" {
		fmt.Println("=== Google Secret Manager Integration ===")
		if err := demonstrateGSM(ctx, *gsmProjectID); err != nil {
			log.Printf("GSM demo error: %v\n", err)
		}
		fmt.Println()
	}

	// Example 2: GitHub Integration
	if *testGitHub {
		fmt.Println("=== GitHub Integration ===")
		if err := demonstrateGitHub(ctx); err != nil {
			log.Printf("GitHub demo error: %v\n", err)
		}
		fmt.Println()
	}

	// Example 3: Combined flow
	fmt.Println("=== Combined Workflow ===")
	if err := demonstrateCombinedFlow(ctx, *useGSM, *gsmProjectID); err != nil {
		log.Printf("Combined workflow error: %v\n", err)
	}
}

// demonstrateGSM shows how to use Google Secret Manager.
func demonstrateGSM(ctx context.Context, projectID string) error {
	gsmConfig := &secrets.GSMConfig{
		ProjectID: projectID,
		Enabled:   true,
	}

	client, err := secrets.NewGSMClient(gsmConfig)
	if err != nil {
		return fmt.Errorf("failed to create GSM client: %w", err)
	}
	defer client.Close()

	fmt.Printf("✓ Created GSM client for project: %s\n", projectID)

	// To test this, you need to have a secret in GSM
	// Example: gcloud secrets create test-secret --data-file=- <<< "test-value"
	// For this demo, we just show the structure
	fmt.Println("  Ready to retrieve secrets from Google Secret Manager")
	fmt.Println("  Example usage in code:")
	fmt.Println("    secret, err := client.GetSecret(ctx, \"my-secret\")")
	fmt.Println("    secretV2, err := client.GetSecretVersion(ctx, \"my-secret\", \"2\")")

	return nil
}

// demonstrateGitHub shows how to use GitHub integration.
func demonstrateGitHub(ctx context.Context) error {
	token := envconfig.GitHubToken()
	if token == "" {
		fmt.Println("⚠ OLLAMA_GITHUB_TOKEN not set")
		fmt.Println("  Set environment variable: export OLLAMA_GITHUB_TOKEN=your_token")
		fmt.Println("  To generate a token: https://github.com/settings/tokens")
		return nil
	}

	// Validate token format
	token = github.TokenFromString(token)
	fmt.Printf("✓ GitHub token configured (length: %d)\n", len(token))

	// Create client
	client := github.NewClient(token)

	// Get authenticated user
	user, err := client.GetAuthenticatedUser(ctx)
	if err != nil {
		return fmt.Errorf("failed to get user: %w", err)
	}

	fmt.Printf("✓ Authenticated as: %s (ID: %d)\n", user.Login, user.ID)
	if user.Name != "" {
		fmt.Printf("  Name: %s\n", user.Name)
	}
	if user.Email != "" {
		fmt.Printf("  Email: %s\n", user.Email)
	}

	// Get repository information
	repo, err := client.GetRepository(ctx, "ollama", "ollama")
	if err != nil {
		return fmt.Errorf("failed to get repository: %w", err)
	}

	fmt.Printf("✓ Repository accessed: %s\n", repo.FullName)
	fmt.Printf("  URL: %s\n", repo.URL)
	fmt.Printf("  Private: %v\n", repo.Private)
	if repo.Description != "" {
		fmt.Printf("  Description: %s\n", repo.Description)
	}

	return nil
}

// demonstrateCombinedFlow shows how to use both GSM and GitHub together.
func demonstrateCombinedFlow(ctx context.Context, useGSM bool, projectID string) error {
	fmt.Println("Typical workflow:")
	fmt.Println("1. Retrieve GitHub token from GSM (if GSM is enabled)")
	fmt.Println("2. Use GitHub token to authenticate with GitHub API")
	fmt.Println("3. Perform GitHub operations (clone, push, etc.)")
	fmt.Println()

	if useGSM && projectID != "" {
		fmt.Printf("GSM Enabled: Project '%s'\n", projectID)
		fmt.Println("  Example: github_token := gsmClient.GetSecret(ctx, \"github-token\")")
	} else {
		fmt.Println("GSM Disabled")
		fmt.Println("  Example: github_token := os.Getenv(\"OLLAMA_GITHUB_TOKEN\")")
	}

	fmt.Println()
	fmt.Println("GitHub Integration: Ready")
	fmt.Println("  Example: client := github.NewClient(github_token)")
	fmt.Println("  Example: user, _ := client.GetAuthenticatedUser(ctx)")

	return nil
}
