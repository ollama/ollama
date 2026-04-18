// Package secrets provides secret management functionality including Google Secret Manager integration.
package secrets

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

// GSMConfig holds configuration for Google Secret Manager.
type GSMConfig struct {
	ProjectID string
	Enabled   bool
}

// GSMClient wraps the Google Secret Manager client.
type GSMClient struct {
	config *GSMConfig
	client *secretmanager.Client
}

var (
	// ErrGSMDisabled is returned when GSM is not enabled.
	ErrGSMDisabled = errors.New("google secret manager is disabled")
	// ErrGSMNotConfigured is returned when GSM is not properly configured.
	ErrGSMNotConfigured = errors.New("google secret manager is not configured")
)

// NewGSMClient creates a new GSM client with the given configuration.
// It does not establish a connection immediately; the connection is created on first use.
func NewGSMClient(config *GSMConfig) (*GSMClient, error) {
	if !config.Enabled {
		return nil, ErrGSMDisabled
	}

	if config.ProjectID == "" {
		return nil, ErrGSMNotConfigured
	}

	return &GSMClient{
		config: config,
		client: nil, // will be initialized lazily
	}, nil
}

// GetSecret retrieves a secret value from Google Secret Manager.
// It returns the latest version of the secret.
func (g *GSMClient) GetSecret(ctx context.Context, secretName string) (string, error) {
	if !g.config.Enabled {
		return "", ErrGSMDisabled
	}

	if g.client == nil {
		client, err := secretmanager.NewClient(ctx)
		if err != nil {
			slog.Error("failed to create secret manager client", "error", err)
			return "", fmt.Errorf("failed to create secret manager client: %w", err)
		}
		g.client = client
	}

	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: fmt.Sprintf("projects/%s/secrets/%s/versions/latest", g.config.ProjectID, secretName),
	}

	result, err := g.client.AccessSecretVersion(ctx, req)
	if err != nil {
		slog.Error("failed to access secret version", "secret", secretName, "error", err)
		return "", fmt.Errorf("failed to access secret %q: %w", secretName, err)
	}

	return string(result.Payload.Data), nil
}

// GetSecretVersion retrieves a specific version of a secret from Google Secret Manager.
func (g *GSMClient) GetSecretVersion(ctx context.Context, secretName, version string) (string, error) {
	if !g.config.Enabled {
		return "", ErrGSMDisabled
	}

	if g.client == nil {
		client, err := secretmanager.NewClient(ctx)
		if err != nil {
			slog.Error("failed to create secret manager client", "error", err)
			return "", fmt.Errorf("failed to create secret manager client: %w", err)
		}
		g.client = client
	}

	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: fmt.Sprintf("projects/%s/secrets/%s/versions/%s", g.config.ProjectID, secretName, version),
	}

	result, err := g.client.AccessSecretVersion(ctx, req)
	if err != nil {
		slog.Error("failed to access secret version", "secret", secretName, "version", version, "error", err)
		return "", fmt.Errorf("failed to access secret %q version %q: %w", secretName, version, err)
	}

	return string(result.Payload.Data), nil
}

// Close closes the GSM client connection.
func (g *GSMClient) Close() error {
	if g.client != nil {
		return g.client.Close()
	}
	return nil
}
