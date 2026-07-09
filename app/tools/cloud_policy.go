//go:build windows || darwin

package tools

import (
	"context"
	"errors"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
)

// ensureCloudEnabledForTool checks cloud policy from the connected Ollama server.
// It only blocks when cloud is disabled via the OLLAMA_NO_CLOUD environment variable
// (source "env" or "both"), allowing web search/fetch to work when cloud is disabled
// via the user-facing toggle alone (source "config").
// If policy cannot be determined, this fails closed and blocks the operation.
func ensureCloudEnabledForTool(ctx context.Context, operation string) error {
	disabledMessage := internalcloud.DisabledError(operation)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return errors.New(disabledMessage + " (unable to verify server cloud policy)")
	}

	status, err := client.CloudStatusExperimental(ctx)
	if err != nil {
		return errors.New(disabledMessage + " (unable to verify server cloud policy)")
	}

	if status.Cloud.Disabled && (status.Cloud.Source == "env" || status.Cloud.Source == "both") {
		return errors.New(disabledMessage)
	}

	return nil
}
