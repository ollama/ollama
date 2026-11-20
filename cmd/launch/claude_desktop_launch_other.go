//go:build !darwin

package launch

import "context"

func (c *launcherClient) launchClaudeDesktop(_ context.Context, desktop *ClaudeDesktop, _ IntegrationLaunchRequest) error {
	return desktop.Supported()
}
