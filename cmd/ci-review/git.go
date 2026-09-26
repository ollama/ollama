package main

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

func fetchHead(ctx context.Context, number int, head string) error {
	if number <= 0 || !validSHA(head) {
		return fmt.Errorf("invalid pull request head")
	}
	ref := pullHeadRefspec(number)
	cmd := exec.CommandContext(ctx, "git", "fetch", "--no-tags", "--depth=1", "origin", ref)
	cmd.Env = append(cmd.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("fetch pull request head: %w: %s", err, strings.TrimSpace(string(output)))
	}
	got, err := gitOutput(ctx, "rev-parse", "refs/ollama-ci-review/head")
	if err != nil {
		return err
	}
	if strings.TrimSpace(got) != head {
		return fmt.Errorf("fetched pull request head does not match event head")
	}
	return nil
}

func pullHeadRefspec(number int) string {
	// This runner reuses one private destination ref for every PR. Force it so a
	// different PR head cannot be rejected as a non-fast-forward update.
	return fmt.Sprintf("+refs/pull/%d/head:refs/ollama-ci-review/head", number)
}

func validSHA(value string) bool {
	if len(value) != 40 {
		return false
	}
	for _, r := range value {
		if !(r >= '0' && r <= '9') && !(r >= 'a' && r <= 'f') {
			return false
		}
	}
	return true
}
