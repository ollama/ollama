//go:build integration && !fast && !release && !library && !imagegen

package integration

import "testing"

func TestIntegrationRequiresScope(t *testing.T) {
	t.Fatal("integration tests require one of the fast, release, or library tags")
}
