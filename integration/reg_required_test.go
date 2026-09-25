//go:build integration && !fast && !release && !library && !create

package integration

import "testing"

func TestIntegrationRequiresScope(t *testing.T) {
	t.Fatal("integration tests require one of the fast, release, library, or create tags")
}
