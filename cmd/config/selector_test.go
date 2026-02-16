package config

import (
	"testing"
)

func TestErrCancelled(t *testing.T) {
	t.Run("NotNil", func(t *testing.T) {
		if errCancelled == nil {
			t.Error("errCancelled should not be nil")
		}
	})

	t.Run("Message", func(t *testing.T) {
		if errCancelled.Error() != "cancelled" {
			t.Errorf("expected 'cancelled', got %q", errCancelled.Error())
		}
	})
}
