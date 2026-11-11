package config

import (
	"encoding/json"
	"errors"
	"slices"
)

// SwapPrimaryModelSelection changes only models and primary, if they still
// match before. Callers can swap the snapshots back for conditional rollback.
// Other integrations, aliases, onboarding, and unknown fields remain untouched.
func SwapPrimaryModelSelection(name string, before, after *IntegrationConfig) error {
	return updateIntegrationFields(name, func(fields map[string]json.RawMessage, original json.RawMessage) error {
		var current, expected IntegrationConfig
		if len(original) > 0 {
			if err := json.Unmarshal(original, &current); err != nil {
				return err
			}
		}
		if before != nil {
			expected = *before
		}
		if !slices.Equal(current.Models, expected.Models) || current.Primary != expected.Primary {
			return errors.New("model selection changed after this attempt; preserving the newer selection")
		}
		if after == nil {
			delete(fields, "models")
			delete(fields, "primary")
			return nil
		}
		var err error
		fields["models"], err = json.Marshal(after.Models)
		if err != nil {
			return err
		}
		if after.Primary == "" {
			delete(fields, "primary")
		} else {
			fields["primary"], err = json.Marshal(after.Primary)
		}
		return err
	})
}
