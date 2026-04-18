package secrets

import (
	"context"
	"testing"
)

func TestGSMConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		config  *GSMConfig
		wantErr string
	}{
		{
			name: "disabled_gsm",
			config: &GSMConfig{
				Enabled:   false,
				ProjectID: "project-123",
			},
			wantErr: "google secret manager is disabled",
		},
		{
			name: "missing_project_id",
			config: &GSMConfig{
				Enabled:   true,
				ProjectID: "",
			},
			wantErr: "google secret manager is not configured",
		},
		{
			name: "valid_config",
			config: &GSMConfig{
				Enabled:   true,
				ProjectID: "project-123",
			},
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGSMClient(tt.config)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Errorf("got %v, want %s", err, tt.wantErr)
				}
			} else if err != nil {
				t.Errorf("got unexpected error: %v", err)
			}
		})
	}
}

func TestGSMClientDisabledGetSecret(t *testing.T) {
	client := &GSMClient{
		config: &GSMConfig{Enabled: false},
	}

	_, err := client.GetSecret(context.Background(), "test-secret")
	if err != ErrGSMDisabled {
		t.Errorf("got %v, want %v", err, ErrGSMDisabled)
	}
}

func TestGSMClientDisabledGetSecretVersion(t *testing.T) {
	client := &GSMClient{
		config: &GSMConfig{Enabled: false},
	}

	_, err := client.GetSecretVersion(context.Background(), "test-secret", "1")
	if err != ErrGSMDisabled {
		t.Errorf("got %v, want %v", err, ErrGSMDisabled)
	}
}
