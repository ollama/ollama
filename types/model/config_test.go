package model

import (
	"slices"
	"testing"
)

func TestSetCapabilities(t *testing.T) {
	for _, tt := range []struct {
		name     string
		base     ConfigV2
		declared []string
		want     []string
		explicit bool
	}{
		{"infer when omitted", ConfigV2{Capabilities: []string{"completion", "vision"}}, nil, []string{"completion", "vision"}, false},
		{"inherit explicit", ConfigV2{Capabilities: []string{"decision"}, CapabilitiesExplicit: true}, nil, []string{"decision"}, true},
		{"replace inference", ConfigV2{Capabilities: []string{"completion", "vision"}}, []string{"decision"}, []string{"decision"}, true},
		{"replace inherited declaration", ConfigV2{Capabilities: []string{"decision", "vision"}, CapabilitiesExplicit: true}, []string{"completion", "decision", "completion"}, []string{"completion", "decision"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tt.base.SetCapabilities(tt.declared...)
			if !slices.Equal(tt.base.Capabilities, tt.want) || tt.base.CapabilitiesExplicit != tt.explicit {
				t.Fatalf("capabilities = %v (explicit %t), want %v (explicit %t)", tt.base.Capabilities, tt.base.CapabilitiesExplicit, tt.want, tt.explicit)
			}
		})
	}
}
