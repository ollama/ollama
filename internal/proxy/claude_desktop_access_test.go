package proxy

import "testing"

func TestEvaluateClaudeDesktopModelAccess(t *testing.T) {
	local := newClaudeDesktopModel("qwen3:8b", "", "", 0)
	free := newClaudeDesktopModel("gemma4:31b-cloud", "", "free", 0)
	pro := newClaudeDesktopModel("glm-5.2:cloud", "", "pro", 0)

	tests := []struct {
		name      string
		model     ClaudeDesktopModel
		state     ClaudeDesktopAccessState
		installed bool
		want      ClaudeDesktopModelAccess
	}{
		{name: "installed local", model: local, installed: true, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityAvailable}},
		{name: "missing local", model: local, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnavailable, Reason: ClaudeDesktopAccessModelNotInstalled}},
		{name: "cloud off", model: free, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOff}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnavailable, Reason: ClaudeDesktopAccessCloudOff, RequiredPlan: "free"}},
		{name: "cloud unknown", model: free, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudUnknown}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnknown, Reason: ClaudeDesktopAccessVerificationUnavailable, RequiredPlan: "free"}},
		{name: "signed out", model: free, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedOut}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnavailable, Reason: ClaudeDesktopAccessSignInRequired, RequiredPlan: "free"}},
		{name: "account unknown", model: free, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountUnknown}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnknown, Reason: ClaudeDesktopAccessVerificationUnavailable, RequiredPlan: "free"}},
		{name: "free account free model", model: free, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityAvailable, RequiredPlan: "free"}},
		{name: "free account pro model", model: pro, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityUnavailable, Reason: ClaudeDesktopAccessUpgradeRequired, RequiredPlan: "pro"}},
		{name: "pro account pro model", model: pro, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "pro"}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityAvailable, RequiredPlan: "pro"}},
		{name: "team account pro model", model: pro, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "team"}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityAvailable, RequiredPlan: "pro"}},
		{name: "future paid plan", model: pro, state: ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "enterprise"}, want: ClaudeDesktopModelAccess{Availability: ClaudeDesktopAvailabilityAvailable, RequiredPlan: "pro"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EvaluateClaudeDesktopModelAccess(tt.model, tt.state, tt.installed, true); got != tt.want {
				t.Fatalf("EvaluateClaudeDesktopModelAccess() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestEvaluateClaudeDesktopModelAccessUnknownLocalInventory(t *testing.T) {
	model := newClaudeDesktopModel("qwen3:8b", "", "", 0)
	want := ClaudeDesktopModelAccess{
		Availability: ClaudeDesktopAvailabilityUnknown,
		Reason:       ClaudeDesktopAccessVerificationUnavailable,
	}
	if got := EvaluateClaudeDesktopModelAccess(model, ClaudeDesktopAccessState{}, false, false); got != want {
		t.Fatalf("EvaluateClaudeDesktopModelAccess() = %+v, want %+v", got, want)
	}
}

func TestEvaluateClaudeDesktopModelAccessUnknownCloudEntitlement(t *testing.T) {
	selected := SelectClaudeDesktopModels(DefaultClaudeDesktopModels(), []string{"retired-pro-model:cloud"})
	if len(selected) != 1 {
		t.Fatalf("selected models = %d, want 1", len(selected))
	}
	want := ClaudeDesktopModelAccess{
		Availability: ClaudeDesktopAvailabilityUnknown,
		Reason:       ClaudeDesktopAccessVerificationUnavailable,
	}
	state := ClaudeDesktopAccessState{
		Cloud:   ClaudeDesktopCloudOn,
		Account: ClaudeDesktopAccountSignedIn,
		Plan:    "free",
	}
	if got := EvaluateClaudeDesktopModelAccess(selected[0], state, false, true); got != want {
		t.Fatalf("EvaluateClaudeDesktopModelAccess() = %+v, want %+v", got, want)
	}
}
