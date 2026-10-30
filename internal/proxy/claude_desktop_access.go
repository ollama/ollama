package proxy

import "strings"

type ClaudeDesktopCloudStatus string

const (
	ClaudeDesktopCloudUnknown ClaudeDesktopCloudStatus = "unknown"
	ClaudeDesktopCloudOff     ClaudeDesktopCloudStatus = "off"
	ClaudeDesktopCloudOn      ClaudeDesktopCloudStatus = "on"
)

type ClaudeDesktopAccountStatus string

const (
	ClaudeDesktopAccountUnknown   ClaudeDesktopAccountStatus = "unknown"
	ClaudeDesktopAccountSignedOut ClaudeDesktopAccountStatus = "signed_out"
	ClaudeDesktopAccountSignedIn  ClaudeDesktopAccountStatus = "signed_in"
)

// ClaudeDesktopAccessState is the current account and Cloud state used to
// evaluate Claude Desktop models. Callers should resolve it again at each user
// action or proxy request because both settings and account state can change.
type ClaudeDesktopAccessState struct {
	Cloud   ClaudeDesktopCloudStatus
	Account ClaudeDesktopAccountStatus
	Plan    string
}

type ClaudeDesktopAvailability string

const (
	ClaudeDesktopAvailabilityUnknown     ClaudeDesktopAvailability = "unknown"
	ClaudeDesktopAvailabilityAvailable   ClaudeDesktopAvailability = "available"
	ClaudeDesktopAvailabilityUnavailable ClaudeDesktopAvailability = "unavailable"
)

type ClaudeDesktopAccessReason string

const (
	ClaudeDesktopAccessCloudOff                ClaudeDesktopAccessReason = "cloud_off"
	ClaudeDesktopAccessSignInRequired          ClaudeDesktopAccessReason = "sign_in_required"
	ClaudeDesktopAccessUpgradeRequired         ClaudeDesktopAccessReason = "upgrade_required"
	ClaudeDesktopAccessVerificationUnavailable ClaudeDesktopAccessReason = "verification_unavailable"
	ClaudeDesktopAccessModelNotInstalled       ClaudeDesktopAccessReason = "model_not_installed"
)

// ClaudeDesktopModelAccess is the policy decision for one model.
type ClaudeDesktopModelAccess struct {
	Availability ClaudeDesktopAvailability
	Reason       ClaudeDesktopAccessReason
	RequiredPlan string
}

// EvaluateClaudeDesktopModelAccess applies Claude Desktop's model policy to
// facts resolved by the app. installed and inventoryKnown only affect local
// models.
func EvaluateClaudeDesktopModelAccess(model ClaudeDesktopModel, state ClaudeDesktopAccessState, installed, inventoryKnown bool) ClaudeDesktopModelAccess {
	decision := ClaudeDesktopModelAccess{RequiredPlan: strings.TrimSpace(model.RequiredPlan)}
	if !model.Cloud {
		if !inventoryKnown {
			decision.Availability = ClaudeDesktopAvailabilityUnknown
			decision.Reason = ClaudeDesktopAccessVerificationUnavailable
			return decision
		}
		if installed {
			decision.Availability = ClaudeDesktopAvailabilityAvailable
			return decision
		}
		decision.Availability = ClaudeDesktopAvailabilityUnavailable
		decision.Reason = ClaudeDesktopAccessModelNotInstalled
		return decision
	}

	if state.Cloud == ClaudeDesktopCloudOff {
		decision.Availability = ClaudeDesktopAvailabilityUnavailable
		decision.Reason = ClaudeDesktopAccessCloudOff
		return decision
	}
	// A failed status check must not expose models that the local server may be
	// configured to reject. The next discovery or request evaluates again.
	if state.Cloud != ClaudeDesktopCloudOn {
		decision.Availability = ClaudeDesktopAvailabilityUnknown
		decision.Reason = ClaudeDesktopAccessVerificationUnavailable
		return decision
	}
	if state.Account == ClaudeDesktopAccountSignedOut {
		decision.Availability = ClaudeDesktopAvailabilityUnavailable
		decision.Reason = ClaudeDesktopAccessSignInRequired
		return decision
	}
	// Treat account lookup failures as recoverable instead of guessing that a
	// user is signed out or entitled to a paid model.
	if state.Account != ClaudeDesktopAccountSignedIn {
		decision.Availability = ClaudeDesktopAvailabilityUnknown
		decision.Reason = ClaudeDesktopAccessVerificationUnavailable
		return decision
	}
	if !model.entitlementKnown {
		decision.Availability = ClaudeDesktopAvailabilityUnknown
		decision.Reason = ClaudeDesktopAccessVerificationUnavailable
		return decision
	}
	if !claudeDesktopPlanSatisfies(state.Plan, decision.RequiredPlan) {
		decision.Availability = ClaudeDesktopAvailabilityUnavailable
		decision.Reason = ClaudeDesktopAccessUpgradeRequired
		return decision
	}

	decision.Availability = ClaudeDesktopAvailabilityAvailable
	return decision
}

func claudeDesktopPlanSatisfies(plan, required string) bool {
	plan = strings.ToLower(strings.TrimSpace(plan))
	required = strings.ToLower(strings.TrimSpace(required))
	// The recommendation contract currently has two access levels. Empty and
	// "free" are available to every signed-in account; any other requirement
	// needs a non-free account until the server exposes an ordered tier model.
	if required == "" || required == "free" {
		return true
	}
	return plan != "" && plan != "free"
}
