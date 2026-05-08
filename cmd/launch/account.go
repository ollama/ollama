package launch

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	// DefaultUpgradeURL is the fixed destination for subscription upgrades.
	DefaultUpgradeURL = "https://ollama.com/upgrade"

	accountCheckTimeout = 3 * time.Second
)

var (
	ErrPlanVerificationUnavailable = errors.New("Could not verify your plan. Try again in a moment.")
	errUpgradeCancelled            = errors.New("upgrade cancelled")
)

type accountStateStatus int

const (
	accountStateUnknown accountStateStatus = iota
	accountStateSignedOut
	accountStateSignedIn
)

type AccountState struct {
	Status accountStateStatus
	Plan   string
}

type AccountStatePrefetch struct {
	done  chan struct{}
	state AccountState
}

func StartAccountStatePrefetch(ctx context.Context) *AccountStatePrefetch {
	if ctx == nil {
		ctx = context.Background()
	}
	p := &AccountStatePrefetch{done: make(chan struct{})}
	go func() {
		state := AccountState{Status: accountStateUnknown}
		client, err := api.ClientFromEnvironment()
		if err == nil {
			prefetchCtx, cancel := context.WithTimeout(ctx, accountCheckTimeout)
			defer cancel()
			if disabled, known := cloudStatusDisabled(prefetchCtx, client); !known || !disabled {
				state = launchAccountState(prefetchCtx, client)
			}
		}
		p.state = state
		close(p.done)
	}()
	return p
}

func (p *AccountStatePrefetch) StateIfReady() *AccountState {
	if p == nil {
		return nil
	}
	select {
	case <-p.done:
		state := p.state
		return &state
	default:
		return nil
	}
}

func (p *AccountStatePrefetch) StateUpdates(ctx context.Context) <-chan *AccountState {
	if p == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan *AccountState, 1)
	go func() {
		defer close(out)
		select {
		case <-p.done:
			if p.state.Status == accountStateUnknown {
				return
			}
			state := p.state
			select {
			case out <- &state:
			case <-ctx.Done():
			}
		case <-ctx.Done():
		}
	}()
	return out
}

func launchAccountState(ctx context.Context, client *api.Client) AccountState {
	if client == nil {
		return AccountState{Status: accountStateUnknown}
	}

	user, err := whoamiWithTimeout(ctx, client)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized {
			return AccountState{Status: accountStateSignedOut}
		}
		return AccountState{Status: accountStateUnknown}
	}
	if user == nil || strings.TrimSpace(user.Name) == "" {
		return AccountState{Status: accountStateSignedOut}
	}
	return AccountState{
		Status: accountStateSignedIn,
		Plan:   strings.TrimSpace(user.Plan),
	}
}

func whoamiWithTimeout(ctx context.Context, client *api.Client) (*api.UserResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	checkCtx, cancel := context.WithTimeout(ctx, accountCheckTimeout)
	defer cancel()
	return client.Whoami(checkCtx)
}

func ApplyAccountStateToSelectionItems(items []ModelItem, state AccountState) []SelectionItem {
	out := make([]SelectionItem, len(items))
	for i, item := range items {
		out[i] = SelectionItem{
			Name:              item.Name,
			Description:       item.Description,
			Recommended:       item.Recommended,
			AvailabilityBadge: availabilityBadge(item, state),
		}
	}
	return out
}

func SelectionItemsWithAccountState(items []ModelItem, state *AccountState) []SelectionItem {
	if state == nil || !selectionItemsNeedAccountState(items) {
		return ApplyAccountStateToSelectionItems(items, AccountState{Status: accountStateUnknown})
	}
	return ApplyAccountStateToSelectionItems(items, *state)
}

func selectionItemsNeedAccountState(items []ModelItem) bool {
	for _, item := range items {
		if isCloudModelName(item.Name) && itemHasRecommendationMetadata(item) {
			return true
		}
	}
	return false
}

func (c *launcherClient) selectionItemUpdates(ctx context.Context, items []ModelItem, state *AccountState) <-chan []SelectionItem {
	if !selectionItemsNeedAccountState(items) || state != nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	stateUpdates := c.accountStateUpdateSource(ctx)
	if stateUpdates == nil {
		return nil
	}

	out := make(chan []SelectionItem, 1)
	go func() {
		defer close(out)
		select {
		case state, ok := <-stateUpdates:
			if !ok || state == nil {
				return
			}
			select {
			case out <- SelectionItemsWithAccountState(items, state):
			case <-ctx.Done():
			}
		case <-ctx.Done():
		}
	}()
	return out
}

func (c *launcherClient) accountStateUpdateSource(ctx context.Context) <-chan *AccountState {
	if c.accountStateUpdates != nil {
		return c.accountStateUpdates(ctx)
	}
	if c.apiClient == nil {
		return nil
	}
	out := make(chan *AccountState, 1)
	go func() {
		defer close(out)
		state := launchAccountState(ctx, c.apiClient)
		if state.Status == accountStateUnknown {
			return
		}
		select {
		case out <- &state:
		case <-ctx.Done():
		}
	}()
	return out
}

func availabilityBadge(item ModelItem, state AccountState) string {
	if !isCloudModelName(item.Name) {
		return ""
	}
	switch state.Status {
	case accountStateSignedOut:
		if itemHasRecommendationMetadata(item) {
			return "Sign in required"
		}
	case accountStateSignedIn:
		if item.RequiredPlan != "" && !PlanSatisfies(state.Plan, item.RequiredPlan) {
			return "Upgrade required"
		}
	}
	return ""
}

func itemHasRecommendationMetadata(item ModelItem) bool {
	return item.Recommended || strings.TrimSpace(item.RequiredPlan) != ""
}

func (c *launcherClient) ensureCloudModelAccess(ctx context.Context, model string) error {
	item, ok := c.modelRecommendationItem(ctx, model)
	if !ok || strings.TrimSpace(item.RequiredPlan) == "" {
		return nil
	}

	state := launchAccountState(ctx, c.apiClient)
	if state.Status != accountStateUnknown {
		c.accountState = &state
	}
	if state.Status == accountStateUnknown {
		return ErrPlanVerificationUnavailable
	}

	if state.Status == accountStateSignedOut {
		if err := ensureCloudAuth(ctx, c.apiClient, model); err != nil {
			return err
		}
		state = launchAccountState(ctx, c.apiClient)
		if state.Status != accountStateUnknown {
			c.accountState = &state
		}
		if state.Status == accountStateUnknown {
			return ErrPlanVerificationUnavailable
		}
	}

	if PlanSatisfies(state.Plan, item.RequiredPlan) {
		return nil
	}

	if err := c.runUpgradeFlow(ctx, item); err != nil {
		return err
	}
	state = launchAccountState(ctx, c.apiClient)
	if state.Status == accountStateUnknown {
		return ErrPlanVerificationUnavailable
	}
	if state.Status != accountStateSignedIn || !PlanSatisfies(state.Plan, item.RequiredPlan) {
		return errUpgradeCancelled
	}
	return nil
}

func (c *launcherClient) modelRecommendationItem(ctx context.Context, model string) (ModelItem, bool) {
	for _, item := range c.recommendations(ctx) {
		if item.Name == model {
			return item, true
		}
	}
	return ModelItem{}, false
}

func (c *launcherClient) runUpgradeFlow(ctx context.Context, item ModelItem) error {
	if DefaultUpgrade != nil {
		if _, err := DefaultUpgrade(item.Name, item.RequiredPlan); err != nil {
			if errors.Is(err, ErrCancelled) {
				return errUpgradeCancelled
			}
			return err
		}
		return nil
	}

	yes, err := ConfirmPrompt(fmt.Sprintf("Upgrade to use %s?", item.Name))
	if errors.Is(err, ErrCancelled) {
		return errUpgradeCancelled
	}
	if err != nil {
		return err
	}
	if !yes {
		return errUpgradeCancelled
	}

	fmt.Fprintf(os.Stderr, "\nTo upgrade, navigate to:\n    %s\n\n", DefaultUpgradeURL)
	openNow, err := ConfirmPrompt("Open now?")
	if errors.Is(err, ErrCancelled) {
		return errUpgradeCancelled
	}
	if err != nil {
		return err
	}
	if openNow {
		OpenBrowser(DefaultUpgradeURL)
	} else {
		return errUpgradeCancelled
	}

	spinnerFrames := []string{"|", "/", "-", "\\"}
	frame := 0
	fmt.Fprintf(os.Stderr, "\033[90mwaiting for upgrade to complete... %s\033[0m", spinnerFrames[0])

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\r\033[K")
			return ctx.Err()
		case <-ticker.C:
			frame++
			fmt.Fprintf(os.Stderr, "\r\033[90mwaiting for upgrade to complete... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])
			if frame%10 != 0 {
				continue
			}
			state := launchAccountState(ctx, c.apiClient)
			if state.Status == accountStateUnknown {
				fmt.Fprintf(os.Stderr, "\r\033[K")
				return ErrPlanVerificationUnavailable
			}
			if state.Status == accountStateSignedIn && PlanSatisfies(state.Plan, item.RequiredPlan) {
				fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1mplan updated\033[0m\n")
				return nil
			}
		}
	}
}

// PlanSatisfies reports whether currentPlan can use a model that has a requiredPlan.
func PlanSatisfies(currentPlan, requiredPlan string) bool {
	required := normalizePlan(requiredPlan)
	if required == "" || required == "free" {
		return true
	}
	current := normalizePlan(currentPlan)
	return current != "" && current != "free"
}

func normalizePlan(plan string) string {
	return strings.ToLower(strings.TrimSpace(plan))
}
