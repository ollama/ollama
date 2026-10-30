package launch

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	modelUpgradeURL = "https://ollama.com/upgrade"
	modelCreditsURL = "https://ollama.com/settings"

	accountCheckTimeout = 3 * time.Second
)

var errModelPaymentDeclined = errors.New("model payment declined")

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
			return "Use usage credits or upgrade your plan"
		}
	}
	return ""
}

func itemHasRecommendationMetadata(item ModelItem) bool {
	return item.Recommended || strings.TrimSpace(item.RequiredPlan) != ""
}

func (c *launcherClient) confirmCloudModelPayment(ctx context.Context, model string) error {
	if !isCloudModelName(model) {
		return nil
	}
	item, ok := c.modelRecommendationItem(ctx, model)
	if !ok || strings.TrimSpace(item.RequiredPlan) == "" {
		return nil
	}

	state := launchAccountState(ctx, c.apiClient)
	if state.Status != accountStateUnknown {
		c.accountState = &state
	}
	if state.Status == accountStateUnknown {
		return nil
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
			return nil
		}
	}

	if PlanSatisfies(state.Plan, item.RequiredPlan) {
		return nil
	}

	restore := withLaunchConfirmPolicy(c.policy.confirmPolicy())
	defer restore()
	proceed, err := confirmModelPayment(model)
	if errors.Is(err, ErrCancelled) || (err == nil && !proceed) {
		return errModelPaymentDeclined
	}
	return err
}

func (c *launcherClient) modelRecommendationItem(ctx context.Context, model string) (ModelItem, bool) {
	for _, item := range c.recommendations(ctx) {
		if item.Name == model {
			return item, true
		}
	}
	return ModelItem{}, false
}

// confirmModelPayment shows advisory payment guidance without checking credit availability.
func confirmModelPayment(modelName string) (bool, error) {
	prompt := fmt.Sprintf("%s\n\nThis model requires usage credits or a paid plan.\n\nAdd credits: %s\nUpgrade: %s\n\nContinue with %s?", modelName, modelCreditsURL, modelUpgradeURL, modelName)
	return ConfirmPromptWithOptions(prompt, ConfirmOptions{
		YesLabel: "Continue",
		NoLabel:  "Choose another model",
		Default:  ConfirmDefaultYes,
	})
}

// PlanSatisfies compares the subscription hints in recommendation metadata.
// Inference authorization also accounts for usage credits and other server policy.
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
