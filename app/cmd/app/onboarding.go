//go:build windows || darwin

package main

import (
	"log/slog"
	"net/url"

	"github.com/ollama/ollama/app/store"
)

// showApp opens onboarding until people finish it, here or in the CLI, and
// Settings after that.
func showApp() {
	if onboardingNeeded() {
		showOnboarding()
		return
	}
	showSettings(settingsPaneDefault)
}

func onboardingNeeded() bool {
	if appStore == nil {
		return false
	}
	s, err := appStore.Settings()
	if err != nil {
		slog.Warn("failed to read onboarding state", "error", err)
		return false
	}
	return s.OnboardingVersion < store.CurrentOnboardingVersion
}

// completeOnboarding records that people finished onboarding, which the CLI
// reads too.
func completeOnboarding() error {
	s, err := appStore.Settings()
	if err != nil {
		return err
	}
	s.OnboardingVersion = store.CurrentOnboardingVersion
	return appStore.SetSettings(s)
}

// onboardingSignInURL returns the ollama.com page that connects this device
// to an account, starting with a new account when signUp is set.
func onboardingSignInURL(signUp bool) (string, error) {
	signInURL, err := settings.SignInURL()
	if err != nil {
		return "", err
	}
	u, err := url.Parse(signInURL)
	if err != nil {
		return "", err
	}
	query := u.Query()
	query.Del("launch")
	if signUp {
		query.Set("signup", "true")
	}
	u.RawQuery = query.Encode()
	return u.String(), nil
}
