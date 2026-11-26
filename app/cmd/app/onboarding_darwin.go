//go:build darwin

package main

// #include <stdbool.h>
// #include "app_darwin.h"
import "C"

// The onboarding window calls these exports from background dispatch queues,
// like the settings window does.

type onboardingResult struct {
	URL   string `json:"url,omitempty"`
	Error string `json:"error,omitempty"`
}

func showOnboarding() {
	C.showOnboarding()
}

// ShowApp opens onboarding or Settings, as opening Ollama does.
//
//export ShowApp
func ShowApp() {
	showApp()
}

//export OnboardingSignInURL
func OnboardingSignInURL(signUp C.bool) *C.char {
	u, err := onboardingSignInURL(bool(signUp))
	if err != nil {
		return jsonCString(onboardingResult{Error: err.Error()})
	}
	return jsonCString(onboardingResult{URL: u})
}

//export OnboardingComplete
func OnboardingComplete() *C.char {
	if err := completeOnboarding(); err != nil {
		return jsonCString(onboardingResult{Error: err.Error()})
	}
	return jsonCString(onboardingResult{})
}
