//go:build darwin

package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Cocoa

#include <stdlib.h>
#include "settings_darwin.h"
*/
import "C"

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"golang.org/x/crypto/ssh"

	appauth "github.com/ollama/ollama/app/auth"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
)

// settingsStore is a reference to the app's store for settings
var settingsStore *store.Store

// SetSettingsStore sets the store reference for settings callbacks
func SetSettingsStore(s *store.Store) {
	settingsStore = s
}

//export getSettingsExpose
func getSettingsExpose() C.bool {
	if settingsStore == nil {
		return C.bool(false)
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return C.bool(false)
	}
	return C.bool(settings.Expose)
}

//export setSettingsExpose
func setSettingsExpose(expose C.bool) {
	if settingsStore == nil {
		return
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return
	}
	settings.Expose = bool(expose)
	if err := settingsStore.SetSettings(settings); err != nil {
		slog.Error("failed to save settings", "error", err)
	}
}

//export getSettingsBrowser
func getSettingsBrowser() C.bool {
	if settingsStore == nil {
		return C.bool(false)
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return C.bool(false)
	}
	return C.bool(settings.Browser)
}

//export setSettingsBrowser
func setSettingsBrowser(browser C.bool) {
	if settingsStore == nil {
		return
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return
	}
	settings.Browser = bool(browser)
	if err := settingsStore.SetSettings(settings); err != nil {
		slog.Error("failed to save settings", "error", err)
	}
}

//export getSettingsModels
func getSettingsModels() *C.char {
	if settingsStore == nil {
		return C.CString(envconfig.Models())
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return C.CString(envconfig.Models())
	}
	if settings.Models == "" {
		return C.CString(envconfig.Models())
	}
	return C.CString(settings.Models)
}

//export setSettingsModels
func setSettingsModels(path *C.char) {
	if settingsStore == nil {
		return
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return
	}
	settings.Models = C.GoString(path)
	if err := settingsStore.SetSettings(settings); err != nil {
		slog.Error("failed to save settings", "error", err)
	}
}

//export getSettingsContextLength
func getSettingsContextLength() C.int {
	if settingsStore == nil {
		return C.int(4096)
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return C.int(4096)
	}
	if settings.ContextLength <= 0 {
		return C.int(4096)
	}
	return C.int(settings.ContextLength)
}

//export setSettingsContextLength
func setSettingsContextLength(length C.int) {
	if settingsStore == nil {
		return
	}
	settings, err := settingsStore.Settings()
	if err != nil {
		slog.Error("failed to get settings", "error", err)
		return
	}
	settings.ContextLength = int(length)
	if err := settingsStore.SetSettings(settings); err != nil {
		slog.Error("failed to save settings", "error", err)
	}
}

// restartCallback is set by the app to restart the ollama server
var restartCallback func()

// SetRestartCallback sets the function to call when settings change requires a restart
func SetRestartCallback(cb func()) {
	restartCallback = cb
}

//export restartOllamaServer
func restartOllamaServer() {
	if restartCallback != nil {
		slog.Info("restarting ollama server due to settings change")
		go restartCallback()
	}
}

// hasOllamaKey checks if the user has an Ollama key file
func hasOllamaKey() bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	keyPath := filepath.Join(home, ".ollama", "id_ed25519")
	_, err = os.Stat(keyPath)
	return err == nil
}

// ensureKeypair generates a new keypair if one doesn't exist
func ensureKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")

	// Check if key already exists
	if _, err := os.Stat(privKeyPath); err == nil {
		return nil // Key exists
	}

	// Generate new keypair
	slog.Info("generating new keypair for ollama account")

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return fmt.Errorf("failed to generate key: %w", err)
	}

	// Marshal private key
	privKeyBytes, err := ssh.MarshalPrivateKey(privKey, "")
	if err != nil {
		return fmt.Errorf("failed to marshal private key: %w", err)
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(privKeyPath), 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Write private key
	if err := os.WriteFile(privKeyPath, pem.EncodeToMemory(privKeyBytes), 0o600); err != nil {
		return fmt.Errorf("failed to write private key: %w", err)
	}

	// Write public key
	sshPubKey, err := ssh.NewPublicKey(pubKey)
	if err != nil {
		return fmt.Errorf("failed to create ssh public key: %w", err)
	}
	pubKeyBytes := ssh.MarshalAuthorizedKey(sshPubKey)
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")
	if err := os.WriteFile(pubKeyPath, pubKeyBytes, 0o644); err != nil {
		return fmt.Errorf("failed to write public key: %w", err)
	}

	slog.Info("keypair generated successfully")
	return nil
}

// userResponse matches the API response from ollama.com/api/me
type userResponse struct {
	Name      string `json:"name"`
	Email     string `json:"email"`
	Plan      string `json:"plan"`
	AvatarURL string `json:"avatarurl"`
}

// fetchUserFromAPI fetches user data from ollama.com using signed request
func fetchUserFromAPI() (*userResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	signString := fmt.Sprintf("POST,/api/me?ts=%s", timestamp)
	signature, err := auth.Sign(ctx, []byte(signString))
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	endpoint := fmt.Sprintf("https://ollama.com/api/me?ts=%s", timestamp)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call ollama.com: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	var user userResponse
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Make avatar URL absolute
	if user.AvatarURL != "" && !strings.HasPrefix(user.AvatarURL, "http") {
		user.AvatarURL = "https://ollama.com/" + user.AvatarURL
	}

	// Cache the avatar URL
	cachedAvatarURL = user.AvatarURL

	// Cache the user data
	if settingsStore != nil {
		storeUser := store.User{
			Name:  user.Name,
			Email: user.Email,
			Plan:  user.Plan,
		}
		if err := settingsStore.SetUser(storeUser); err != nil {
			slog.Warn("failed to cache user", "error", err)
		}
	}

	return &user, nil
}

//export getAccountName
func getAccountName() *C.char {
	// Only return cached data - never block on network
	if settingsStore == nil {
		return C.CString("")
	}
	user, err := settingsStore.User()
	if err != nil || user == nil {
		return C.CString("")
	}
	return C.CString(user.Name)
}

// cachedAvatarURL stores the avatar URL from the last API fetch
var cachedAvatarURL string

//export getAccountAvatarURL
func getAccountAvatarURL() *C.char {
	return C.CString(cachedAvatarURL)
}

//export getAccountEmail
func getAccountEmail() *C.char {
	if settingsStore != nil {
		user, err := settingsStore.User()
		if err == nil && user != nil {
			return C.CString(user.Email)
		}
	}
	return C.CString("")
}

//export getAccountPlan
func getAccountPlan() *C.char {
	if settingsStore != nil {
		user, err := settingsStore.User()
		if err == nil && user != nil {
			return C.CString(user.Plan)
		}
	}
	return C.CString("")
}

//export signOutAccount
func signOutAccount() {
	if settingsStore != nil {
		if err := settingsStore.ClearUser(); err != nil {
			slog.Error("failed to clear user", "error", err)
		}
	}

	// Also remove the key file
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Error("failed to get home dir", "error", err)
		return
	}
	keyPath := filepath.Join(home, ".ollama", "id_ed25519")
	if err := os.Remove(keyPath); err != nil && !os.IsNotExist(err) {
		slog.Error("failed to remove key file", "error", err)
	}
}

//export openConnectUrl
func openConnectUrl() {
	// Ensure keypair exists (generate if needed)
	if err := ensureKeypair(); err != nil {
		slog.Error("failed to ensure keypair", "error", err)
		// Fallback to basic connect page
		cmd := exec.Command("open", "https://ollama.com/connect")
		cmd.Start()
		return
	}

	// Build connect URL with public key
	connectURL, err := appauth.BuildConnectURL("https://ollama.com")
	if err != nil {
		slog.Error("failed to build connect URL", "error", err)
		// Fallback to basic connect page
		connectURL = "https://ollama.com/connect"
	}

	cmd := exec.Command("open", connectURL)
	if err := cmd.Start(); err != nil {
		slog.Error("failed to open connect URL", "error", err)
	}
}

//export refreshAccountFromAPI
func refreshAccountFromAPI() {
	if !hasOllamaKey() {
		return
	}
	_, err := fetchUserFromAPI()
	if err != nil {
		slog.Debug("failed to refresh account", "error", err)
	}
}

//export prefetchAccountData
func prefetchAccountData() {
	// Run in background goroutine to not block app startup
	go func() {
		if !hasOllamaKey() {
			return
		}
		_, err := fetchUserFromAPI()
		if err != nil {
			slog.Debug("failed to prefetch account data", "error", err)
		} else {
			slog.Debug("prefetched account data successfully")
		}
	}()
}

// OpenNativeSettings opens the native settings window
func OpenNativeSettings() {
	C.openNativeSettings()
}

// Ensure the CString is freed (caller must free)
func freeCString(s *C.char) {
	C.free(unsafe.Pointer(s))
}
