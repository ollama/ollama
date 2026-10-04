//go:build windows || darwin

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	ollamaAuth "github.com/ollama/ollama/auth"
)

// ollamaDotCom is the ollama.com base URL. OLLAMA_DOT_COM_URL overrides it
// for testing against other environments.
var ollamaDotCom = func() string {
	if u := os.Getenv("OLLAMA_DOT_COM_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	return "https://ollama.com"
}()

var (
	accountClient   = &http.Client{Timeout: 10 * time.Second}
	signOllamaData  = ollamaAuth.Sign
	ollamaPublicKey = ollamaAuth.GetPublicKey
	errSignedOut    = errors.New("not signed in to ollama.com")
)

// accountState is the ollama.com account shown in Settings.
type accountState struct {
	SignedIn  bool   `json:"signedIn"`
	Name      string `json:"name,omitempty"`
	Email     string `json:"email,omitempty"`
	Plan      string `json:"plan,omitempty"`
	AvatarURL string `json:"avatarURL,omitempty"`
	// Cached is set when ollama.com could not be reached and the state comes
	// from the last successful check.
	Cached bool `json:"cached,omitempty"`
}

// CachedAccount returns the account from the last successful check, without
// contacting ollama.com.
func (c *settingsController) CachedAccount() accountState {
	user, err := c.store.User()
	if err != nil || user == nil || user.Name == "" {
		return accountState{}
	}
	return accountState{SignedIn: true, Name: user.Name, Email: user.Email, Plan: user.Plan, Cached: true}
}

// Account checks this device's ollama.com account. When ollama.com cannot be
// reached it returns the cached account with the error.
func (c *settingsController) Account(ctx context.Context) (accountState, error) {
	user, err := fetchAccount(ctx)
	if errors.Is(err, errSignedOut) {
		if err := c.store.ClearUser(); err != nil {
			return accountState{}, err
		}
		return accountState{}, nil
	}
	if err != nil {
		return c.CachedAccount(), err
	}

	if err := c.store.SetUser(store.User{Name: user.Name, Email: user.Email, Plan: user.Plan}); err != nil {
		return accountState{}, err
	}
	return accountState{
		SignedIn:  true,
		Name:      user.Name,
		Email:     user.Email,
		Plan:      user.Plan,
		AvatarURL: absoluteOllamaURL(user.AvatarURL),
	}, nil
}

// SignInURL returns the ollama.com page that connects this device's key to
// an account.
func (c *settingsController) SignInURL() (string, error) {
	publicKey, err := ollamaPublicKey()
	if err != nil {
		return "", fmt.Errorf("read device key: %w", err)
	}
	hostname, _ := os.Hostname()
	query := url.Values{
		"name":   {hostname},
		"key":    {base64.RawURLEncoding.EncodeToString([]byte(publicKey))},
		"launch": {"true"},
	}
	return ollamaDotCom + "/connect?" + query.Encode(), nil
}

// SignOut disconnects this device's key from its ollama.com account.
func (c *settingsController) SignOut(ctx context.Context) error {
	publicKey, err := ollamaPublicKey()
	if err != nil {
		return fmt.Errorf("read device key: %w", err)
	}
	encodedKey := base64.RawURLEncoding.EncodeToString([]byte(publicKey))
	req, err := newSignedOllamaRequest(ctx, http.MethodDelete, ollamaDotCom+"/api/user/keys/"+url.PathEscape(encodedKey))
	if err != nil {
		return err
	}
	resp, err := accountClient.Do(req)
	if err != nil {
		return fmt.Errorf("sign out: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	// An unauthorized key is already signed out.
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusUnauthorized {
		return fmt.Errorf("sign out: ollama.com returned %s", resp.Status)
	}
	return c.store.ClearUser()
}

func fetchAccount(ctx context.Context) (*api.UserResponse, error) {
	req, err := newSignedOllamaRequest(ctx, http.MethodPost, ollamaDotCom+"/api/me")
	if err != nil {
		return nil, err
	}
	resp, err := accountClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("check account: %w", err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, errSignedOut
	default:
		return nil, fmt.Errorf("check account: ollama.com returned %s", resp.Status)
	}

	var user api.UserResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&user); err != nil {
		return nil, fmt.Errorf("check account: %w", err)
	}
	if strings.TrimSpace(user.Name) == "" {
		return nil, errSignedOut
	}
	return &user, nil
}

// newSignedOllamaRequest signs a request with this device's key so
// ollama.com can identify its account.
func newSignedOllamaRequest(ctx context.Context, method, endpoint string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
	if err != nil {
		return nil, err
	}
	query := req.URL.Query()
	query.Set("ts", strconv.FormatInt(time.Now().Unix(), 10))
	req.URL.RawQuery = query.Encode()
	signature, err := signOllamaData(ctx, []byte(fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())))
	if err != nil {
		return nil, fmt.Errorf("sign request: %w", err)
	}
	req.Header.Set("Authorization", signature)
	return req, nil
}

func absoluteOllamaURL(path string) string {
	if path == "" || strings.HasPrefix(path, "https://") || strings.HasPrefix(path, "http://") {
		return path
	}
	return ollamaDotCom + "/" + strings.TrimPrefix(path, "/")
}
