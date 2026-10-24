package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
)

type registryChallenge struct {
	Realm   string
	Service string
	Scope   string
}

// trustedRealmHosts maps a registry host to the additional hosts it is allowed
// to delegate token issuance to. Some registries serve their API and their
// authentication endpoint from different hosts: the default Ollama registry
// serves its API from registry.ollama.ai but issues tokens from ollama.com.
var trustedRealmHosts = map[string][]string{
	"registry.ollama.ai": {"ollama.com"},
}

// realmHostAllowed reports whether a token realm served from realmHost is
// trusted for a registry that was contacted at originalHost. It is used to
// permit known registry/auth host pairs while still rejecting arbitrary
// cross-origin realms.
func realmHostAllowed(originalHost, realmHost string) bool {
	for _, host := range trustedRealmHosts[originalHost] {
		if host == realmHost {
			return true
		}
	}
	return false
}

func (r registryChallenge) URL() (*url.URL, error) {
	redirectURL, err := url.Parse(r.Realm)
	if err != nil {
		return nil, err
	}

	values := redirectURL.Query()
	values.Add("service", r.Service)
	for _, s := range strings.Split(r.Scope, " ") {
		values.Add("scope", s)
	}

	values.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		return nil, err
	}

	values.Add("nonce", nonce)

	redirectURL.RawQuery = values.Encode()
	return redirectURL, nil
}

func getAuthorizationToken(ctx context.Context, challenge registryChallenge, originalHost string) (string, error) {
	redirectURL, err := challenge.URL()
	if err != nil {
		return "", err
	}

	// Validate that the realm host is allowed for the original request host to
	// prevent sending tokens cross-origin. The realm host is accepted when it
	// matches the original host or is a known host of the same registry.
	if redirectURL.Host != originalHost && !realmHostAllowed(originalHost, redirectURL.Host) {
		return "", fmt.Errorf("realm host %q does not match original host %q", redirectURL.Host, originalHost)
	}

	sha256sum := sha256.Sum256(nil)
	data := []byte(fmt.Sprintf("%s,%s,%s", http.MethodGet, redirectURL.String(), base64.StdEncoding.EncodeToString([]byte(hex.EncodeToString(sha256sum[:])))))

	headers := make(http.Header)
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return "", err
	}

	headers.Add("Authorization", signature)

	response, err := makeRequest(ctx, http.MethodGet, redirectURL, headers, nil, &registryOptions{})
	if err != nil {
		return "", err
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return "", fmt.Errorf("%d: %v", response.StatusCode, err)
	}

	if response.StatusCode >= http.StatusBadRequest {
		if len(body) > 0 {
			return "", fmt.Errorf("%d: %s", response.StatusCode, body)
		} else {
			return "", fmt.Errorf("%d", response.StatusCode)
		}
	}

	var token api.TokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		return "", err
	}

	return token.Token, nil
}
