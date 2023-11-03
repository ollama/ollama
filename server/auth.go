package server

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/jmorganca/ollama/api"
)

type AuthRedirect struct {
	Realm   string
	Service string
	Scope   string
}

type SignatureData struct {
	Method string
	Path   string
	Data   []byte
}

func generateNonce(length int) (string, error) {
	nonce := make([]byte, length)
	_, err := rand.Read(nonce)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func (r AuthRedirect) URL() (*url.URL, error) {
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

	nonce, err := generateNonce(16)
	if err != nil {
		return nil, err
	}
	values.Add("nonce", nonce)

	redirectURL.RawQuery = values.Encode()
	return redirectURL, nil
}

func getAuthToken(ctx context.Context, redirData AuthRedirect) (string, error) {
	redirectURL, err := redirData.URL()
	if err != nil {
		return "", err
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	keyPath := filepath.Join(home, ".ollama", "id_ed25519")

	rawKey, err := os.ReadFile(keyPath)
	if err != nil {
		log.Printf("Failed to load private key: %v", err)
		return "", err
	}

	s := SignatureData{
		Method: http.MethodGet,
		Path:   redirectURL.String(),
		Data:   nil,
	}

	sig, err := s.Sign(rawKey)
	if err != nil {
		return "", err
	}

	headers := make(http.Header)
	headers.Set("Authorization", sig)
	resp, err := makeRequest(ctx, http.MethodGet, redirectURL, headers, nil, nil)
	if err != nil {
		log.Printf("couldn't get token: %q", err)
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("on pull registry responded with code %d: %s", resp.StatusCode, body)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var tok api.TokenResponse
	if err := json.Unmarshal(respBody, &tok); err != nil {
		return "", err
	}

	return tok.Token, nil
}

// Bytes returns a byte slice of the data to sign for the request
func (s SignatureData) Bytes() []byte {
	// We first derive the content hash of the request body using:
	//     base64(hex(sha256(request body)))

	hash := sha256.Sum256(s.Data)
	hashHex := make([]byte, hex.EncodedLen(len(hash)))
	hex.Encode(hashHex, hash[:])
	contentHash := base64.StdEncoding.EncodeToString(hashHex)

	// We then put the entire request together in a serialize string using:
	//       "<method>,<uri>,<content hash>"
	// e.g.  "GET,http://localhost,OTdkZjM1O..."

	return []byte(strings.Join([]string{s.Method, s.Path, contentHash}, ","))
}

// SignData takes a SignatureData object and signs it with a raw private key
func (s SignatureData) Sign(rawKey []byte) (string, error) {
	privateKey, err := ssh.ParseRawPrivateKey(rawKey)
	if err != nil {
		return "", err
	}

	signer, err := ssh.NewSignerFromKey(privateKey)
	if err != nil {
		return "", err
	}

	// get the pubkey, but remove the type
	pubKey := ssh.MarshalAuthorizedKey(signer.PublicKey())
	parts := bytes.Split(pubKey, []byte(" "))
	if len(parts) < 2 {
		return "", fmt.Errorf("malformed public key")
	}

	signedData, err := signer.Sign(nil, s.Bytes())
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	sig := fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob))
	return sig, nil
}
