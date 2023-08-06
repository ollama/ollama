package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"strings"

	"golang.org/x/crypto/ssh"

	"github.com/jmorganca/ollama/api"
)

type SignatureData struct {
	Method string
	Path   string
	Data   []byte
}

func getAuthToken(mp ModelPath, regOpts *RegistryOptions) (string, error) {
	//url := fmt.Sprintf("%s/token", mp.Registry)
	url := fmt.Sprintf("localhost/token?service=%s&scope=repository:%s:pull,push", mp.Registry, mp.GetNamespaceRepository())
	//url := "localhost/token"
	fmt.Printf("url = '%s'", url)

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	keyPath := path.Join(home, ".ollama/id_ed25519")

	rawKey, err := ioutil.ReadFile(keyPath)
	if err != nil {
		log.Printf("Failed to load private key: %v", err)
		return "", err
	}

	s := SignatureData{
		Method: "GET",
		Path:   url,
		Data:   nil,
	}

	if !strings.HasPrefix(s.Path, "http") {
		if regOpts.Insecure {
			s.Path = "http://" + url
		} else {
			s.Path = "https://" + url
		}
	}

	sig, err := s.Sign(rawKey)
	if err != nil {
		return "", err
	}

	log.Printf("sig = %s", sig)

	headers := map[string]string{
		"Authorization": sig,
	}

	resp, err := makeRequest("GET", url, headers, nil, regOpts)
	if err != nil {
		log.Printf("couldn't get token: %q", err)
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusOK {
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

func (s SignatureData) Bytes() []byte {
	// contentHash = base64(hex(sha256(s.Data)))
	hash := sha256.Sum256(s.Data)
	hashHex := make([]byte, hex.EncodedLen(len(hash)))
	hex.Encode(hashHex, hash[:])
	contentHash := base64.StdEncoding.EncodeToString(hashHex)
	log.Printf("data = %v hash = %v hashHex = %v contentHash = %v", s.Data, hash, string(hashHex), contentHash)

	log.Printf("string = '%s'", strings.Join([]string{s.Method, s.Path, contentHash}, ","))
	// bytesToSign e.g.: "GET,http://localhost,OTdkZjM1O...
	bytesToSign := []byte(strings.Join([]string{s.Method, s.Path, contentHash}, ","))

	return bytesToSign
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
		return "", fmt.Errorf("malformed private key")
	}

	signedData, err := signer.Sign(nil, s.Bytes())
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	sig := fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob))
	return sig, nil
}
