package signature

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"

	"golang.org/x/crypto/ssh"
)

type SignatureData struct {
	Method string
	Path   string
	Data   []byte
}

func GetBytesToSign(s SignatureData) []byte {
	// contentHash = base64(hex(sha256(s.Data)))
	hash := sha256.Sum256(s.Data)
	hashHex := make([]byte, hex.EncodedLen(len(hash)))
	hex.Encode(hashHex, hash[:])
	contentHash := base64.StdEncoding.EncodeToString(hashHex)

	// bytesToSign e.g.: "GET,http://localhost,OTdkZjM1O...
	bytesToSign := []byte(strings.Join([]string{s.Method, s.Path, contentHash}, ","))

	return bytesToSign
}

// SignData takes a SignatureData object and signs it with a raw private key
func SignAuthData(s SignatureData, rawKey []byte) (string, error) {
	bytesToSign := GetBytesToSign(s)

	// TODO replace this w/ a non-SSH based private key
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

	signedData, err := signer.Sign(nil, bytesToSign)
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	sig := fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob))
	return sig, nil
}
