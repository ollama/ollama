package updater

import "github.com/ollama/ollama/app/updatetrust"

// updatePublicKey is the ed25519 public key trusted for update artifacts.
// Tests and the macOS rehearsal replace it. Production uses the key embedded
// by scripts/generate-update-key.sh. Empty fails closed.
var updatePublicKey = updatetrust.PublicKeyPEM

// verifyStagedArtifact checks the ed25519 sidecar written next to a staged
// zip or exe. Apple codesign and Windows Authenticode run after this.
func verifyStagedArtifact(path string) error {
	return updatetrust.VerifyFileWithSidecar(updatePublicKey, path)
}
