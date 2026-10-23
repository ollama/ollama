package updatetrust

// PublicKeyPEM is the SPKI ed25519 public key the app trusts for update
// artifacts. scripts/generate-update-key.sh rewrites this file on the release
// Mac. The matching private key stays on that Mac and is never committed.
// An empty value fails closed: the updater will not install an update.
const PublicKeyPEM = ""
