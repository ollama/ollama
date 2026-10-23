// Command updatesign generates, checks, and applies the fork's ed25519 update key.
//
//	go run ./scripts/updatesign generate <private.pem>
//	go run ./scripts/updatesign write-public <private.pem>
//	go run ./scripts/updatesign check-embedded <private.pem>
//	go run ./scripts/updatesign sign <private.pem> <file>...
//	go run ./scripts/updatesign verify <file>...
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/app/updatetrust"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "generate":
		if len(os.Args) != 3 {
			usage()
		}
		publicPEM, privatePEM, err := updatetrust.GenerateKeyPair()
		if err != nil {
			die(err)
		}
		if err := os.MkdirAll(filepath.Dir(os.Args[2]), 0o755); err != nil {
			die(err)
		}
		if err := os.WriteFile(os.Args[2], []byte(privatePEM), 0o600); err != nil {
			die(err)
		}
		fmt.Printf("Wrote %s\n%s", os.Args[2], publicPEM)
	case "write-public":
		if len(os.Args) != 3 {
			usage()
		}
		privatePEM, err := os.ReadFile(os.Args[2])
		if err != nil {
			die(err)
		}
		publicPEM, err := updatetrust.PublicFromPrivate(string(privatePEM))
		if err != nil {
			die(err)
		}
		const dest = "app/updatetrust/public_key.go"
		if err := os.WriteFile(dest, []byte(updatetrust.RenderPublicKeyFile(publicPEM)), 0o644); err != nil {
			die(err)
		}
		fmt.Printf("Wrote public key into %s\n", dest)
	case "check-embedded":
		if len(os.Args) != 3 {
			usage()
		}
		privatePEM, err := os.ReadFile(os.Args[2])
		if err != nil {
			die(err)
		}
		if err := updatetrust.CheckEmbedded(string(privatePEM)); err != nil {
			die(err)
		}
		fmt.Println("Update signing key matches the embedded public key.")
	case "sign":
		if len(os.Args) < 4 {
			usage()
		}
		privatePEM, err := os.ReadFile(os.Args[2])
		if err != nil {
			die(err)
		}
		for _, path := range os.Args[3:] {
			if err := updatetrust.WriteSignature(string(privatePEM), path); err != nil {
				die(err)
			}
			fmt.Printf("Signed %s\n", path)
		}
	case "verify":
		if len(os.Args) < 3 {
			usage()
		}
		for _, path := range os.Args[2:] {
			if err := updatetrust.VerifyFileWithSidecar(updatetrust.PublicKeyPEM, path); err != nil {
				die(fmt.Errorf("%s: %w", path, err))
			}
			fmt.Printf("Verified %s\n", path)
		}
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run ./scripts/updatesign generate|write-public|check-embedded|sign|verify ...\n")
	os.Exit(2)
}

func die(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
