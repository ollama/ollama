package main

import (
	"context"
	"fmt"
	"log"

	// Replace this with the correct import path to your local auth package
	"your/module/path/auth"
)

func main() {
	ctx := context.Background()

	// Deploy script as a byte slice
	deployScript := []byte(`#!/bin/bash
echo "Starting deployment..."
docker pull myimage:latest
docker-compose up -d
echo "Deployment finished."
`)

	// Sign the content of the script
	signature, err := auth.Sign(ctx, deployScript)
	if err != nil {
		log.Fatalf("Failed to sign deploy script: %v", err)
	}

	fmt.Println("Generated signature:")
	fmt.Println(signature)
}
