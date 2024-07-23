package server

/**
 * These tests validate the client/server interactions with various combinations
 * TLS enabled. The tests validate both the client and server, but live here in
 * the server package to avoid cyclic dependencies.
 */

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig/configtest"
	"github.com/stretchr/testify/assert"
)

/////////////
// Helpers //
/////////////

func setupListener(t *testing.T) (*net.Listener, int) {
	// Create a listener on a random port
	ln, _ := net.Listen("tcp", "localhost:0")
	port := ln.Addr().(*net.TCPAddr).Port
	t.Cleanup(func() { ln.Close() })
	return &ln, port
}

func startServer(t *testing.T, ln *net.Listener) {
	server, serverErr := ServeNonBlocking(*ln)
	t.Cleanup(func() {
		http.DefaultServeMux = new(http.ServeMux)
	})
	assert.Nil(t, serverErr)
	t.Cleanup(server.Terminate)
}

func setupClient(t *testing.T) *api.Client {
	client, clientErr := api.ClientFromEnvironment()
	assert.Nil(t, clientErr)
	assert.NotNil(t, client)
	return client
}

func runEndToEndTest(t *testing.T, ln *net.Listener) error {
	startServer(t, ln)
	client := setupClient(t)
	return client.Heartbeat(context.Background())
}

// Test that the client and server can communicate with no TLS enabled
func TestClientServerNoTLS(t *testing.T) {
	// Create a listener on a random port
	ln, port := setupListener(t)

	// Set the testing env and re-parse config
	t.Setenv("OLLAMA_HOST", fmt.Sprintf("http://localhost:%d", port))

	// Run the end to end test
	assert.Nil(t, runEndToEndTest(t, ln))
}

// Test that the client and server can communicate with non-mutual TLS enabled
func TestClientServerTLS(t *testing.T) {
	// Generate the TLS data
	tlsTestData := configtest.NewTLSTestData(t.TempDir())

	// Create a listener on a random port
	ln, port := setupListener(t)

	// Set the testing env and re-parse config
	t.Setenv("OLLAMA_HOST", fmt.Sprintf("https://localhost:%d", port))
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)

	// Run the end to end test
	assert.Nil(t, runEndToEndTest(t, ln))
}

// Test that the client and server can communicate with mutual TLS enabled where
// the client and server share a CA
func TestClientServerMTLSSharedCA(t *testing.T) {
	// Generate the TLS data
	tlsTestData := configtest.NewTLSTestData(t.TempDir())

	// Create a listener on a random port
	ln, port := setupListener(t)

	// Set the testing env and re-parse config
	// NOTE: Reusing server TLS data for client to share CA
	t.Setenv("OLLAMA_HOST", fmt.Sprintf("https://localhost:%d", port))
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	t.Setenv("OLLAMA_TLS_CLIENT_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_CLIENT_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ServerCA)

	// Run the end to end test
	assert.Nil(t, runEndToEndTest(t, ln))
}

// Test that the client and server can communicate with mutual TLS enabled where
// the client and server use separate CAs
func TestClientServerMTLSSeparateCA(t *testing.T) {
	// Generate the TLS data
	tlsTestData := configtest.NewTLSTestData(t.TempDir())

	// Create a listener on a random port
	ln, port := setupListener(t)

	// Set the testing env and re-parse config
	t.Setenv("OLLAMA_HOST", fmt.Sprintf("https://localhost:%d", port))
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	t.Setenv("OLLAMA_TLS_CLIENT_KEY", tlsTestData.ClientKey)
	t.Setenv("OLLAMA_TLS_CLIENT_CERT", tlsTestData.ClientCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ClientCA)

	// Run the end to end test
	assert.Nil(t, runEndToEndTest(t, ln))
}

// Test that a client without a key/cert is rejected by a server with mTLS
func TestClientServerMTLSServerOnly(t *testing.T) {
	// Generate the TLS data
	tlsTestData := configtest.NewTLSTestData(t.TempDir())
	assert.NotNil(t, tlsTestData)

	// Create a listener on a random port
	ln, port := setupListener(t)

	// Set the testing env for the server to require mTLS and re-parse config
	t.Setenv("OLLAMA_HOST", fmt.Sprintf("https://localhost:%d", port))
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ClientCA)
	startServer(t, ln)

	// Reset config for client to only require TLS
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	client := setupClient(t)

	// Run the end to end test and make sure an error is returned
	assert.NotNil(t, client.Heartbeat(context.Background()))
}
