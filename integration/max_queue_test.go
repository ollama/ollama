//go:build integration

package integration

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

func TestMaxQueue(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("Max Queue test requires spawning a local server so we can adjust the queue size")
		return
	}

	// Note: This test can be quite slow when running in CPU mode, so keep the threadCount low unless your on GPU
	// Also note that by default Darwin can't sustain > ~128 connections without adjusting limits
	threadCount := 16
	t.Setenv("OLLAMA_MAX_QUEUE", strconv.Itoa(threadCount))

	req := api.GenerateRequest{
		Model:  smol,
		Prompt: "write a long historical fiction story about christopher columbus.  use at least 10 facts from his actual journey",
		Options: map[string]any{
			"seed":        42,
			"temperature": 0.0,
		},
	}
	resp := []string{"explore", "discover", "ocean"}

	// CPU mode takes much longer at the limit with a large queue setting
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	require.NoError(t, PullIfMissing(ctx, client, req.Model))

	// Context for the worker threads so we can shut them down
	// embedCtx, embedCancel := context.WithCancel(ctx)
	embedCtx := ctx

	var genwg sync.WaitGroup
	genwg.Add(1)
	go func() {
		defer genwg.Done()
		slog.Info("Starting generate request")
		DoGenerate(ctx, t, client, req, resp, 45*time.Second, 5*time.Second)
		slog.Info("generate completed")
	}()

	// Give the generate a chance to get started before we start hammering on embed requests
	time.Sleep(10 * time.Millisecond)

	threadCount += 10 // Add a few extra to ensure we push the queue past its limit
	busyCount := 0
	resetByPeerCount := 0
	canceledCount := 0
	successCount := 0
	counterMu := sync.Mutex{}
	var embedwg sync.WaitGroup
	for i := 0; i < threadCount; i++ {
		embedwg.Add(1)
		go func(i int) {
			defer embedwg.Done()
			slog.Info("embed started", "id", i)
			embedReq := api.EmbeddingRequest{
				Model:   req.Model,
				Prompt:  req.Prompt,
				Options: req.Options,
			}
			// Fresh client for every request
			client, _ = GetTestEndpoint()

			resp, genErr := client.Embeddings(embedCtx, &embedReq)
			counterMu.Lock()
			defer counterMu.Unlock()
			switch {
			case genErr == nil:
				successCount++
				require.Greater(t, len(resp.Embedding), 5) // somewhat arbitrary, but sufficient to be reasonable
			case errors.Is(genErr, context.Canceled):
				canceledCount++
			case strings.Contains(genErr.Error(), "busy"):
				busyCount++
			case strings.Contains(genErr.Error(), "connection reset by peer"):
				resetByPeerCount++
			default:
				require.NoError(t, genErr, "%d request failed", i)
			}

			slog.Info("embed finished", "id", i)
		}(i)
	}
	genwg.Wait()
	slog.Info("generate done, waiting for embeds")
	embedwg.Wait()

	slog.Info("embeds completed", "success", successCount, "busy", busyCount, "reset", resetByPeerCount, "canceled", canceledCount)
	require.Equal(t, resetByPeerCount, 0, "Connections reset by peer, have you updated your fd and socket limits?")
	require.True(t, busyCount > 0, "no requests hit busy error but some should have")
	require.True(t, canceledCount == 0, "no requests should have been canceled due to timeout")

}
