//go:build integration && migration

package integration

import (
	"context"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
)

func TestCompatibilityMigrationCanceledLoad(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("requires a harness-managed server")
	}
	skipIfRemote(t)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_DEBUG", "2")
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	name := defaultTestModel("gpt-oss:20b")
	if err := PullIfMissing(ctx, client, name); err != nil {
		t.Fatal(err)
	}
	before := showOrFatal(ctx, t, client, name)
	const retained = "phase3-test/retained:latest"
	if _, err := client.Copy(ctx, &api.CopyRequest{Source: name, Destination: retained}); err != nil {
		t.Fatal(err)
	}
	retainedDigests := compatibilityModelDigests(ctx, t, client, retained)

	start := len(serverLog.String())
	loadCtx, cancelLoad := context.WithCancel(ctx)
	defer cancelLoad()
	done := make(chan error, 1)
	go func() {
		_, err := client.Embed(loadCtx, &api.EmbedRequest{Model: name, Input: ""})
		done <- err
	}()
	waitForMigrationLog(ctx, t, start, "starting local compat GGUF migration")
	if strings.Contains(serverLog.String()[start:], "completed local compat GGUF migration") {
		t.Fatal("conversion finished before cancellation; use a larger legacy model")
	}
	cancelLoad()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled request returned %v", err)
	}
	t.Log("initiating request canceled; a second request must wait for the same conversion")
	validateCompatibilityPrimaryCapability(ctx, t, client, name, before, compatKeepAlive)
	logs := serverLog.String()[start:]
	if strings.Count(logs, "starting local compat GGUF migration") != 1 || strings.Count(logs, "completed local compat GGUF migration") != 1 {
		t.Fatal("canceled and subsequent loads did not share exactly one successful conversion")
	}
	converted := ggufDigestsFromShow(t, showOrFatal(ctx, t, client, name))
	for _, digest := range ggufDigestsFromShow(t, before) {
		if exists, err := client.HeadBlob(ctx, digest); err != nil || !exists {
			t.Fatalf("shared source blob removed: %s exists=%v err=%v", digest, exists, err)
		}
	}
	if got := compatibilityModelDigests(ctx, t, client, retained); !slices.Equal(got, retainedDigests) {
		t.Fatal("unloaded legacy alias was changed by another tag's conversion")
	}
	for _, operation := range []string{"remove", "replace"} {
		if !t.Run(operation+"_during_conversion", func(t *testing.T) {
			const changing = "phase3-test/changing:latest"
			if _, err := client.Copy(ctx, &api.CopyRequest{Source: retained, Destination: changing}); err != nil {
				t.Fatal(err)
			}
			start := len(serverLog.String())
			loadCtx, cancelLoad := context.WithCancel(ctx)
			defer cancelLoad()
			done := make(chan error, 1)
			go func() {
				_, err := client.Embed(loadCtx, &api.EmbedRequest{Model: changing, Input: ""})
				done <- err
			}()
			waitForMigrationLog(ctx, t, start, "starting local compat GGUF migration")
			if strings.Contains(serverLog.String()[start:], "completed local compat GGUF migration") {
				t.Fatal("conversion finished before tag mutation; use a larger legacy model")
			}
			if operation == "remove" {
				if err := client.Delete(ctx, &api.DeleteRequest{Model: changing}); err != nil {
					t.Fatal(err)
				}
			} else if _, err := client.Copy(ctx, &api.CopyRequest{Source: name, Destination: changing}); err != nil {
				t.Fatal(err)
			}
			if err := <-done; err == nil {
				t.Fatal("load succeeded after its source tag changed during conversion")
			}
			waitForMigrationLog(ctx, t, start, "local compatibility migration failed")
			got, err := client.Show(ctx, &api.ShowRequest{Model: changing})
			if operation == "remove" {
				var status api.StatusError
				if !errors.As(err, &status) || status.StatusCode != 404 {
					t.Fatalf("removed tag resurrected: show=%+v err=%v", got, err)
				}
			} else if err != nil || !slices.Equal(ggufDigestsFromShow(t, got), converted) {
				t.Fatalf("replacement was overwritten: show=%+v err=%v", got, err)
			}
			for _, digest := range converted {
				if exists, err := client.HeadBlob(ctx, digest); err != nil || !exists {
					t.Fatalf("aborted conversion removed a referenced output: %s exists=%v err=%v", digest, exists, err)
				}
			}
		}) {
			return
		}
	}
	if err := client.Delete(ctx, &api.DeleteRequest{Model: retained}); err != nil {
		t.Fatal(err)
	}
	for _, digest := range ggufDigestsFromShow(t, before) {
		if slices.Contains(converted, digest) {
			continue
		}
		if exists, err := client.HeadBlob(ctx, digest); err != nil || exists {
			t.Fatalf("source blob retained after final reference was removed: %s exists=%v err=%v", digest, exists, err)
		}
	}
	t.Log("conversion survived cancellation; shared blobs survived until their last reference was removed")
}

func TestCompatibilityStartupRetirement(t *testing.T) {
	phase2 := os.Getenv("OLLAMA_TEST_PHASE2_BIN")
	if phase2 == "" {
		t.Skip("set OLLAMA_TEST_PHASE2_BIN to a Phase 2 release binary")
	}
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("requires harness-managed servers")
	}
	skipIfRemote(t)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_DEBUG", "2")
	current := ollamaBin()
	t.Setenv("OLLAMA_BIN", phase2)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer func() {
		if cleanup != nil {
			cleanup()
		}
	}()
	name := defaultTestModel("gpt-oss:20b")
	if err := PullIfMissing(ctx, client, name); err != nil {
		t.Fatal(err)
	}
	before := showOrFatal(ctx, t, client, name)
	const untouched = "gemma3:1b"
	if err := PullIfMissing(ctx, client, untouched); err != nil {
		t.Fatal(err)
	}
	untouchedBefore := compatibilityModelDigests(ctx, t, client, untouched)
	if _, err := client.Embed(ctx, &api.EmbedRequest{Model: name, Input: "", KeepAlive: &api.Duration{Duration: 0}}); err != nil {
		t.Fatal(err)
	}
	waitForMigrationLog(ctx, t, 0, "completed local compat GGUF migration")
	children, err := client.ShowManifests(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		t.Fatal(err)
	}
	if len(children.Manifests) != 2 {
		t.Fatalf("Phase 2 fixture must have both children, got %d", len(children.Manifests))
	}
	var converted []string
	for _, child := range children.Manifests {
		if child.Runner == manifest.RunnerLlamaCPP {
			converted = ggufDigestsFromShow(t, &child.ShowResponse)
		}
	}
	if len(converted) == 0 {
		t.Fatal("Phase 2 did not install a converted child")
	}
	cleanup()
	cleanup = nil
	t.Setenv("OLLAMA_BIN", current)
	client, _, cleanup = InitServerConnection(ctx, t)
	children, err = client.ShowManifests(ctx, &api.ShowRequest{Model: name})
	if err != nil || len(children.Manifests) != 1 || children.Manifests[0].Runner != manifest.RunnerLlamaCPP {
		t.Fatalf("startup did not retire legacy child: response=%+v err=%v", children, err)
	}
	if got := ggufDigestsFromShow(t, &children.Manifests[0].ShowResponse); !slices.Equal(got, converted) {
		t.Fatal("startup changed the existing converted GGUF")
	}
	for _, digest := range ggufDigestsFromShow(t, before) {
		if slices.Contains(converted, digest) {
			continue
		}
		if exists, err := client.HeadBlob(ctx, digest); err != nil || exists {
			t.Fatalf("startup retained orphaned legacy GGUF %s: exists=%v err=%v", digest, exists, err)
		}
	}
	if got := compatibilityModelDigests(ctx, t, client, untouched); !slices.Equal(got, untouchedBefore) {
		t.Fatal("startup modified a legacy model that has never been loaded")
	}
	if strings.Contains(serverLog.String(), "starting local compat GGUF migration") {
		t.Fatal("startup performed a conversion")
	}
	validateCompatibilityPrimaryCapability(ctx, t, client, name, before, compatKeepAlive)
	if strings.Contains(serverLog.String(), "starting local compat GGUF migration") {
		t.Fatal("loading the retired store converted the model again")
	}
	t.Log("startup reclaimed legacy GGUFs, retained the converted child, and left unloaded legacy models unchanged")
}

func waitForMigrationLog(ctx context.Context, t *testing.T, start int, marker string) {
	t.Helper()
	deadline := time.NewTimer(migrationConversionMaxTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	for {
		if strings.Contains(serverLog.String()[start:], marker) {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("waiting for %q: %v", marker, ctx.Err())
		case <-deadline.C:
			t.Fatalf("timed out waiting for %q", marker)
		case <-tick.C:
		}
	}
}
