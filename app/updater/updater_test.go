//go:build windows || darwin

package updater

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestIsNewReleaseAvailable(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	slog.Debug("server", "url", server.URL)

	updater := &Updater{Store: &store.Store{}}
	defer updater.Store.Close() // Ensure database is closed
	UpdateCheckURLBase = server.URL + "/update.json"
	updatePresent, resp := updater.checkForUpdate(t.Context())
	if !updatePresent {
		t.Fatal("expected update to be available")
	}
	if resp.UpdateVersion != "9.9.9" {
		t.Fatal("unexpected response", "url", resp.UpdateURL, "version", resp.UpdateVersion)
	}
}

func TestBackgoundChecker(t *testing.T) {
	UpdateStageDir = t.TempDir()
	haveUpdate := false
	verified := false
	done := make(chan int)
	cb := func(ver string) error {
		haveUpdate = true
		done <- 0
		return nil
	}
	stallTimer := time.NewTimer(5 * time.Second)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func() error {
		verified = true
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else if r.URL.Path == "/9.9.9/"+Installer {
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{}}
	defer updater.Store.Close() // Ensure database is closed
	updater.StartBackgroundUpdaterChecker(ctx, cb)
	select {
	case <-stallTimer.C:
		t.Fatal("stalled")
	case <-done:
		if !haveUpdate {
			t.Fatal("no update received")
		}
		if !verified {
			t.Fatal("unverified")
		}
	}
}
