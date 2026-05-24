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
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestUpdateStagePathRejectsUnsafeFilename(t *testing.T) {
	stageDir := t.TempDir()
	for _, tt := range []struct {
		name     string
		filename string
	}{
		{"empty", ""},
		{"dot", "."},
		{"dotdot", ".."},
		{"posix_parent", "../OllamaSetup.exe"},
		{"windows_parent", `..\OllamaSetup.exe`},
		{"posix_absolute_tmp", "/tmp/OllamaSetup.exe"},
		{"darwin_absolute_app", "/Applications/Ollama.app"},
		{"darwin_bundle_path", "Ollama.app/Contents/MacOS/Ollama"},
		{"darwin_user_download", "~/Downloads/Ollama-darwin.zip"},
		{"windows_absolute", `C:\Users\Public\OllamaSetup.exe`},
		{"colon", "Ollama:Setup.exe"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := updateStagePath(stageDir, "etag", tt.filename); err == nil {
				t.Fatal("expected unsafe filename to be rejected")
			}
		})
	}
}

func TestUpdateStagePathHashesETag(t *testing.T) {
	stageDir := t.TempDir()
	stageFilename, err := updateStagePath(stageDir, `../escaped`, "OllamaSetup.exe")
	if err != nil {
		t.Fatal(err)
	}

	rel, err := filepath.Rel(stageDir, stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		t.Fatalf("stage filename escaped stage dir: %s", stageFilename)
	}
	etagDir := filepath.Base(filepath.Dir(stageFilename))
	if etagDir == ".." || etagDir == "escaped" || strings.ContainsAny(etagDir, `/\`) {
		t.Fatalf("stage filename used raw etag path component: %s", stageFilename)
	}
}

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

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
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

func TestDownloadNewReleaseRejectsUnsafeHeaderFilename(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldInstaller := Installer
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	defer func() {
		Installer = oldInstaller
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
	}()
	Installer = "OllamaSetup.exe"
	UpdateDownloaded = false
	VerifyDownload = func() error {
		t.Fatal("verification should not run for rejected downloads")
		return nil
	}

	var getAttempted atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("ETag", `"safe"`)
			w.Header().Set("Content-Disposition", `attachment; filename="../OllamaSetup.exe"`)
			w.WriteHeader(http.StatusOK)
			return
		}
		getAttempted.Store(true)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{UpdateURL: server.URL + "/download"})
	if err == nil || !strings.Contains(err.Error(), "unsafe update filename") {
		t.Fatalf("expected unsafe filename error, got %v", err)
	}
	if getAttempted.Load() {
		t.Fatal("download should not continue after unsafe filename")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(UpdateStageDir), "OllamaSetup.exe")); err == nil {
		t.Fatal("download escaped update stage dir")
	}
}

func TestDownloadNewReleaseDoesNotUseRawETagAsPathComponent(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldInstaller := Installer
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	defer func() {
		Installer = oldInstaller
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
	}()
	Installer = "OllamaSetup.exe"
	UpdateDownloaded = false
	VerifyDownload = func() error {
		return nil
	}

	payload := []byte("payload")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"../escaped"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write(payload)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	if err := updater.DownloadNewRelease(t.Context(), UpdateResponse{UpdateURL: server.URL + "/download"}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(filepath.Join(filepath.Dir(UpdateStageDir), "escaped", Installer)); err == nil {
		t.Fatal("download escaped update stage dir via etag")
	}

	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected one staged update dir, got %d", len(entries))
	}
	stageFilename := filepath.Join(UpdateStageDir, entries[0].Name(), Installer)
	got, err := os.ReadFile(stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("unexpected staged payload %q", got)
	}
}

func TestBackgroundCheckerSkipsAlreadyStagedETagDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldInstaller := Installer
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	oldUpdateCheckInitialDelay := UpdateCheckInitialDelay
	oldUpdateCheckInterval := UpdateCheckInterval
	oldUpdateCheckURLBase := UpdateCheckURLBase
	defer func() {
		Installer = oldInstaller
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
		UpdateCheckInitialDelay = oldUpdateCheckInitialDelay
		UpdateCheckInterval = oldUpdateCheckInterval
		UpdateCheckURLBase = oldUpdateCheckURLBase
	}()
	Installer = "OllamaSetup.exe"
	UpdateDownloaded = false
	UpdateCheckInitialDelay = time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond

	var verifyCount atomic.Int32
	VerifyDownload = func() error {
		verifyCount.Add(1)
		return nil
	}

	headETag := `"old-update"`
	getETag := `"download-response-etag"`
	payload := []byte("payload")
	var headCount atomic.Int32
	var getCount atomic.Int32
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/update.json":
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		case "/9.9.9/" + Installer:
			w.Header().Set("Content-Disposition", `attachment; filename="OllamaSetup.exe"`)
			switch r.Method {
			case http.MethodHead:
				etag := headETag
				if getCount.Load() > 0 {
					etag = getETag
				}
				w.Header().Set("ETag", etag)
				headCount.Add(1)
				w.WriteHeader(http.StatusOK)
			case http.MethodGet:
				w.Header().Set("ETag", getETag)
				getCount.Add(1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(payload)
			default:
				t.Errorf("unexpected request method %s", r.Method)
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		default:
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	callbacks := make(chan string, 4)
	updater.StartBackgroundUpdaterChecker(ctx, func(ver string) error {
		callbacks <- ver
		return nil
	})

	for range 2 {
		select {
		case <-callbacks:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for repeated update checks")
		}
	}
	cancel()

	stageFilename, err := updateStagePath(UpdateStageDir, getETag, Installer)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("unexpected staged payload %q", got)
	}

	if headCount.Load() < 2 {
		t.Fatalf("HEAD count = %d, want at least 2", headCount.Load())
	}
	if getCount.Load() != 1 {
		t.Fatalf("GET count = %d, want 1", getCount.Load())
	}
	if verifyCount.Load() != 1 {
		t.Fatalf("verification count = %d, want 1", verifyCount.Load())
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should stay true for already staged update")
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

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

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

func TestAutoUpdateDisabledSkipsDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	done := make(chan struct{})

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func() error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	// Ensure auto-update is disabled
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		t.Fatal("callback should not be called when auto-update is disabled")
		return nil
	}

	updater.StartBackgroundUpdaterChecker(ctx, cb)

	// Wait enough time for multiple check cycles
	time.Sleep(50 * time.Millisecond)
	close(done)

	if downloadAttempted.Load() {
		t.Fatal("download should not be attempted when auto-update is disabled")
	}
}

func TestAutoUpdateReenabledDownloadsUpdate(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	callbackCalled := make(chan struct{}, 1)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func() error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	upd := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer upd.Store.Close()

	// Start with auto-update disabled
	settings, err := upd.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		select {
		case callbackCalled <- struct{}{}:
		default:
		}
		return nil
	}

	upd.StartBackgroundUpdaterChecker(ctx, cb)

	// Wait for a few cycles with auto-update disabled - no download should happen
	time.Sleep(50 * time.Millisecond)
	if downloadAttempted.Load() {
		t.Fatal("download should not happen while auto-update is disabled")
	}

	// Re-enable auto-update
	settings.AutoUpdateEnabled = true
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	// Wait for the checker to pick it up and download
	select {
	case <-callbackCalled:
		// Success: download happened and callback was called after re-enabling
		if !downloadAttempted.Load() {
			t.Fatal("expected download to be attempted after re-enabling")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("expected download and callback after re-enabling auto-update")
	}
}

func TestCancelOngoingDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	downloadStarted := make(chan struct{})
	downloadCancelled := make(chan struct{})

	ctx := t.Context()
	VerifyDownload = func() error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			if r.Method == http.MethodHead {
				w.Header().Set("Content-Length", "1000000")
				w.WriteHeader(http.StatusOK)
				return
			}
			// Signal that download has started
			close(downloadStarted)
			// Wait for cancellation or timeout
			select {
			case <-r.Context().Done():
				close(downloadCancelled)
				return
			case <-time.After(5 * time.Second):
				t.Error("download was not cancelled in time")
			}
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	_, resp := updater.checkForUpdate(ctx)

	// Start download in goroutine
	go func() {
		_ = updater.DownloadNewRelease(ctx, resp)
	}()

	// Wait for download to start
	select {
	case <-downloadStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("download did not start in time")
	}

	// Cancel the download
	updater.CancelOngoingDownload()

	// Verify cancellation was received
	select {
	case <-downloadCancelled:
		// Success
	case <-time.After(2 * time.Second):
		t.Fatal("download cancellation was not received by server")
	}
}

func TestTriggerImmediateCheck(t *testing.T) {
	UpdateStageDir = t.TempDir()
	checkCount := atomic.Int32{}
	checkDone := make(chan struct{}, 10)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	// Set a very long interval so only TriggerImmediateCheck causes checks
	UpdateCheckInitialDelay = 1 * time.Millisecond
	UpdateCheckInterval = 1 * time.Hour
	VerifyDownload = func() error {
		return nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			checkCount.Add(1)
			select {
			case checkDone <- struct{}{}:
			default:
			}
			// Return no update available
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	cb := func(ver string) error {
		return nil
	}

	updater.StartBackgroundUpdaterChecker(ctx, cb)

	// Wait for the initial check that fires after the initial delay
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("initial check did not happen")
	}

	initialCount := checkCount.Load()

	// Trigger immediate check
	updater.TriggerImmediateCheck()

	// Wait for the triggered check
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("triggered check did not happen")
	}

	finalCount := checkCount.Load()
	if finalCount <= initialCount {
		t.Fatalf("TriggerImmediateCheck did not cause additional check: initial=%d, final=%d", initialCount, finalCount)
	}
}
