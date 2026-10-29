//go:build windows

package main

import (
	"encoding/xml"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSettingsPage(t *testing.T) {
	page, err := settingsPage(`C:\Program Files\R&D Tools\app.ico`)
	if err != nil {
		t.Fatal(err)
	}

	// The page must be well-formed XML, with the icon path escaped.
	decoder := xml.NewDecoder(strings.NewReader(page))
	for {
		if _, err := decoder.Token(); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatalf("page is not well-formed XML: %v", err)
		}
	}
	if !strings.Contains(page, `file:///C:/Program%20Files/R&amp;D%20Tools/app.ico`) {
		t.Errorf("icon URL is missing or unescaped")
	}

	// Each context length is a ComboBoxItem, in order, starting with
	// Automatic, so the selected index maps to contextLengths.
	if got := strings.Count(page, "<ComboBoxItem"); got != len(contextLengths) {
		t.Errorf("page has %d context lengths, want %d", got, len(contextLengths))
	}
	first := page[strings.Index(page, "<ComboBoxItem"):]
	first = first[:strings.Index(first, "/>")]
	if !strings.Contains(first, `x:Name="ContextLengthAutomatic"`) || !strings.Contains(first, `Content="Automatic"`) {
		t.Errorf("first context length = %q, want Automatic", first)
	}
	if !strings.Contains(page, `Content="256K"`) {
		t.Errorf("page is missing the 256K context length")
	}
}

func TestSettingsPageWithoutIcon(t *testing.T) {
	page, err := settingsPage("")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(page, "ImageSource=") || strings.Contains(page, "<Image ") {
		t.Errorf("page refers to an icon although there is none")
	}
}

func TestContextLengthTitle(t *testing.T) {
	for length, want := range map[int]string{4096: "4K", 32768: "32K", 262144: "256K"} {
		if got := contextLengthTitle(length); got != want {
			t.Errorf("contextLengthTitle(%d) = %q, want %q", length, got, want)
		}
	}
}

func TestAvailablePath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Ollama Chats")
	if got := availablePath(path); got != path {
		t.Fatalf("availablePath = %q, want %q", got, path)
	}
	for _, name := range []string{"Ollama Chats", "Ollama Chats (2)"} {
		if err := os.Mkdir(filepath.Join(dir, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if got, want := availablePath(path), path+" (3)"; got != want {
		t.Fatalf("availablePath = %q, want %q", got, want)
	}
}

func TestLoginItem(t *testing.T) {
	// Use a shortcut name that no real Startup folder entry has.
	app, startup := t.TempDir(), t.TempDir()
	previousApp, previousShortcut := appPath, startupShortcut
	appPath, startupShortcut = app, filepath.Join(startup, "Ollama Settings Test.lnk")
	t.Cleanup(func() { appPath, startupShortcut = previousApp, previousShortcut })

	if got := getLoginItemStatus(); got != loginItemUnavailable {
		t.Fatalf("without the installer's shortcut, status = %v", got)
	}

	if err := os.MkdirAll(filepath.Dir(installedShortcut()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(installedShortcut(), []byte("shortcut"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := getLoginItemStatus(); got != loginItemOff {
		t.Fatalf("before turning it on, status = %v", got)
	}

	if err := setLoginItemEnabled(true); err != nil {
		t.Fatal(err)
	}
	if got := getLoginItemStatus(); got != loginItemOn {
		t.Fatalf("after turning it on, status = %v", got)
	}
	if data, err := os.ReadFile(startupShortcut); err != nil || string(data) != "shortcut" {
		t.Fatalf("startup shortcut = %q, %v", data, err)
	}

	if err := setLoginItemEnabled(false); err != nil {
		t.Fatal(err)
	}
	if got := getLoginItemStatus(); got != loginItemOff {
		t.Fatalf("after turning it off, status = %v", got)
	}
	if err := setLoginItemEnabled(false); err != nil {
		t.Fatalf("turning it off twice: %v", err)
	}
}
