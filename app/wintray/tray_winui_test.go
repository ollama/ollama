//go:build windows && winui

package wintray

import (
	"testing"

	"golang.org/x/sys/windows"
)

func TestTrayPopupOffset(t *testing.T) {
	popup := windows.Rect{Left: 100, Top: 100, Right: 300, Bottom: 300}
	const overlap = int32(18)

	tests := []struct {
		name    string
		surface windows.Rect
		edge    trayEdge
		wantX   int32
		wantY   int32
		wantOK  bool
	}{
		{name: "bottom", surface: windows.Rect{Left: 0, Top: 280, Right: 500, Bottom: 330}, edge: trayEdgeBottom, wantY: 280 - popup.Bottom + overlap, wantOK: true},
		{name: "top", surface: windows.Rect{Left: 0, Top: 70, Right: 500, Bottom: 120}, edge: trayEdgeTop, wantY: 120 - popup.Top - overlap, wantOK: true},
		{name: "left", surface: windows.Rect{Left: 70, Top: 0, Right: 120, Bottom: 500}, edge: trayEdgeLeft, wantX: 120 - popup.Left - overlap, wantOK: true},
		{name: "right", surface: windows.Rect{Left: 280, Top: 0, Right: 330, Bottom: 500}, edge: trayEdgeRight, wantX: 280 - popup.Right + overlap, wantOK: true},
		{name: "no overlap", surface: windows.Rect{Left: 0, Top: 400, Right: 500, Bottom: 450}, edge: trayEdgeBottom},
		{name: "unknown edge", surface: windows.Rect{Left: 0, Top: 280, Right: 500, Bottom: 330}, edge: trayEdgeUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY, gotOK := trayPopupOffset(popup, tt.surface, tt.edge, overlap)
			if gotX != tt.wantX || gotY != tt.wantY || gotOK != tt.wantOK {
				t.Fatalf("trayPopupOffset() = (%d, %d, %t), want (%d, %d, %t)", gotX, gotY, gotOK, tt.wantX, tt.wantY, tt.wantOK)
			}
		})
	}
}

func TestUpdateNotificationBody(t *testing.T) {
	tests := []struct {
		name    string
		version string
		want    string
	}{
		{name: "pending update", want: "A new version of Ollama is ready to install"},
		{name: "versioned update", version: " 0.32.13 ", want: "Ollama version 0.32.13 is ready to install"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := updateNotificationBody(tt.version); got != tt.want {
				t.Fatalf("updateNotificationBody(%q) = %q, want %q", tt.version, got, tt.want)
			}
		})
	}
}
