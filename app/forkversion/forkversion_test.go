package forkversion

import (
	"testing"
	"time"
)

func TestBuildTag(t *testing.T) {
	for _, tt := range []struct{ version, want string }{
		{"0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload-0-g6c178a3", "0.35.1-b11325-glm5next-unload"},
		{"v0.35.1-b11325-glm5next-unload-12-gabcdef0", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload-dirty", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-rc0-3-g1234567", "0.35.1-rc0"},
		{"0.0.0", "0.0.0"},
		{"6c178a3-dirty", "6c178a3"},
	} {
		if got := BuildTag(tt.version); got != tt.want {
			t.Errorf("BuildTag(%q) = %q, want %q", tt.version, got, tt.want)
		}
	}
}

func TestIsUpdate(t *testing.T) {
	installed := time.Date(2026, 10, 7, 16, 24, 0, 0, time.UTC)
	before := installed.Add(-time.Hour)
	after := installed.Add(time.Hour)
	const current = "0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty"

	for _, tt := range []struct {
		name      string
		tag       string
		current   string
		published time.Time
		want      bool
	}{
		{"same tag as build", "v0.35.1-b11325-glm5next-unload", current, after, false},
		{"exact tag build", "v0.35.1-b11325-glm5next-unload", "0.35.1-b11325-glm5next-unload-0-g1234567", after, false},
		{"older suffix published before install", "v0.35.1-b11325-glm5next", current, before, false},
		{"prefix of build tag published later", "v0.35.1-b11325-glm5next", current, after, true},
		{"new suffix published later", "v0.35.1-b11325-glm5next-keepalive", current, after, true},
		{"new suffix published before install", "v0.35.1-b11325-glm5next-keepalive", current, before, false},
		{"new suffix without publish time", "v0.35.1-b11325-glm5next-keepalive", current, time.Time{}, true},
		{"newer patch", "v0.35.2-b11325-glm5next", current, before, true},
		{"newer minor", "v0.40.0-b11400-glm5next", current, before, true},
		{"newer llama rev", "v0.35.1-b11400-glm5next", current, before, true},
		{"older patch", "v0.35.0-b11325-glm5next", current, after, false},
		{"older llama rev", "v0.35.1-b11000-glm5next", current, after, false},
		{"numeric not lexical", "v0.35.10-b11325-x", "0.35.9-b11325-x-1-gabcdef0", before, true},
		{"dev build", "v0.35.1-b11325-glm5next-unload", "0.0.0", before, true},
		{"empty tag", "", current, after, false},
		{"ds41 bump published before install", "v0.40.5-b11724-ds41.4", "0.40.5-b11724-ds41.3-0-gabcdef0", before, true},
		{"ds41 older number", "v0.40.5-b11724-ds41.3", "0.40.5-b11724-ds41.4-0-gabcdef0", after, false},
		{"ds41.10 beats ds41.9", "v0.40.5-b11724-ds41.10", "0.40.5-b11724-ds41.9-0-gabcdef0", before, true},
		{"same ds41 tag", "v0.40.5-b11724-ds41.3", "0.40.5-b11724-ds41.3-0-gabcdef0", after, false},
		{"numbered ds41 beats bare ds41", "v0.40.5-b11724-ds41.1", "0.40.5-b11724-ds41-0-gabcdef0", before, true},
		{"bare ds41 does not beat ds41.1", "v0.40.5-b11724-ds41", "0.40.5-b11724-ds41.1-0-gabcdef0", after, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsUpdate(tt.tag, tt.current, tt.published, installed); got != tt.want {
				t.Errorf("IsUpdate(%q, %q) = %v, want %v", tt.tag, tt.current, got, tt.want)
			}
		})
	}
}

func TestNext(t *testing.T) {
	tags := []string{
		"v0.35.1-b11325-glm5next-unload",
		"v0.40.5-b11724-ds41",
		"v0.40.5-b11724-ds41.1",
		"v0.40.5-b11724-ds41.3",
		"v0.40.5-b11724-ds41.2",
	}
	latest, err := Latest(tags)
	if err != nil {
		t.Fatal(err)
	}
	if latest != "v0.40.5-b11724-ds41.3" {
		t.Fatalf("latest = %s", latest)
	}
	next, err := Next("patch", tags)
	if err != nil {
		t.Fatal(err)
	}
	if next != "v0.40.5-b11724-ds41.4" {
		t.Fatalf("patch = %s", next)
	}
	explicit, err := Next("v0.40.5-b11724-ds41.9", tags)
	if err != nil {
		t.Fatal(err)
	}
	if explicit != "v0.40.5-b11724-ds41.9" {
		t.Fatalf("explicit = %s", explicit)
	}
	if _, err := Next("v0.40.5-b11724-ds41.3", tags); err == nil {
		t.Fatal("existing tag was accepted")
	}
	if _, err := Next("v0.40.5-b11724-ds41.1", tags); err == nil {
		t.Fatal("older tag was accepted")
	}
	if _, err := Next("1.2.3", tags); err == nil {
		t.Fatal("non-ds41 version was accepted")
	}
	if _, err := Next("patch", nil); err == nil {
		t.Fatal("patch with no tags was accepted")
	}
	bare, err := Next("v0.40.5-b11724-ds41", []string{"v0.40.5-b11724-ds41"})
	if err == nil {
		t.Fatalf("same bare tag accepted as %s", bare)
	}
	bumped, err := Next("patch", []string{"v0.40.5-b11724-ds41"})
	if err != nil {
		t.Fatal(err)
	}
	if bumped != "v0.40.5-b11724-ds41.1" {
		t.Fatalf("bump bare ds41 = %s", bumped)
	}
}
