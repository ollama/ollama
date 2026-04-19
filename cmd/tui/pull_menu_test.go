package tui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/api"
)

// serveModels starts an httptest server that returns the given models as a
// RemoteListResponse and overrides remoteModelsURL for the duration of the test.
func serveModels(t *testing.T, models []api.RemoteModel) {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(api.RemoteListResponse{Object: "list", Data: models})
	}))
	t.Cleanup(ts.Close)

	origURL := api.RemoteModelsURL
	api.RemoteModelsURL = ts.URL + "/v1/models"
	t.Cleanup(func() { api.RemoteModelsURL = origURL })
}

// sendKey drives a pullMenuModel through a single key press and returns the result.
func sendKey(m pullMenuModel, k tea.KeyType) (pullMenuModel, tea.Cmd) {
	updated, cmd := m.Update(tea.KeyMsg{Type: k})
	return updated.(pullMenuModel), cmd
}

// baseListModel returns a pullMenuModel already in pullStateBaseList with the
// given base models and their tags pre-populated.
func baseListModel(baseModels []SelectItem, tagsByBase map[string][]SelectItem) pullMenuModel {
	m := newPullMenuModel()
	m.state = pullStateBaseList
	m.baseModels = baseModels
	m.tagsByBase = tagsByBase
	m.baseSelector = selectorModel{
		title:    "Pull a Model",
		items:    baseModels,
		helpText: "↑/↓ navigate • enter select • → view tags • esc cancel",
	}
	return m
}

// --- fetchRemoteModelsCmd grouping logic ---

func TestFetchRemoteModels_GroupsTagsByBase(t *testing.T) {
	serveModels(t, []api.RemoteModel{
		{ID: "gemma3:4b"},
		{ID: "gemma3:12b"},
		{ID: "llama3:8b"},
	})

	msg := fetchRemoteModelsCmd().(remoteModelsMsg)
	if msg.err != nil {
		t.Fatalf("unexpected error: %v", msg.err)
	}
	if len(msg.baseModels) != 2 {
		t.Fatalf("expected 2 base models, got %d", len(msg.baseModels))
	}
	if len(msg.tagsByBase["gemma3"]) != 2 {
		t.Errorf("expected 2 tags for gemma3, got %d", len(msg.tagsByBase["gemma3"]))
	}
	if len(msg.tagsByBase["llama3"]) != 1 {
		t.Errorf("expected 1 tag for llama3, got %d", len(msg.tagsByBase["llama3"]))
	}
}

func TestFetchRemoteModels_SingleTagNoDescription(t *testing.T) {
	serveModels(t, []api.RemoteModel{{ID: "llama3:8b"}})

	msg := fetchRemoteModelsCmd().(remoteModelsMsg)
	if msg.err != nil {
		t.Fatalf("unexpected error: %v", msg.err)
	}
	if len(msg.baseModels) != 1 {
		t.Fatalf("expected 1 base model, got %d", len(msg.baseModels))
	}
	if msg.baseModels[0].Description != "" {
		t.Errorf("expected empty description for single-tag model, got %q", msg.baseModels[0].Description)
	}
}

func TestFetchRemoteModels_MultiTagHasDescription(t *testing.T) {
	serveModels(t, []api.RemoteModel{
		{ID: "gemma3:4b"},
		{ID: "gemma3:12b"},
	})

	msg := fetchRemoteModelsCmd().(remoteModelsMsg)
	if msg.err != nil {
		t.Fatalf("unexpected error: %v", msg.err)
	}
	desc := msg.baseModels[0].Description
	if !strings.Contains(desc, "4b") || !strings.Contains(desc, "12b") {
		t.Errorf("expected description to contain tag variants, got %q", desc)
	}
}

func TestFetchRemoteModels_EmptyIDSkipped(t *testing.T) {
	serveModels(t, []api.RemoteModel{
		{ID: ""},
		{ID: "llama3:8b"},
	})

	msg := fetchRemoteModelsCmd().(remoteModelsMsg)
	if msg.err != nil {
		t.Fatalf("unexpected error: %v", msg.err)
	}
	if len(msg.baseModels) != 1 {
		t.Errorf("expected 1 base model (empty ID skipped), got %d", len(msg.baseModels))
	}
}

func TestFetchRemoteModels_Timeout(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done() // block until request context is cancelled
	}))
	t.Cleanup(ts.Close)

	origURL := api.RemoteModelsURL
	api.RemoteModelsURL = ts.URL + "/v1/models"
	t.Cleanup(func() { api.RemoteModelsURL = origURL })

	origTimeout := pullMenuTimeout
	pullMenuTimeout = 1 * time.Millisecond
	t.Cleanup(func() { pullMenuTimeout = origTimeout })

	msg := fetchRemoteModelsCmd().(remoteModelsMsg)
	if msg.err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}

// --- pullMenuModel.Update() state transitions ---

func TestPullMenu_LoadingToBaseList(t *testing.T) {
	m := newPullMenuModel()
	baseModels := []SelectItem{{Name: "llama3"}}
	tagsByBase := map[string][]SelectItem{"llama3": {{Name: "llama3:8b"}}}

	updated, _ := m.Update(remoteModelsMsg{baseModels: baseModels, tagsByBase: tagsByBase})
	fm := updated.(pullMenuModel)

	if fm.state != pullStateBaseList {
		t.Errorf("expected pullStateBaseList, got %v", fm.state)
	}
}

func TestPullMenu_LoadingToError(t *testing.T) {
	m := newPullMenuModel()

	updated, _ := m.Update(remoteModelsMsg{err: errTest})
	fm := updated.(pullMenuModel)

	if fm.state != pullStateError {
		t.Errorf("expected pullStateError, got %v", fm.state)
	}
}

func TestPullMenu_EscCancels(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "llama3"}},
		map[string][]SelectItem{"llama3": {{Name: "llama3:8b"}}},
	)

	fm, cmd := sendKey(m, tea.KeyEsc)

	if fm.selected != "" {
		t.Errorf("expected empty selected, got %q", fm.selected)
	}
	if cmd == nil {
		t.Error("expected tea.Quit cmd, got nil")
	}
}

func TestPullMenu_EnterSingleTagPulls(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "llama3"}},
		map[string][]SelectItem{"llama3": {{Name: "llama3:8b"}}},
	)

	fm, cmd := sendKey(m, tea.KeyEnter)

	if fm.selected != "llama3:8b" {
		t.Errorf("expected selected %q, got %q", "llama3:8b", fm.selected)
	}
	if cmd == nil {
		t.Error("expected tea.Quit cmd, got nil")
	}
}

func TestPullMenu_EnterMultiTagGoesToTagList(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "gemma3"}},
		map[string][]SelectItem{"gemma3": {{Name: "gemma3:4b"}, {Name: "gemma3:12b"}}},
	)

	fm, _ := sendKey(m, tea.KeyEnter)

	if fm.state != pullStateTagList {
		t.Errorf("expected pullStateTagList, got %v", fm.state)
	}
}

func TestPullMenu_RightArrowGoesToTagList(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "gemma3"}},
		map[string][]SelectItem{"gemma3": {{Name: "gemma3:4b"}, {Name: "gemma3:12b"}}},
	)

	fm, _ := sendKey(m, tea.KeyRight)

	if fm.state != pullStateTagList {
		t.Errorf("expected pullStateTagList, got %v", fm.state)
	}
}

func TestPullMenu_LeftArrowGoesBack(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "gemma3"}},
		map[string][]SelectItem{"gemma3": {{Name: "gemma3:4b"}, {Name: "gemma3:12b"}}},
	)
	m, _ = sendKey(m, tea.KeyRight)
	if m.state != pullStateTagList {
		t.Fatalf("setup failed: expected pullStateTagList")
	}

	fm, _ := sendKey(m, tea.KeyLeft)

	if fm.state != pullStateBaseList {
		t.Errorf("expected pullStateBaseList, got %v", fm.state)
	}
}

func TestPullMenu_EnterInTagListPulls(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "gemma3"}},
		map[string][]SelectItem{"gemma3": {{Name: "gemma3:4b"}, {Name: "gemma3:12b"}}},
	)
	m, _ = sendKey(m, tea.KeyRight)

	fm, cmd := sendKey(m, tea.KeyEnter)

	if fm.selected != "gemma3:4b" {
		t.Errorf("expected selected %q, got %q", "gemma3:4b", fm.selected)
	}
	if cmd == nil {
		t.Error("expected tea.Quit cmd, got nil")
	}
}

func TestPullMenu_EscInTagListCancels(t *testing.T) {
	m := baseListModel(
		[]SelectItem{{Name: "gemma3"}},
		map[string][]SelectItem{"gemma3": {{Name: "gemma3:4b"}, {Name: "gemma3:12b"}}},
	)
	m, _ = sendKey(m, tea.KeyRight)

	fm, cmd := sendKey(m, tea.KeyEsc)

	if fm.selected != "" {
		t.Errorf("expected empty selected, got %q", fm.selected)
	}
	if cmd == nil {
		t.Error("expected tea.Quit cmd, got nil")
	}
}

func TestPullMenu_WindowSizeUpdatesWidth(t *testing.T) {
	m := newPullMenuModel()

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	fm := updated.(pullMenuModel)

	if fm.width != 120 {
		t.Errorf("expected width 120, got %d", fm.width)
	}
}

// --- pullMenuModel.View() rendering ---

func TestPullMenuView_LoadingContainsFetching(t *testing.T) {
	m := newPullMenuModel()
	view := m.View()
	if !strings.Contains(view, "Fetching available models") {
		t.Errorf("loading view missing fetch message, got:\n%s", view)
	}
}

func TestPullMenuView_ErrorContainsMessage(t *testing.T) {
	m := newPullMenuModel()
	m.state = pullStateError
	m.err = errTest

	view := m.View()
	if !strings.Contains(view, "Could not fetch model list") {
		t.Errorf("error view missing error message, got:\n%s", view)
	}
	if !strings.Contains(view, "ollama pull") {
		t.Errorf("error view missing fallback instruction, got:\n%s", view)
	}
}

func TestPullMenuView_SelectedReturnsEmpty(t *testing.T) {
	m := newPullMenuModel()
	m.selected = "gemma3:4b"

	if view := m.View(); view != "" {
		t.Errorf("expected empty view when selected, got:\n%s", view)
	}
}

// errTest is a sentinel error for use in tests.
var errTest = fmt.Errorf("test error")
