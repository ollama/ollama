// routes_slot.go — Ollama HTTP API for persisting and restoring per-slot KV
// cache state from disk.
//
// These endpoints are a thin proxy over upstream llama-server's slot
// save/restore HTTP API (`POST /slots/{id}?action=save|restore|erase`), which
// only becomes available when llama-server is invoked with `--slot-save-path
// <dir>`. Ollama threads that flag through from the OLLAMA_SLOT_SAVE_PATH
// environment variable (see envconfig and llm/llama_server.go).
//
// Wire format (Ollama side):
//
//	POST /api/slot/save    { "model": "...", "id_slot": 0, "filename": "..." }
//	POST /api/slot/restore { "model": "...", "id_slot": 0, "filename": "..." }
//	POST /api/slot/erase   { "model": "...", "id_slot": 0 }
//
// The handler schedules the model runner the same way /api/generate does, then
// forwards the action to the per-runner llama-server port. `filename` is
// passed through verbatim — the llama-server side enforces "no path
// separators" so the file lands directly under the configured save dir.
//
// When OLLAMA_SLOT_SAVE_PATH is unset, llama-server rejects the slots-action
// route with a NOT_SUPPORTED error, which is surfaced to the caller as 501.
//
// Background: this exposes the existing upstream API. The LuminaNAO LSCKPT2
// sidecar (which persists *prompt-cache checkpoints* alongside the KV bytes,
// for log-spaced reuse of long prompts) lives further upstream in llama.cpp
// and flows into Ollama automatically when LLAMA_CPP_VERSION is bumped to a
// commit that includes it. See LuminaNAO commit 72fbd6a "Thin slot-save
// checkpoint sidecars geometrically" and surrounding history.

package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// SlotSaveRequest is the body for /api/slot/save and /api/slot/restore.
// Filename is the basename (no path separators) written under
// OLLAMA_SLOT_SAVE_PATH. IDSlot defaults to 0 when omitted.
type SlotSaveRequest struct {
	Model    string `json:"model"`
	IDSlot   int    `json:"id_slot"`
	Filename string `json:"filename"`
}

// SlotEraseRequest is the body for /api/slot/erase.
type SlotEraseRequest struct {
	Model  string `json:"model"`
	IDSlot int    `json:"id_slot"`
}

// SlotResponse mirrors the upstream llama-server slot save/restore reply so
// callers see consistent fields (id_slot, filename, n_saved/n_restored,
// n_written/n_read, timings) regardless of whether the LSCKPT2 sidecar is
// present. Extra fields from a sidecar-aware upstream pass through as
// json.RawMessage in Extras.
type SlotResponse struct {
	IDSlot     int             `json:"id_slot"`
	Filename   string          `json:"filename,omitempty"`
	NSaved     int             `json:"n_saved,omitempty"`
	NRestored  int             `json:"n_restored,omitempty"`
	NWritten   int             `json:"n_written,omitempty"`
	NRead      int             `json:"n_read,omitempty"`
	TimingsRaw json.RawMessage `json:"timings,omitempty"`
}

// SlotSaveHandler proxies POST /api/slot/save to llama-server's
// /slots/{id}?action=save. The model is scheduled (loading it if necessary)
// so the slot KV state is captured against the live runner.
func (s *Server) SlotSaveHandler(c *gin.Context) {
	s.slotActionHandler(c, "save")
}

// SlotRestoreHandler proxies POST /api/slot/restore. The model is scheduled
// first; llama-server resizes the slot's KV cache to match the file on disk.
func (s *Server) SlotRestoreHandler(c *gin.Context) {
	s.slotActionHandler(c, "restore")
}

// SlotEraseHandler proxies POST /api/slot/erase. No file I/O; clears the
// runtime slot state so the next completion starts cold.
func (s *Server) SlotEraseHandler(c *gin.Context) {
	s.slotActionHandler(c, "erase")
}

func (s *Server) slotActionHandler(c *gin.Context, action string) {
	if envconfig.SlotSavePath() == "" {
		c.JSON(http.StatusNotImplemented, gin.H{
			"error": "slot persistence disabled; set OLLAMA_SLOT_SAVE_PATH to enable",
		})
		return
	}

	var (
		modelName string
		idSlot    int
		filename  string
		body      gin.H
	)

	switch action {
	case "save", "restore":
		var req SlotSaveRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if req.Filename == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "filename is required"})
			return
		}
		modelName = req.Model
		idSlot = req.IDSlot
		filename = req.Filename
		body = gin.H{"filename": filename}
	case "erase":
		var req SlotEraseRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		modelName = req.Model
		idSlot = req.IDSlot
		body = gin.H{}
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown slot action"})
		return
	}

	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if idSlot < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "id_slot must be >= 0"})
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), modelName, []model.Capability{}, nil, nil, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	port := r.GetPort()
	url := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=%s",
		port, idSlot, action)

	payload, err := json.Marshal(body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	hreq, err := http.NewRequestWithContext(c.Request.Context(), "POST", url, bytes.NewReader(payload))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Content-Length", strconv.Itoa(len(payload)))

	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), raw)
}
