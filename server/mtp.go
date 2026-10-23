package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const (
	draftNumPredictKey = "draft_num_predict"

	// Draft depths the app offers. llama-server receives the value as
	// --spec-draft-n-max. Zero removes the parameter, which leaves multi-token
	// prediction off for models that do not have a separate draft model.
	minDraftNumPredict = 1
	maxDraftNumPredict = 4
)

var errDraftNumPredictRange = errors.New("draft_num_predict must be 0 to turn off, or from 1 to 4")

// metadataHasMTPLayers reports whether GGUF metadata (or the model_info map
// /api/show returns from it) records nextn prediction layers. The key is
// stored as "<architecture>.nextn_predict_layers".
func metadataHasMTPLayers(kv map[string]any) bool {
	for key, value := range kv {
		if key != "nextn_predict_layers" && !strings.HasSuffix(key, ".nextn_predict_layers") {
			continue
		}
		if numberPositive(value) {
			return true
		}
	}
	return false
}

// modelSupportsMTP reports whether this model can run embedded multi-token
// prediction. Remote and cloud models cannot. A zero or missing
// nextn_predict_layers value does not count.
func modelSupportsMTP(m *Model) bool {
	if m == nil || m.Config.RemoteHost != "" {
		return false
	}
	if metadataHasMTPLayers(m.metadata.KV) {
		return true
	}
	return m.metadata.Int("nextn_predict_layers") > 0
}

func numberPositive(v any) bool {
	switch n := v.(type) {
	case json.Number:
		f, err := n.Float64()
		return err == nil && f > 0 && !math.IsInf(f, 0)
	case float64:
		return n > 0 && !math.IsNaN(n) && !math.IsInf(n, 0)
	case float32:
		return n > 0 && !math.IsNaN(float64(n)) && !math.IsInf(float64(n), 0)
	case int:
		return n > 0
	case int32:
		return n > 0
	case int64:
		return n > 0
	case uint:
		return n > 0
	case uint32:
		return n > 0
	case uint64:
		return n > 0
	default:
		return false
	}
}

// draftNumPredictValue returns the stored draft depth. The boolean is false
// when the parameter is absent.
func draftNumPredictValue(params map[string]any) (int, bool) {
	value, ok := params[draftNumPredictKey]
	if !ok || value == nil {
		return 0, false
	}
	n, ok := wholeNumber(value)
	if !ok {
		// Present, but not a whole number. Callers still treat the key as set
		// so turning the feature off removes it.
		return 0, true
	}
	return n, true
}

func wholeNumber(v any) (int, bool) {
	switch n := v.(type) {
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
		f, err := n.Float64()
		if err != nil || f != math.Trunc(f) || math.IsInf(f, 0) || math.IsNaN(f) {
			return 0, false
		}
		return int(f), true
	case float64:
		if math.IsNaN(n) || math.IsInf(n, 0) || n != math.Trunc(n) {
			return 0, false
		}
		return int(n), true
	case float32:
		f := float64(n)
		if math.IsNaN(f) || math.IsInf(f, 0) || f != math.Trunc(f) {
			return 0, false
		}
		return int(f), true
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		if n > uint64(math.MaxInt) {
			return 0, false
		}
		return int(n), true
	default:
		return 0, false
	}
}

// applyDraftNumPredict sets or removes draft_num_predict in a params-layer
// JSON object. Other keys keep their original JSON literals. n == 0 removes
// the key. A no-op returns the original bytes and changed == false.
func applyDraftNumPredict(raw []byte, n int) ([]byte, bool, error) {
	if n != 0 && (n < minDraftNumPredict || n > maxDraftNumPredict) {
		return nil, false, errDraftNumPredictRange
	}

	params := map[string]any{}
	if len(bytes.TrimSpace(raw)) > 0 {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if err := dec.Decode(&params); err != nil {
			return nil, false, fmt.Errorf("decode model parameters: %w", err)
		}
		if params == nil {
			params = map[string]any{}
		}
	}

	current, present := draftNumPredictValue(params)
	if n == 0 {
		if !present {
			return raw, false, nil
		}
		delete(params, draftNumPredictKey)
	} else if present && current == n {
		return raw, false, nil
	} else {
		params[draftNumPredictKey] = n
	}

	encoded, err := marshalParams(params)
	if err != nil {
		return nil, false, err
	}
	return encoded, true, nil
}

func marshalParams(params map[string]any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(params); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// setModelDraftNumPredict writes draft_num_predict onto the named model's
// parameter layer. Other manifest layers (weights, template, system, license,
// messages) are left in place, and other parameter keys are preserved.
// Manifest lists get the same edit on every local child so the setting does
// not depend on which runner is selected. A missing child is left as-is.
func setModelDraftNumPredict(name model.Name, n int) error {
	if !name.IsFullyQualified() {
		return model.Unqualified(name)
	}
	if n != 0 && (n < minDraftNumPredict || n > maxDraftNumPredict) {
		return errDraftNumPredictRange
	}

	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return err
	}
	var doc manifest.Manifest
	if err := json.Unmarshal(data, &doc); err != nil {
		return err
	}

	if doc.MediaType == manifest.MediaTypeManifestList {
		changed := false
		for i := range doc.Manifests {
			childChanged, err := updateListChild(&doc.Manifests[i], n)
			if err != nil {
				return err
			}
			if childChanged {
				changed = true
			}
		}
		if !changed {
			return nil
		}
		updated, err := json.Marshal(doc)
		if err != nil {
			return err
		}
		// Keep any legacy manifest path. The server reads the v2 manifest,
		// and removing the legacy file would drop the downgrade anchor.
		return manifest.WriteManifestDataPreserveLegacy(name, updated)
	}

	updatedDoc, changed, err := updateRunnableManifest(doc, n)
	if err != nil || !changed {
		return err
	}
	updated, err := json.Marshal(updatedDoc)
	if err != nil {
		return err
	}
	return manifest.WriteManifestDataPreserveLegacy(name, updated)
}

func updateListChild(child *manifest.Manifest, n int) (bool, error) {
	if child.MediaType == manifest.MediaTypeManifestList {
		return false, errors.New("nested manifest lists are not supported")
	}

	full := *child
	reference := child.Config.Digest == "" && child.BlobDigest() != ""
	if reference {
		loaded, err := readManifestAtDigest(child.BlobDigest())
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		full = loaded
	}

	updated, changed, err := updateRunnableManifest(full, n)
	if err != nil || !changed {
		return false, err
	}

	if !reference {
		*child = updated
		return true, nil
	}

	encoded, err := json.Marshal(updated)
	if err != nil {
		return false, err
	}
	digest, err := manifest.WriteManifestBlob(encoded)
	if err != nil {
		return false, err
	}
	runner := updated.Runner
	if runner == "" {
		runner = child.Runner
	}
	format := updated.Format
	if format == "" {
		format = child.Format
	}
	ref, err := manifest.NewManifestReference(digest, runner, format)
	if err != nil {
		return false, err
	}
	*child = ref
	return true, nil
}

func updateRunnableManifest(doc manifest.Manifest, n int) (manifest.Manifest, bool, error) {
	layers, changed, err := updateParamsLayers(doc.Layers, n)
	if err != nil || !changed {
		return doc, false, err
	}
	doc.Layers = layers
	return doc, true, nil
}

func updateParamsLayers(layers []manifest.Layer, n int) ([]manifest.Layer, bool, error) {
	idx := -1
	for i, layer := range layers {
		if layer.MediaType == manifest.MediaTypeImageParams {
			idx = i
		}
	}

	var raw []byte
	if idx >= 0 {
		data, err := readBlob(layers[idx].Digest)
		if err != nil {
			return nil, false, err
		}
		raw = data
	}

	updated, changed, err := applyDraftNumPredict(raw, n)
	if err != nil || !changed {
		return layers, false, err
	}
	layer, err := manifest.NewLayer(bytes.NewReader(updated), manifest.MediaTypeImageParams)
	if err != nil {
		return nil, false, err
	}
	cloned := append([]manifest.Layer(nil), layers...)
	if idx >= 0 {
		layer.From = layers[idx].From
		layer.Name = layers[idx].Name
		cloned[idx] = layer
		return cloned, true, nil
	}
	return append(cloned, layer), true, nil
}

func readBlob(digest string) ([]byte, error) {
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(path)
}

func readManifestAtDigest(digest string) (manifest.Manifest, error) {
	data, err := readBlob(digest)
	if err != nil {
		return manifest.Manifest{}, err
	}
	sum := sha256.Sum256(data)
	if !manifest.SameDigest(digest, fmt.Sprintf("sha256:%x", sum)) {
		return manifest.Manifest{}, fmt.Errorf("manifest blob %s failed digest check", digest)
	}
	var doc manifest.Manifest
	if err := json.Unmarshal(data, &doc); err != nil {
		return manifest.Manifest{}, err
	}
	return doc, nil
}

// ModelMTPHandler sets or clears a model's draft_num_predict parameter.
// It does not load the model. The app reloads a model that is already running
// so llama-server picks up the new speculative-decoding arguments.
func (s *Server) ModelMTPHandler(c *gin.Context) {
	var req struct {
		Model           string `json:"model"`
		Name            string `json:"name"`
		DraftNumPredict *int   `json:"draft_num_predict"`
	}
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.Model == "" {
		req.Model = req.Name
	}
	if req.Model == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	// A missing field must not be treated as "off". Off is an explicit 0,
	// which removes the parameter.
	if req.DraftNumPredict == nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "draft_num_predict is required"})
		return
	}
	draftNumPredict := *req.DraftNumPredict
	if draftNumPredict != 0 && (draftNumPredict < minDraftNumPredict || draftNumPredict > maxDraftNumPredict) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errDraftNumPredictRange.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, err.Error())
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "cloud models do not support multi-token prediction"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	m, err := GetModel(name.String())
	if errors.Is(err, os.ErrNotExist) {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", req.Model)})
		return
	} else if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if !modelSupportsMTP(m) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model has no multi-token prediction layers"})
		return
	}

	if err := setModelDraftNumPredict(name, draftNumPredict); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", req.Model)})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":            "success",
		"draft_num_predict": draftNumPredict,
	})
}
