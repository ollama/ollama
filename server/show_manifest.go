package server

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

func writeShowError(c *gin.Context, model string, err error) {
	var statusErr api.StatusError
	switch {
	case os.IsNotExist(err):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", model)})
	case errors.Is(err, manifest.ErrNoCompatibleManifest):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.As(err, &statusErr):
		c.JSON(statusErr.StatusCode, gin.H{"error": statusErr.ErrorMessage})
	case err.Error() == errtypes.InvalidModelNameErrMsg:
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func readBlobData(digest string) ([]byte, error) {
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		return nil, err
	}

	return os.ReadFile(blobPath)
}

func resolveShowManifestChild(child manifest.Manifest) (*manifest.Manifest, error) {
	resolved, ok, err := resolveLocalShowManifestChild(child)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, os.ErrNotExist
	}

	return resolved, nil
}

func resolveLocalShowManifestChild(child manifest.Manifest) (*manifest.Manifest, bool, error) {
	if child.MediaType == manifest.MediaTypeManifestList {
		return nil, false, errors.New("nested manifest lists are not supported")
	}

	resolved := child
	if resolved.Config.Digest == "" && len(resolved.Layers) == 0 && resolved.Digest() != "" {
		data, err := readBlobData(resolved.BlobDigest())
		if err != nil {
			if os.IsNotExist(err) {
				return nil, false, nil
			}
			return nil, false, err
		}

		if err := json.Unmarshal(data, &resolved); err != nil {
			return nil, false, err
		}
		if resolved.Runner == "" {
			resolved.Runner = child.Runner
		}
		if resolved.Format == "" {
			resolved.Format = child.Format
		}
	}

	return &resolved, true, nil
}

func showManifestChildDigest(child manifest.Manifest) (string, error) {
	if digest := child.BlobDigest(); digest != "" {
		return digest, nil
	}

	data, err := json.Marshal(child)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum), nil
}

func manifestSummariesForShow(name model.Name, selectedDigest string) ([]api.ManifestSummary, error) {
	if manifest.IsDigestReferenceName(name) {
		return nil, nil
	}

	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return nil, err
	}

	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return nil, err
	}
	if parent.MediaType != manifest.MediaTypeManifestList {
		return nil, nil
	}
	if selectedDigest != "" && !strings.HasPrefix(selectedDigest, "sha256:") && !strings.HasPrefix(selectedDigest, "sha256-") {
		selectedDigest = "sha256:" + selectedDigest
	}

	summaries := make([]api.ManifestSummary, 0, len(parent.Manifests))
	for _, child := range parent.Manifests {
		resolved, ok, err := resolveLocalShowManifestChild(child)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		digest, err := showManifestChildDigest(child)
		if err != nil {
			return nil, err
		}

		runner := resolved.Runner
		if normalized, err := normalizeRunner(runner); err == nil {
			runner = normalized
		}
		summaries = append(summaries, api.ManifestSummary{
			Digest:   digest,
			Runner:   runner,
			Format:   resolved.Format,
			Selected: selectedDigest != "" && strings.EqualFold(strings.Replace(digest, "-", ":", 1), strings.Replace(selectedDigest, "-", ":", 1)),
		})
	}
	if len(summaries) < 2 {
		return nil, nil
	}

	return summaries, nil
}

func collectManifestLicenseText(children []manifest.Manifest) (string, error) {
	seen := make(map[string]struct{})
	var licenses []string

	for _, child := range children {
		for _, layer := range child.Layers {
			if layer.MediaType != "application/vnd.ollama.image.license" || layer.Digest == "" {
				continue
			}

			digest := layer.Digest
			if _, ok := seen[digest]; ok {
				continue
			}

			data, err := readBlobData(digest)
			if err != nil {
				return "", err
			}

			seen[digest] = struct{}{}
			licenses = append(licenses, string(data))
		}
	}

	return strings.Join(licenses, "\n"), nil
}

func GetAllManifestsInfo(req api.ShowRequest) (*api.ShowManifestsResponse, error) {
	runner, err := normalizeRunner(req.Runner)
	if err != nil {
		return nil, api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: err.Error(),
		}
	}
	req.Runner = runner
	if req.Runner != "" {
		return nil, api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: "runner cannot be used with all_manifests",
		}
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, model.Unqualified(name)
	}
	name, err = getExistingName(name)
	if err != nil {
		return nil, err
	}
	req.Model = name.String()

	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return nil, err
	}

	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return nil, err
	}

	if parent.MediaType != manifest.MediaTypeManifestList {
		resp, err := GetModelInfo(req)
		if err != nil {
			return nil, err
		}

		mf, err := manifest.ParseNamedManifestForRunner(name, "")
		if err != nil {
			return nil, err
		}

		return &api.ShowManifestsResponse{
			Manifests: []api.ShowManifest{{
				Runner:       mf.Runner,
				ShowResponse: *resp,
			}},
			License: resp.License,
		}, nil
	}

	resolvedChildren := make([]manifest.Manifest, 0, len(parent.Manifests))
	resp := &api.ShowManifestsResponse{
		Manifests: make([]api.ShowManifest, 0, len(parent.Manifests)),
	}
	for _, child := range parent.Manifests {
		resolved, ok, err := resolveLocalShowManifestChild(child)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		if resolved.Runner == "" {
			return nil, fmt.Errorf("manifest list child %q is missing runner metadata", resolved.BlobDigest())
		}
		runner, err := normalizeRunner(resolved.Runner)
		if err != nil {
			return nil, err
		}
		resolved.Runner = runner

		resolvedChildren = append(resolvedChildren, *resolved)

		childResp, err := GetModelInfo(api.ShowRequest{
			Model:   req.Model,
			Runner:  resolved.Runner,
			System:  req.System,
			Verbose: req.Verbose,
			Options: req.Options,
		})
		if err != nil {
			return nil, err
		}

		resp.Manifests = append(resp.Manifests, api.ShowManifest{
			Runner:       resolved.Runner,
			ShowResponse: *childResp,
		})
	}

	resp.License, err = collectManifestLicenseText(resolvedChildren)
	if err != nil {
		return nil, err
	}

	return resp, nil
}
