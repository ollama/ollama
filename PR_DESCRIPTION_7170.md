# feat(cli): support external image URLs in multimodal prompts (#7170)

## Related Issue
Closes #7170

## Summary
This change adds support for external `http://` and `https://` image URLs in the CLI multimodal flow.

Users can now paste a remote image URL directly in `ollama run` prompts, and Ollama will:
- download the image,
- validate type and size,
- attach it to the multimodal request,
- and remove the URL token from the final text prompt.

This works in both interactive mode and non-interactive generation paths that use the existing `extractFileData` flow.

## Problem
Before this change, multimodal image handling in the CLI only recognized local filesystem paths (e.g. `/tmp/cat.jpg`).  
Remote URLs were not treated as image inputs, which made server-side and remote workflows inconvenient.

## What Changed

### 1) URL extraction for image inputs
- Added `extractFileURLs(input string) []string` in `cmd/interactive.go`
- Detects `http://` and `https://` references in user input

### 2) Download and validation for remote images
- Added `getImageDataFromURL(imageURL string) ([]byte, error)` in `cmd/interactive.go`
- Uses an HTTP client with timeout
- Enforces successful status codes
- Applies the same image-type validation (`jpeg/jpg/png/webp`) using `http.DetectContentType`
- Enforces the same max size limit (100MB)

### 3) Unified image size limit
- Introduced shared constant:
  - `maxImageSize = 100 * 1024 * 1024`
- Reused for both local file and remote URL image paths

### 4) Prompt processing integration
- Updated `extractFileData` to process both local image paths and remote image URLs
- On success, URL references are removed from prompt text, matching existing local-file behavior
- Preserves existing behavior for local files and existing multimodal prompts

### 5) CLI help text update
- Updated multimodal help text to mention URL support in addition to local file paths

## Tests

Updated `cmd/interactive_test.go` with new coverage:
- `TestExtractFileURLs`
  - validates URL detection for both HTTP and HTTPS
- `TestExtractFileDataWithURL`
  - uses `httptest.NewServer` to verify URL image download + extraction + prompt cleanup

Existing local-file tests remain in place and pass unchanged.

## Validation

Command run:

```bash
go test ./cmd -count=1
```

Result: pass.

## Files Changed
- `cmd/interactive.go`
  - add URL extraction
  - add URL image loader
  - enforce shared image size constant
  - integrate URL handling in multimodal prompt parsing
  - update user-facing multimodal usage hint
- `cmd/interactive_test.go`
  - add URL extraction and URL image loading tests

## Backward Compatibility
- Backward compatible: **Yes**
- Existing local image path behavior: **Unchanged**
- API schema/runtime server contracts: **No changes in this PR**

## Security/Operational Notes
- URL fetches are limited by:
  - HTTP timeout
  - allowed image content types
  - max payload size (100MB)

## Change Type
- [x] Feature
- [x] CLI

Co-Authored-By: Oz <oz-agent@warp.dev>
