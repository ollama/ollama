# feat: support external image URLs for multimodal inputs (#7170)
## Related Issue
Closes #7170
## Summary
This PR adds first-class support for external image URLs in multimodal workflows, including CLI prompt parsing and server-side request handling.
Users can now provide `http://` / `https://` image URLs, and Ollama will download, validate, and attach those images to multimodal requests.
## Problem
Previously, image inputs were effectively limited to local files (CLI) or pre-decoded bytes (`images` field in API payloads).
For remote/server deployments, this made common workflows cumbersome because users had to manually download and encode images before inference.
## What Changed
### CLI
- `cmd/interactive.go`
  - Added URL detection for image links in prompt text.
  - Added URL image loading with:
    - request timeout,
    - status validation,
    - content-type validation,
    - max-size protection.
  - Kept existing local file behavior.
  - Updated multimodal usage hint to include URL usage.
- `cmd/interactive_test.go`
  - Added URL extraction and URL image loading tests.
### API types
- `api/types.go`
  - Added `ImageURL` type:
    - `url`
    - `allow_http`
  - Added `image_urls` to:
    - `GenerateRequest`
    - `Message`
### Server
- `server/image_downloader.go` (new)
  - Added centralized image downloader with:
    - scheme validation (`https` by default, optional `http` via `allow_http`),
    - host allow-list enforcement,
    - download timeout,
    - max-size enforcement,
    - cache support.
- `server/routes.go`
  - Restored/fixed `GenerateHandler` declaration.
  - Added generate request `image_urls` processing (`processImageURLs`).
  - Added chat message `image_urls` processing (`processMessageImageURLs`).
  - Converts downloaded images into existing `images` flow before inference.
### Env config
- `envconfig/config.go`
  - Added settings for image URL handling:
    - `OLLAMA_IMAGE_URL_ENABLED`
    - `OLLAMA_IMAGE_URL_MAX_SIZE`
    - `OLLAMA_IMAGE_URL_TIMEOUT`
    - `OLLAMA_IMAGE_URL_ALLOWED_HOSTS`
    - `OLLAMA_IMAGE_URL_CACHE_DIR`
## Security and Operational Controls
- URL scheme restrictions (`https` default, optional `http` only when explicitly allowed).
- Host allow-list support.
- Download timeout.
- Maximum image size enforcement.
- Content type checks for image data.
## Backward Compatibility
- Backward compatible: **Yes**
- Existing `images` behavior: **Unchanged**
- Existing local-file CLI behavior: **Unchanged**
- New fields/config are optional.
## Validation
```bash
go test ./cmd -count=1
go test ./server -run TestImageDownloader -count=1
go test ./server -run TestDummyDoesNotExist -count=1
```
## Change Type
- [x] Feature
- [x] CLI
- [x] API / server
