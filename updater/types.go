package updater

import (
	"io"
	"net/http"
	"time"
)

// CheckOptions configures the update check.
type CheckOptions struct {
	// CurrentVersion is the currently running version (defaults to version.Version).
	CurrentVersion string

	// OS is the target operating system (defaults to runtime.GOOS).
	OS string

	// Arch is the target architecture (defaults to runtime.GOARCH).
	Arch string

	// URL overrides the default release endpoint.
	URL string

	// HTTPClient is the HTTP client to use for requests. If nil, http.DefaultClient is used.
	HTTPClient *http.Client

	// IncludeRC allows checking for release candidate (rc) versions.
	IncludeRC bool
}

// ReleaseAsset represents an artifact attached to a release.
type ReleaseAsset struct {
	Name        string `json:"name"`
	DownloadURL string `json:"download_url"`
	Size        int64  `json:"size"`
}

// ReleaseInfo holds the result of checking for an Ollama update.
type ReleaseInfo struct {
	// CurrentVersion is the normalized current version of Ollama.
	CurrentVersion string `json:"current_version"`

	// LatestVersion is the latest available version (tag name).
	LatestVersion string `json:"latest_version"`

	// UpdateAvailable is true if LatestVersion is strictly newer than CurrentVersion.
	UpdateAvailable bool `json:"update_available"`

	// IsDevVersion is true if the current version appears to be a development build.
	IsDevVersion bool `json:"is_dev_version"`

	// IsRC is true if LatestVersion is a release candidate version.
	IsRC bool `json:"is_rc"`

	// ReleaseURL is the URL to the release page on GitHub.
	ReleaseURL string `json:"release_url"`

	// ReleaseNotes contains the release notes or changelog body.
	ReleaseNotes string `json:"release_notes,omitempty"`

	// PublishedAt is the publication timestamp of the release.
	PublishedAt time.Time `json:"published_at,omitempty"`

	// Asset is the best matching download asset for the host platform.
	Asset *ReleaseAsset `json:"asset,omitempty"`

	// ExtraAssets lists supplementary assets for this platform (e.g. JetPack or ROCm libraries).
	ExtraAssets []ReleaseAsset `json:"extra_assets,omitempty"`

	// AllAssets lists all available assets for this release.
	AllAssets []ReleaseAsset `json:"assets,omitempty"`
}

// PullOptions configures pulling (downloading) a release asset.
type PullOptions struct {
	// Dir is the target directory where the downloaded file will be saved.
	// If empty, the user cache directory (e.g. ~/.cache/ollama/updates) is used.
	Dir string

	// Force will re-download the asset even if it already exists or Ollama is up to date.
	Force bool

	// HTTPClient is the HTTP client to use. If nil, http.DefaultClient is used.
	HTTPClient *http.Client

	// ProgressFn is called periodically during download with bytes downloaded and total bytes.
	ProgressFn func(downloaded, total int64)

	// Writer allows redirecting the downloaded content (useful for testing or piping).
	Writer io.Writer
}

// PullResult describes the outcome of pulling an update.
type PullResult struct {
	// FilePath is the full local path to the downloaded archive or installer.
	FilePath string `json:"file_path"`

	// Filename is the base name of the downloaded file.
	Filename string `json:"filename"`

	// Version is the version that was pulled.
	Version string `json:"version"`

	// Size is the total size in bytes of the downloaded file.
	Size int64 `json:"size"`

	// AlreadyExisted is true if the file was already downloaded with matching size.
	AlreadyExisted bool `json:"already_existed"`
}

// ProgressFunc is a callback for tracking progress.
type ProgressFunc func(current, total int64)
