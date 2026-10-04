package updater

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"runtime"
	"strings"
	"time"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/version"
)

const (
	DefaultGitHubAPIURL      = "https://api.github.com/repos/ollama/ollama/releases/latest"
	DefaultGitHubTagsAPIURL  = "https://api.github.com/repos/ollama/ollama/tags"
	DefaultGitHubReleasesURL = "https://github.com/ollama/ollama/releases/latest"
	DefaultGitHubTagsWebURL  = "https://github.com/ollama/ollama/tags"
	DefaultOllamaUpdateURL   = "https://ollama.com/api/update"
)

var tagRegex = regexp.MustCompile(`/releases/tag/([vV]?[0-9]+\.[0-9]+[a-zA-Z0-9\.\-]*)`)

type gitHubRelease struct {
	TagName     string        `json:"tag_name"`
	Name        string        `json:"name"`
	HTMLURL     string        `json:"html_url"`
	Body        string        `json:"body"`
	PublishedAt time.Time     `json:"published_at"`
	Assets      []gitHubAsset `json:"assets"`
}

type gitHubAsset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
	Size               int64  `json:"size"`
}

// Check queries available release channels to find the latest Ollama release.
func Check(ctx context.Context, opts CheckOptions) (*ReleaseInfo, error) {
	if opts.CurrentVersion == "" {
		opts.CurrentVersion = version.Version
	}
	if opts.OS == "" {
		opts.OS = runtime.GOOS
	}
	if opts.Arch == "" {
		opts.Arch = runtime.GOARCH
	}
	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	endpoint := opts.URL
	if endpoint == "" {
		endpoint = os.Getenv("OLLAMA_UPDATE_URL")
	}

	var rel *gitHubRelease
	var err error

	includeRC := opts.IncludeRC || IsRCVersion(opts.CurrentVersion)

	if endpoint != "" {
		rel, err = fetchFromEndpoint(ctx, client, endpoint, opts)
		if err != nil {
			return nil, err
		}
	} else if includeRC {
		// When RC versions are requested or current version is an RC, check tags first
		tag, tagErr := fetchLatestTagFromGitHub(ctx, client, true)
		if tagErr == nil && tag != "" {
			releaseByTag, relErr := fetchReleaseByTag(ctx, client, tag, opts)
			if relErr == nil && releaseByTag != nil {
				rel = releaseByTag
			} else {
				rel = &gitHubRelease{
					TagName: tag,
					Name:    tag,
					HTMLURL: fmt.Sprintf("https://github.com/ollama/ollama/releases/tag/%s", tag),
				}
			}
		} else {
			slog.Debug("fetching rc tags failed, falling back to releases/latest", "error", tagErr)
			rel, err = fetchFromGitHubAPI(ctx, client, opts)
			if err != nil {
				rel, err = fetchFromGitHubRedirect(ctx, client)
				if err != nil {
					rel, err = fetchFromOllamaAPI(ctx, client, opts)
					if err != nil {
						return nil, fmt.Errorf("failed to check for updates: %w", err)
					}
				}
			}
		}
	} else {
		// Stable releases check: GitHub API -> GitHub Web Redirect -> Ollama API -> Tags fallback
		rel, err = fetchFromGitHubAPI(ctx, client, opts)
		if err != nil {
			slog.Debug("github api update check failed, attempting fallback", "error", err)
			rel, err = fetchFromGitHubRedirect(ctx, client)
			if err != nil {
				slog.Debug("github redirect update check failed, attempting ollama.com fallback", "error", err)
				rel, err = fetchFromOllamaAPI(ctx, client, opts)
				if err != nil {
					slog.Debug("ollama.com fallback failed, attempting tags check", "error", err)
					tag, tagErr := fetchLatestTagFromGitHub(ctx, client, false)
					if tagErr != nil {
						return nil, fmt.Errorf("failed to check for updates: %w", err)
					}
					rel = &gitHubRelease{
						TagName: tag,
						Name:    tag,
						HTMLURL: fmt.Sprintf("https://github.com/ollama/ollama/releases/tag/%s", tag),
					}
				}
			}
		}
	}

	return buildReleaseInfo(opts.CurrentVersion, opts.OS, opts.Arch, rel)
}

func fetchLatestTagFromGitHub(ctx context.Context, client *http.Client, includeRC bool) (string, error) {
	// Attempt 1: GitHub Tags API
	tags, err := fetchTagsFromGitHubAPI(ctx, client)
	if err == nil && len(tags) > 0 {
		if highest := findHighestTag(tags, includeRC); highest != "" {
			return highest, nil
		}
	}

	// Attempt 2: Scrape https://github.com/ollama/ollama/tags directly
	webTags, err := fetchTagsFromGitHubWeb(ctx, client)
	if err == nil && len(webTags) > 0 {
		if highest := findHighestTag(webTags, includeRC); highest != "" {
			return highest, nil
		}
	}

	if err != nil {
		return "", err
	}
	return "", errors.New("no matching tags found")
}

func fetchTagsFromGitHubAPI(ctx context.Context, client *http.Client) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, DefaultGitHubTagsAPIURL+"?per_page=50", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tags api returned status %d", resp.StatusCode)
	}

	var rawTags []struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&rawTags); err != nil {
		return nil, err
	}

	var names []string
	for _, t := range rawTags {
		if t.Name != "" {
			names = append(names, t.Name)
		}
	}
	return names, nil
}

func fetchTagsFromGitHubWeb(ctx context.Context, client *http.Client) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, DefaultGitHubTagsWebURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tags web page returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	matches := tagRegex.FindAllStringSubmatch(string(body), -1)
	seen := make(map[string]bool)
	var names []string
	for _, m := range matches {
		if len(m) > 1 {
			tag := m[1]
			if !seen[tag] {
				seen[tag] = true
				names = append(names, tag)
			}
		}
	}
	return names, nil
}

func findHighestTag(tags []string, includeRC bool) string {
	var highest string
	for _, tag := range tags {
		clean := strings.TrimSpace(tag)
		if !strings.HasPrefix(clean, "v") {
			clean = "v" + clean
		}
		if !semver.IsValid(clean) {
			continue
		}
		if !includeRC && semver.Prerelease(clean) != "" {
			continue
		}
		if highest == "" || semver.Compare(clean, highest) > 0 {
			highest = tag
		}
	}
	return highest
}

func fetchReleaseByTag(ctx context.Context, client *http.Client, tag string, opts CheckOptions) (*gitHubRelease, error) {
	cleanTag := tag
	if !strings.HasPrefix(cleanTag, "v") {
		cleanTag = "v" + cleanTag
	}
	reqURL := fmt.Sprintf("https://api.github.com/repos/ollama/ollama/releases/tags/%s", cleanTag)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s)", opts.CurrentVersion, opts.Arch, opts.OS))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	var release gitHubRelease
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return nil, err
	}
	return &release, nil
}

func fetchFromGitHubAPI(ctx context.Context, client *http.Client, opts CheckOptions) (*gitHubRelease, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, DefaultGitHubAPIURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s)", opts.CurrentVersion, opts.Arch, opts.OS))
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusTooManyRequests {
		return nil, fmt.Errorf("github api rate limit exceeded (status %d)", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d from github api", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var release gitHubRelease
	if err := json.Unmarshal(body, &release); err != nil {
		return nil, fmt.Errorf("malformed release response: %w", err)
	}
	return &release, nil
}

func fetchFromGitHubRedirect(ctx context.Context, client *http.Client) (*gitHubRelease, error) {
	redirectClient := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	if client.Transport != nil {
		redirectClient.Transport = client.Transport
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodHead, DefaultGitHubReleasesURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := redirectClient.Do(req)
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	if resp.StatusCode != http.StatusFound && resp.StatusCode != http.StatusMovedPermanently && resp.StatusCode != http.StatusTemporaryRedirect {
		return nil, fmt.Errorf("expected redirect from %s, got status %d", DefaultGitHubReleasesURL, resp.StatusCode)
	}

	location := resp.Header.Get("Location")
	if location == "" {
		return nil, fmt.Errorf("missing Location header in redirect")
	}

	u, err := url.Parse(location)
	if err != nil {
		return nil, fmt.Errorf("failed to parse redirect location %q: %w", location, err)
	}

	tag := path.Base(u.Path)
	if tag == "" || tag == "." || tag == "/" {
		return nil, fmt.Errorf("could not determine release tag from location %q", location)
	}

	return &gitHubRelease{
		TagName: tag,
		Name:    tag,
		HTMLURL: location,
	}, nil
}

func fetchFromOllamaAPI(ctx context.Context, client *http.Client, opts CheckOptions) (*gitHubRelease, error) {
	u, err := url.Parse(DefaultOllamaUpdateURL)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("os", opts.OS)
	q.Set("arch", opts.Arch)
	q.Set("version", opts.CurrentVersion)
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s)", opts.CurrentVersion, opts.Arch, opts.OS))

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return &gitHubRelease{
			TagName: opts.CurrentVersion,
			Name:    opts.CurrentVersion,
		}, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama update API returned status %d", resp.StatusCode)
	}

	var updatePayload struct {
		URL     string `json:"url"`
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&updatePayload); err != nil {
		return nil, err
	}

	tag := updatePayload.Version
	if tag == "" && updatePayload.URL != "" {
		tag = path.Base(path.Dir(updatePayload.URL))
	}
	if tag == "" {
		return nil, fmt.Errorf("could not determine version from ollama update response")
	}

	assetName := path.Base(updatePayload.URL)
	return &gitHubRelease{
		TagName: tag,
		Name:    tag,
		HTMLURL: fmt.Sprintf("https://github.com/ollama/ollama/releases/tag/%s", tag),
		Assets: []gitHubAsset{
			{
				Name:               assetName,
				BrowserDownloadURL: updatePayload.URL,
			},
		},
	}, nil
}

func fetchFromEndpoint(ctx context.Context, client *http.Client, endpoint string, opts CheckOptions) (*gitHubRelease, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s)", opts.CurrentVersion, opts.Arch, opts.OS))
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("custom update endpoint returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var release gitHubRelease
	if err := json.Unmarshal(body, &release); err == nil && release.TagName != "" {
		return &release, nil
	}

	// Try Ollama update payload format
	var updatePayload struct {
		URL     string `json:"url"`
		Version string `json:"version"`
	}
	if err := json.Unmarshal(body, &updatePayload); err == nil && (updatePayload.Version != "" || updatePayload.URL != "") {
		tag := updatePayload.Version
		if tag == "" {
			tag = path.Base(path.Dir(updatePayload.URL))
		}
		return &gitHubRelease{
			TagName: tag,
			Name:    tag,
			Assets: []gitHubAsset{
				{
					Name:               path.Base(updatePayload.URL),
					BrowserDownloadURL: updatePayload.URL,
				},
			},
		}, nil
	}

	return nil, fmt.Errorf("unrecognized response format from update endpoint")
}

func buildReleaseInfo(currentVersion, goos, goarch string, rel *gitHubRelease) (*ReleaseInfo, error) {
	info := &ReleaseInfo{
		CurrentVersion: currentVersion,
		LatestVersion:  rel.TagName,
		ReleaseURL:     rel.HTMLURL,
		ReleaseNotes:   rel.Body,
		PublishedAt:    rel.PublishedAt,
		IsDevVersion:   isDevVersion(currentVersion),
		IsRC:           IsRCVersion(rel.TagName),
	}

	if info.ReleaseURL == "" && rel.TagName != "" {
		info.ReleaseURL = fmt.Sprintf("https://github.com/ollama/ollama/releases/tag/%s", rel.TagName)
	}

	for _, a := range rel.Assets {
		info.AllAssets = append(info.AllAssets, ReleaseAsset{
			Name:        a.Name,
			DownloadURL: a.BrowserDownloadURL,
			Size:        a.Size,
		})
	}

	// Determine matching asset
	info.Asset = SelectAsset(info.AllAssets, goos, goarch)
	if info.Asset == nil && rel.TagName != "" {
		info.Asset = DefaultAssetForTag(rel.TagName, goos, goarch)
	}
	info.ExtraAssets = SelectExtraAssets(info.AllAssets, goos, goarch)

	// Version comparison
	cmp, err := CompareVersions(currentVersion, rel.TagName)
	if err != nil {
		info.UpdateAvailable = currentVersion != rel.TagName
	} else {
		info.UpdateAvailable = cmp < 0
	}

	return info, nil
}

// IsRCVersion returns true if the version string is a release candidate (contains "-rc").
func IsRCVersion(v string) bool {
	clean := strings.TrimSpace(v)
	if clean == "" {
		return false
	}
	if !strings.HasPrefix(clean, "v") {
		clean = "v" + clean
	}
	pre := semver.Prerelease(clean)
	return strings.Contains(strings.ToLower(pre), "rc")
}

// CompareVersions compares two version strings.
// Returns -1 if current < latest, 0 if current == latest, and 1 if current > latest.
func CompareVersions(current, latest string) (int, error) {
	cleanCurrent := strings.TrimSpace(current)
	cleanLatest := strings.TrimSpace(latest)

	if isDevVersion(cleanCurrent) {
		return -1, nil
	}

	vCurrent := cleanCurrent
	if !strings.HasPrefix(vCurrent, "v") {
		vCurrent = "v" + vCurrent
	}
	vLatest := cleanLatest
	if !strings.HasPrefix(vLatest, "v") {
		vLatest = "v" + vLatest
	}

	if !semver.IsValid(vCurrent) {
		return 0, fmt.Errorf("invalid current version: %q", current)
	}
	if !semver.IsValid(vLatest) {
		return 0, fmt.Errorf("invalid latest version: %q", latest)
	}

	return semver.Compare(vCurrent, vLatest), nil
}

func isDevVersion(v string) bool {
	v = strings.TrimSpace(v)
	if v == "" || v == "0.0.0" || v == "v0.0.0" {
		return true
	}
	lower := strings.ToLower(v)
	return strings.Contains(lower, "dev") || strings.Contains(lower, "dirty")
}

// SelectAsset selects the primary release asset (containing the ollama executable) for the given OS and architecture.
func SelectAsset(assets []ReleaseAsset, goos, goarch string) *ReleaseAsset {
	if len(assets) == 0 {
		return nil
	}

	find := func(name string) *ReleaseAsset {
		for _, a := range assets {
			if strings.EqualFold(a.Name, name) {
				return &a
			}
		}
		return nil
	}

	switch goos {
	case "linux":
		switch goarch {
		case "amd64":
			if a := find("ollama-linux-amd64.tar.zst"); a != nil {
				return a
			}
			if a := find("ollama-linux-amd64.tgz"); a != nil {
				return a
			}
		case "arm64":
			if a := find("ollama-linux-arm64.tar.zst"); a != nil {
				return a
			}
			if a := find("ollama-linux-arm64.tgz"); a != nil {
				return a
			}
		}
	case "darwin":
		if a := find("Ollama-darwin.zip"); a != nil {
			return a
		}
		if a := find("ollama-darwin.tgz"); a != nil {
			return a
		}
		if a := find("Ollama.dmg"); a != nil {
			return a
		}
	case "windows":
		if goarch == "arm64" {
			if a := find("ollama-windows-arm64.zip"); a != nil {
				return a
			}
			if a := find("OllamaSetup.exe"); a != nil {
				return a
			}
		} else {
			if a := find("OllamaSetup.exe"); a != nil {
				return a
			}
			if a := find("ollama-windows-amd64.zip"); a != nil {
				return a
			}
		}
	}
	return nil
}

// SelectExtraAssets selects supplementary add-on packages (e.g. JetPack GPU libraries) for the platform.
func SelectExtraAssets(assets []ReleaseAsset, goos, goarch string) []ReleaseAsset {
	if len(assets) == 0 {
		return nil
	}

	find := func(name string) *ReleaseAsset {
		for _, a := range assets {
			if strings.EqualFold(a.Name, name) {
				return &a
			}
		}
		return nil
	}

	var extras []ReleaseAsset
	if goos == "linux" && goarch == "arm64" {
		if isJetPack6() {
			if a := find("ollama-linux-arm64-jetpack6.tar.zst"); a != nil {
				extras = append(extras, *a)
			}
		} else if isJetPack5() {
			if a := find("ollama-linux-arm64-jetpack5.tar.zst"); a != nil {
				extras = append(extras, *a)
			}
		}
	}
	return extras
}

// DefaultAssetForTag synthesizes the standard download asset information for a given release tag.
func DefaultAssetForTag(tag, goos, goarch string) *ReleaseAsset {
	cleanTag := tag
	if !strings.HasPrefix(cleanTag, "v") {
		cleanTag = "v" + cleanTag
	}
	baseURL := fmt.Sprintf("https://github.com/ollama/ollama/releases/download/%s", cleanTag)

	var filename string
	switch goos {
	case "linux":
		filename = fmt.Sprintf("ollama-linux-%s.tar.zst", goarch)
	case "darwin":
		filename = "Ollama-darwin.zip"
	case "windows":
		filename = "OllamaSetup.exe"
	default:
		filename = fmt.Sprintf("ollama-%s-%s.tar.zst", goos, goarch)
	}

	return &ReleaseAsset{
		Name:        filename,
		DownloadURL: fmt.Sprintf("%s/%s", baseURL, filename),
	}
}

func isJetPack6() bool {
	data, err := os.ReadFile("/etc/nv_tegra_release")
	if err != nil {
		return false
	}
	return strings.Contains(string(data), "R36")
}

func isJetPack5() bool {
	data, err := os.ReadFile("/etc/nv_tegra_release")
	if err != nil {
		return false
	}
	return strings.Contains(string(data), "R35")
}
