package skills

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/pelletier/go-toml/v2"
)

const (
	envSkillsDir      = "OLLAMA_SKILLS"
	envSkillAllow     = "OLLAMA_SKILL_ALLOW"
	skillManifestFile = "skill.toml"
	installedDirName  = "installed"
	enabledStateFile  = "enabled.json"
	grantsStateFile   = "grants.json"
	backupDirName     = "backups"
	logDirName        = "logs"
	metadataFile      = ".ollama-skill.json"
	policyFile        = "policy.json"
	auditFile         = "audit.log"
)

var (
	// ErrSkillNotFound indicates a requested skill has not been installed.
	ErrSkillNotFound = errors.New("skill not found")
	// ErrSkillNotEnabled indicates a requested skill is installed but disabled.
	ErrSkillNotEnabled = errors.New("skill not enabled")
	// ErrPermissionDenied indicates required permissions were not granted.
	ErrPermissionDenied = errors.New("permission denied")
	// ErrSandboxDenied indicates skill sandbox policy blocked execution.
	ErrSandboxDenied = errors.New("sandbox denied")
	// ErrProvenanceVerification indicates declared provenance could not be verified.
	ErrProvenanceVerification = errors.New("provenance verification failed")
	// ErrPolicyDenied indicates global policy rejected a skill operation.
	ErrPolicyDenied = errors.New("policy denied")

	validSkillName  = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	validPermission = regexp.MustCompile(`^[a-z][a-z0-9_.-]*$`)
	validFieldType  = regexp.MustCompile(`^[a-z][a-z0-9_.-]*$`)

	auditMaxLogBytes   int64 = 5 * 1024 * 1024
	auditMaxLogBackups       = 5
)

// Spec describes the skill.toml MVP schema.
type Spec struct {
	Name        string         `toml:"name"`
	Description string         `toml:"description,omitempty"`
	Version     string         `toml:"version,omitempty"`
	Tags        []string       `toml:"tags,omitempty"`
	Command     string         `toml:"command"`
	Args        []string       `toml:"args,omitempty"`
	IO          IOSpec         `toml:"io"`
	Permissions PermissionSpec `toml:"permissions"`
	Examples    []ExampleSpec  `toml:"examples,omitempty"`
	Sandbox     SandboxSpec    `toml:"sandbox,omitempty"`
	Provenance  ProvenanceSpec `toml:"provenance,omitempty"`
}

type IOSpec struct {
	Inputs       []string    `toml:"inputs,omitempty"`
	Outputs      []string    `toml:"outputs,omitempty"`
	InputSchema  []FieldSpec `toml:"input_schema,omitempty"`
	OutputSchema []FieldSpec `toml:"output_schema,omitempty"`
}

type FieldSpec struct {
	Name        string `toml:"name"`
	Type        string `toml:"type"`
	Description string `toml:"description,omitempty"`
	Required    bool   `toml:"required,omitempty"`
}

type PermissionSpec struct {
	Required []string `toml:"required,omitempty"`
}

type ExampleSpec struct {
	Prompt string   `toml:"prompt"`
	Args   []string `toml:"args,omitempty"`
	Output string   `toml:"output,omitempty"`
}

type SandboxSpec struct {
	TimeoutSeconds  int      `toml:"timeout_seconds,omitempty"`
	MaxOutputBytes  int64    `toml:"max_output_bytes,omitempty"`
	AllowNetwork    *bool    `toml:"allow_network,omitempty"`
	AllowFilesystem *bool    `toml:"allow_filesystem,omitempty"`
	AllowedPaths    []string `toml:"allowed_paths,omitempty"`
}

type ProvenanceSpec struct {
	SHA256    string `toml:"sha256,omitempty"`
	Signature string `toml:"signature,omitempty"`
	PublicKey string `toml:"public_key,omitempty"`
}

type Skill struct {
	Spec               Spec
	Dir                string
	Enabled            bool
	GrantedPermissions []string
	Metadata           Metadata
}

type Metadata struct {
	Source      string `json:"source,omitempty"`
	Repository  string `json:"repository,omitempty"`
	Ref         string `json:"ref,omitempty"`
	Commit      string `json:"commit,omitempty"`
	InstalledAt string `json:"installed_at,omitempty"`
	UpdatedAt   string `json:"updated_at,omitempty"`
	Digest      string `json:"digest,omitempty"`
	VerifiedAt  string `json:"verified_at,omitempty"`
	Verified    bool   `json:"verified,omitempty"`
	Signed      bool   `json:"signed,omitempty"`
	PublicKey   string `json:"public_key,omitempty"`
}

type Policy struct {
	RequireSHA256      bool     `json:"require_sha256,omitempty"`
	RequireSignature   bool     `json:"require_signature,omitempty"`
	TrustedPublicKeys  []string `json:"trusted_public_keys,omitempty"`
	AllowedSources     []string `json:"allowed_sources,omitempty"`
	AllowedPermissions []string `json:"allowed_permissions,omitempty"`
	DeniedPermissions  []string `json:"denied_permissions,omitempty"`
	AllowNetwork       *bool    `json:"allow_network,omitempty"`
	AllowFilesystem    *bool    `json:"allow_filesystem,omitempty"`
	MaxTimeoutSeconds  int      `json:"max_timeout_seconds,omitempty"`
	MaxOutputBytes     int64    `json:"max_output_bytes,omitempty"`
}

type VerifyResult struct {
	Skill      string
	Digest     string
	Verified   bool
	Signed     bool
	PublicKey  string
	Provenance ProvenanceSpec
}

// RunOptions controls how a skill is executed.
type RunOptions struct {
	// GrantedPermissions are permissions allowed for this execution.
	// If empty, persisted grants and OLLAMA_SKILL_ALLOW are used.
	GrantedPermissions []string
	// DryRun prints the execution plan without running the command.
	DryRun bool
	// Timeout overrides the skill timeout.
	Timeout time.Duration
	// MaxOutputBytes overrides output size guard.
	MaxOutputBytes int64
}

type sourceSpec struct {
	raw        string
	localPath  string
	repository string
	ref        string
	isGit      bool
}

type PermissionError struct {
	Skill   string
	Missing []string
}

type ProvenanceError struct {
	Skill  string
	Reason string
}

func (e *ProvenanceError) Error() string {
	return fmt.Sprintf("%v: %s", ErrProvenanceVerification, e.Reason)
}

func (e *ProvenanceError) Unwrap() error { return ErrProvenanceVerification }

type PolicyError struct {
	Skill  string
	Reason string
}

func (e *PolicyError) Error() string {
	return fmt.Sprintf("%v: %s", ErrPolicyDenied, e.Reason)
}

func (e *PolicyError) Unwrap() error { return ErrPolicyDenied }

func (e *PermissionError) Error() string {
	return fmt.Sprintf("%v: missing required permissions for %s: %s", ErrPermissionDenied, e.Skill, strings.Join(e.Missing, ", "))
}

func (e *PermissionError) Unwrap() error { return ErrPermissionDenied }

func MissingPermissions(err error) []string {
	var pErr *PermissionError
	if errors.As(err, &pErr) {
		return slices.Clone(pErr.Missing)
	}
	return nil
}

type SandboxError struct {
	Skill  string
	Reason string
}

func (e *SandboxError) Error() string {
	return fmt.Sprintf("%v: %s", ErrSandboxDenied, e.Reason)
}

func (e *SandboxError) Unwrap() error { return ErrSandboxDenied }

var (
	sessionGrantMu sync.Mutex
	sessionGrants  = map[string][]string{}
)

func sessionGrantKey(skillName string) string {
	root, err := RootDir()
	if err != nil {
		return skillName
	}
	return root + "|" + skillName
}

func RootDir() (string, error) {
	if v := strings.TrimSpace(os.Getenv(envSkillsDir)); v != "" {
		return filepath.Clean(v), nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", "skills"), nil
}

func installedDir() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, installedDirName), nil
}

func enabledPath() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, enabledStateFile), nil
}

func grantsPath() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, grantsStateFile), nil
}

func policyPath() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, policyFile), nil
}

func auditPath() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, auditFile), nil
}

func backupRootDir() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, backupDirName), nil
}

func logRootDir() (string, error) {
	root, err := RootDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(root, logDirName), nil
}

func ensureLayout() error {
	root, err := RootDir()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}

	installed, err := installedDir()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(installed, 0o755); err != nil {
		return err
	}

	backups, err := backupRootDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(backups, 0o755); err != nil {
		return err
	}

	logs, err := logRootDir()
	if err != nil {
		return err
	}
	return os.MkdirAll(logs, 0o755)
}

func (s *Spec) normalize(fallbackName string) error {
	s.Name = strings.TrimSpace(s.Name)
	if s.Name == "" {
		s.Name = strings.TrimSpace(fallbackName)
	}
	if s.Name == "" {
		return errors.New("skill name is required")
	}
	if !validSkillName.MatchString(s.Name) {
		return fmt.Errorf("invalid skill name %q: only letters, numbers, '.', '_' and '-' are allowed", s.Name)
	}

	s.Description = strings.TrimSpace(s.Description)
	s.Version = strings.TrimSpace(s.Version)
	s.Tags = normalizeStringList(s.Tags)
	s.Command = strings.TrimSpace(s.Command)
	if s.Command == "" {
		return errors.New("skill command is required")
	}

	s.IO.Inputs = normalizeStringList(s.IO.Inputs)
	s.IO.Outputs = normalizeStringList(s.IO.Outputs)
	if err := normalizeFields(s.IO.InputSchema); err != nil {
		return err
	}
	if err := normalizeFields(s.IO.OutputSchema); err != nil {
		return err
	}
	if len(s.IO.Inputs) == 0 && len(s.IO.InputSchema) > 0 {
		inputs := make([]string, 0, len(s.IO.InputSchema))
		for _, field := range s.IO.InputSchema {
			inputs = append(inputs, field.Name+":"+field.Type)
		}
		s.IO.Inputs = inputs
	}
	if len(s.IO.Outputs) == 0 && len(s.IO.OutputSchema) > 0 {
		outputs := make([]string, 0, len(s.IO.OutputSchema))
		for _, field := range s.IO.OutputSchema {
			outputs = append(outputs, field.Name+":"+field.Type)
		}
		s.IO.Outputs = outputs
	}

	requiredPerms, err := normalizePermissions(s.Permissions.Required)
	if err != nil {
		return err
	}
	s.Permissions.Required = requiredPerms

	for i := range s.Examples {
		s.Examples[i].Prompt = strings.TrimSpace(s.Examples[i].Prompt)
		s.Examples[i].Output = strings.TrimSpace(s.Examples[i].Output)
		s.Examples[i].Args = normalizeStringList(s.Examples[i].Args)
	}

	if s.Sandbox.TimeoutSeconds < 0 {
		return errors.New("sandbox timeout_seconds cannot be negative")
	}
	if s.Sandbox.TimeoutSeconds == 0 {
		s.Sandbox.TimeoutSeconds = 60
	}
	if s.Sandbox.MaxOutputBytes < 0 {
		return errors.New("sandbox max_output_bytes cannot be negative")
	}
	if s.Sandbox.MaxOutputBytes == 0 {
		s.Sandbox.MaxOutputBytes = 2 * 1024 * 1024
	}
	if s.Sandbox.AllowNetwork == nil {
		v := true
		s.Sandbox.AllowNetwork = &v
	}
	if s.Sandbox.AllowFilesystem == nil {
		v := true
		s.Sandbox.AllowFilesystem = &v
	}
	if len(s.Sandbox.AllowedPaths) > 0 {
		for i, value := range s.Sandbox.AllowedPaths {
			p := strings.TrimSpace(value)
			if p == "" {
				continue
			}
			s.Sandbox.AllowedPaths[i] = filepath.Clean(p)
		}
		s.Sandbox.AllowedPaths = normalizeStringList(s.Sandbox.AllowedPaths)
	}

	s.Provenance.SHA256 = strings.ToLower(strings.TrimSpace(s.Provenance.SHA256))
	s.Provenance.Signature = strings.TrimSpace(s.Provenance.Signature)
	s.Provenance.PublicKey = strings.TrimSpace(s.Provenance.PublicKey)
	if s.Provenance.SHA256 != "" {
		if _, err := hex.DecodeString(s.Provenance.SHA256); err != nil || len(s.Provenance.SHA256) != 64 {
			return fmt.Errorf("invalid provenance sha256 %q", s.Provenance.SHA256)
		}
	}
	if (s.Provenance.Signature == "") != (s.Provenance.PublicKey == "") {
		return errors.New("provenance signature and public_key must be provided together")
	}

	return nil
}

func LoadSpec(path string) (Spec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Spec{}, err
	}

	var spec Spec
	if err := toml.Unmarshal(data, &spec); err != nil {
		return Spec{}, fmt.Errorf("parse %s: %w", path, err)
	}

	if err := spec.normalize(filepath.Base(filepath.Dir(path))); err != nil {
		return Spec{}, err
	}
	if len(spec.Sandbox.AllowedPaths) > 0 {
		baseDir := filepath.Dir(path)
		for i, value := range spec.Sandbox.AllowedPaths {
			p := strings.TrimSpace(value)
			if p == "" {
				continue
			}
			if !filepath.IsAbs(p) {
				p = filepath.Join(baseDir, p)
			}
			abs, err := filepath.Abs(p)
			if err != nil {
				return Spec{}, err
			}
			spec.Sandbox.AllowedPaths[i] = filepath.Clean(abs)
		}
		spec.Sandbox.AllowedPaths = normalizeStringList(spec.Sandbox.AllowedPaths)
	}

	return spec, nil
}

func Install(source string) (Skill, error) {
	return installWithAction(source, "install")
}

func installWithAction(source, action string) (installed Skill, err error) {
	if action == "" {
		action = "install"
	}

	src, err := resolveInstallSource(source)
	if err != nil {
		return Skill{}, err
	}

	metadata := Metadata{
		Source:      src.raw,
		InstalledAt: time.Now().UTC().Format(time.RFC3339),
	}
	defer func() {
		_ = appendAuditEvent(auditEvent{
			Time:    time.Now().UTC().Format(time.RFC3339),
			Action:  action,
			Skill:   installed.Spec.Name,
			Source:  metadata.Source,
			Success: err == nil,
			Details: map[string]any{
				"repository": metadata.Repository,
				"ref":        metadata.Ref,
				"commit":     metadata.Commit,
				"verified":   metadata.Verified,
				"signed":     metadata.Signed,
				"digest":     metadata.Digest,
			},
		})
	}()

	workDir := src.localPath
	if src.isGit {
		clonedDir, commit, err := checkoutGitSource(src.repository, src.ref)
		if err != nil {
			return Skill{}, err
		}
		defer os.RemoveAll(clonedDir)
		workDir = clonedDir
		metadata.Repository = src.repository
		metadata.Ref = src.ref
		metadata.Commit = commit
	}

	specPath := filepath.Join(workDir, skillManifestFile)
	spec, err := LoadSpec(specPath)
	if err != nil {
		return Skill{}, err
	}

	if err := ensureLayout(); err != nil {
		return Skill{}, err
	}

	root, err := installedDir()
	if err != nil {
		return Skill{}, err
	}

	tmpDir, err := os.MkdirTemp(root, spec.Name+"-tmp-")
	if err != nil {
		return Skill{}, err
	}
	defer os.RemoveAll(tmpDir)

	if err := copyDir(workDir, tmpDir); err != nil {
		return Skill{}, err
	}

	copiedSpecPath := filepath.Join(tmpDir, skillManifestFile)
	copiedSpec, err := LoadSpec(copiedSpecPath)
	if err != nil {
		return Skill{}, err
	}

	policy, err := loadPolicy()
	if err != nil {
		return Skill{}, err
	}
	if err := enforceInstallPolicy(policy, copiedSpec, metadata.Source); err != nil {
		return Skill{}, err
	}

	verifyResult, err := verifyProvenanceForDir(tmpDir, copiedSpec, policy)
	if err != nil {
		return Skill{}, err
	}
	metadata.Digest = verifyResult.Digest
	metadata.Verified = verifyResult.Verified
	metadata.Signed = verifyResult.Signed
	metadata.PublicKey = verifyResult.PublicKey
	if verifyResult.Verified {
		metadata.VerifiedAt = time.Now().UTC().Format(time.RFC3339)
	}

	metadata.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	if err := writeMetadata(tmpDir, metadata); err != nil {
		return Skill{}, err
	}

	dest := filepath.Join(root, copiedSpec.Name)
	backup := ""
	if _, err := os.Stat(dest); err == nil {
		backup, err = moveToBackup(copiedSpec.Name, dest, action)
		if err != nil {
			return Skill{}, fmt.Errorf("prepare install swap: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return Skill{}, err
	}

	if err := os.Rename(tmpDir, dest); err != nil {
		if backup != "" {
			_ = os.Rename(backup, dest)
		}
		return Skill{}, fmt.Errorf("finalize install swap: %w", err)
	}

	if err := pruneBackups(copiedSpec.Name, 10); err != nil {
		return Skill{}, err
	}

	state, err := readEnabledState()
	if err != nil {
		return Skill{}, err
	}
	grants, err := readGrantedState()
	if err != nil {
		return Skill{}, err
	}
	skillMeta, err := readMetadata(dest)
	if err != nil {
		return Skill{}, err
	}

	installed = Skill{
		Spec:               copiedSpec,
		Dir:                dest,
		Enabled:            state[copiedSpec.Name],
		GrantedPermissions: grants[copiedSpec.Name],
		Metadata:           skillMeta,
	}
	return installed, nil
}

func List() ([]Skill, error) {
	if err := ensureLayout(); err != nil {
		return nil, err
	}

	root, err := installedDir()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	state, err := readEnabledState()
	if err != nil {
		return nil, err
	}
	grants, err := readGrantedState()
	if err != nil {
		return nil, err
	}
	session := snapshotSessionGrants()

	out := make([]Skill, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		dir := filepath.Join(root, entry.Name())
		spec, err := LoadSpec(filepath.Join(dir, skillManifestFile))
		if err != nil {
			return nil, err
		}
		metadata, err := readMetadata(dir)
		if err != nil {
			return nil, err
		}

		allGrants, err := normalizeGrantedPermissions(append(grants[spec.Name], session[spec.Name]...))
		if err != nil {
			return nil, err
		}
		if slices.Contains(allGrants, "*") {
			allGrants = []string{"*"}
		}
		sort.Strings(allGrants)

		out = append(out, Skill{
			Spec:               spec,
			Dir:                dir,
			Enabled:            state[spec.Name],
			GrantedPermissions: allGrants,
			Metadata:           metadata,
		})
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Spec.Name < out[j].Spec.Name
	})

	return out, nil
}

func Enabled() ([]Skill, error) {
	all, err := List()
	if err != nil {
		return nil, err
	}

	enabled := all[:0]
	for _, skill := range all {
		if skill.Enabled {
			enabled = append(enabled, skill)
		}
	}

	return slices.Clip(enabled), nil
}

func Search(query string) ([]Skill, error) {
	all, err := List()
	if err != nil {
		return nil, err
	}

	trimmed := strings.TrimSpace(strings.ToLower(query))
	if trimmed == "" {
		return all, nil
	}

	terms := strings.Fields(trimmed)
	filtered := make([]Skill, 0, len(all))
	for _, skill := range all {
		index := strings.ToLower(strings.Join([]string{
			skill.Spec.Name,
			skill.Spec.Description,
			strings.Join(skill.Spec.Tags, " "),
			strings.Join(skill.Spec.IO.Inputs, " "),
			strings.Join(skill.Spec.IO.Outputs, " "),
			strings.Join(skill.Spec.Permissions.Required, " "),
		}, " "))

		match := true
		for _, term := range terms {
			if !strings.Contains(index, term) {
				match = false
				break
			}
		}
		if match {
			filtered = append(filtered, skill)
		}
	}

	return filtered, nil
}

func Update(name, source, toRef string) (Skill, error) {
	skill, err := Get(name)
	if err != nil {
		return Skill{}, err
	}

	resolvedSource := strings.TrimSpace(source)
	if resolvedSource == "" {
		resolvedSource = strings.TrimSpace(skill.Metadata.Source)
	}
	if resolvedSource == "" {
		return Skill{}, fmt.Errorf("no source is recorded for skill %q; provide --source", skill.Spec.Name)
	}

	if ref := strings.TrimSpace(toRef); ref != "" {
		base, _ := splitPinnedRef(resolvedSource)
		if !isGitLikeSource(base) && !isGitHubShorthand(base) {
			return Skill{}, errors.New("--to can only be used with git or GitHub sources")
		}
		resolvedSource = base + "@" + ref
	}

	return installWithAction(resolvedSource, "update")
}

func Uninstall(name string) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	if _, err := moveToBackup(skill.Spec.Name, skill.Dir, "uninstall"); err != nil {
		return err
	}

	state, err := readEnabledState()
	if err != nil {
		return err
	}
	delete(state, skill.Spec.Name)
	if err := writeEnabledState(state); err != nil {
		return err
	}

	grants, err := readGrantedState()
	if err != nil {
		return err
	}
	delete(grants, skill.Spec.Name)
	if err := writeGrantedState(grants); err != nil {
		return err
	}

	_ = appendAuditEvent(auditEvent{
		Time:    time.Now().UTC().Format(time.RFC3339),
		Action:  "uninstall",
		Skill:   skill.Spec.Name,
		Source:  skill.Metadata.Source,
		Success: true,
	})
	return nil
}

func Rollback(name string) (Skill, error) {
	n := strings.TrimSpace(name)
	if n == "" {
		return Skill{}, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}

	if skill, err := Get(n); err == nil {
		n = skill.Spec.Name
	} else if !errors.Is(err, ErrSkillNotFound) {
		return Skill{}, err
	}

	backups, err := listBackups(n)
	if err != nil {
		return Skill{}, err
	}
	if len(backups) == 0 {
		return Skill{}, fmt.Errorf("no backups found for skill %q", n)
	}

	root, err := installedDir()
	if err != nil {
		return Skill{}, err
	}
	dest := filepath.Join(root, n)

	rollbackBackup := ""
	if _, err := os.Stat(dest); err == nil {
		rollbackBackup, err = moveToBackup(n, dest, "rollback-current")
		if err != nil {
			return Skill{}, err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return Skill{}, err
	}

	target := backups[0]
	if err := os.Rename(target, dest); err != nil {
		if rollbackBackup != "" {
			_ = os.Rename(rollbackBackup, dest)
		}
		return Skill{}, err
	}

	return Get(n)
}

func BackupCount(name string) (int, error) {
	skill, err := Get(name)
	if err != nil {
		return 0, err
	}

	backups, err := listBackups(skill.Spec.Name)
	if err != nil {
		return 0, err
	}

	return len(backups), nil
}

func Allow(name string, permissions []string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	granted, err := normalizeGrantedPermissions(permissions)
	if err != nil {
		return nil, err
	}
	if len(granted) == 0 {
		return nil, errors.New("at least one permission is required")
	}

	state, err := readGrantedState()
	if err != nil {
		return nil, err
	}

	current, err := normalizeGrantedPermissions(append(state[skill.Spec.Name], granted...))
	if err != nil {
		return nil, err
	}
	if slices.Contains(current, "*") {
		current = []string{"*"}
	}
	sort.Strings(current)

	state[skill.Spec.Name] = current
	if err := writeGrantedState(state); err != nil {
		return nil, err
	}

	_ = appendAuditEvent(auditEvent{
		Time:    time.Now().UTC().Format(time.RFC3339),
		Action:  "allow",
		Skill:   skill.Spec.Name,
		Source:  skill.Metadata.Source,
		Success: true,
		Details: map[string]any{
			"permissions": current,
		},
	})
	return current, nil
}

func Revoke(name string, permissions []string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	state, err := readGrantedState()
	if err != nil {
		return nil, err
	}

	if len(permissions) == 0 {
		delete(state, skill.Spec.Name)
		if err := writeGrantedState(state); err != nil {
			return nil, err
		}
		_ = appendAuditEvent(auditEvent{
			Time:    time.Now().UTC().Format(time.RFC3339),
			Action:  "revoke",
			Skill:   skill.Spec.Name,
			Source:  skill.Metadata.Source,
			Success: true,
			Details: map[string]any{
				"permissions": []string{},
			},
		})
		return nil, nil
	}

	toRemove, err := normalizeGrantedPermissions(permissions)
	if err != nil {
		return nil, err
	}

	removeSet := make(map[string]bool, len(toRemove))
	for _, permission := range toRemove {
		removeSet[permission] = true
	}

	current, err := normalizeGrantedPermissions(state[skill.Spec.Name])
	if err != nil {
		return nil, err
	}

	filtered := make([]string, 0, len(current))
	for _, permission := range current {
		if !removeSet[permission] {
			filtered = append(filtered, permission)
		}
	}
	sort.Strings(filtered)

	if len(filtered) == 0 {
		delete(state, skill.Spec.Name)
	} else {
		state[skill.Spec.Name] = filtered
	}
	if err := writeGrantedState(state); err != nil {
		return nil, err
	}

	_ = appendAuditEvent(auditEvent{
		Time:    time.Now().UTC().Format(time.RFC3339),
		Action:  "revoke",
		Skill:   skill.Spec.Name,
		Source:  skill.Metadata.Source,
		Success: true,
		Details: map[string]any{
			"permissions": filtered,
		},
	})
	return filtered, nil
}

func Granted(name string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	state, err := readGrantedState()
	if err != nil {
		return nil, err
	}

	grants, err := normalizeGrantedPermissions(state[skill.Spec.Name])
	if err != nil {
		return nil, err
	}
	sort.Strings(grants)
	return grants, nil
}

func AllowSession(name string, permissions []string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	granted, err := normalizeGrantedPermissions(permissions)
	if err != nil {
		return nil, err
	}
	if len(granted) == 0 {
		return nil, errors.New("at least one permission is required")
	}

	sessionGrantMu.Lock()
	defer sessionGrantMu.Unlock()

	key := sessionGrantKey(skill.Spec.Name)
	current, err := normalizeGrantedPermissions(append(sessionGrants[key], granted...))
	if err != nil {
		return nil, err
	}
	if slices.Contains(current, "*") {
		current = []string{"*"}
	}
	sort.Strings(current)
	sessionGrants[key] = current

	return slices.Clone(current), nil
}

func RevokeSession(name string, permissions []string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	sessionGrantMu.Lock()
	defer sessionGrantMu.Unlock()
	key := sessionGrantKey(skill.Spec.Name)

	if len(permissions) == 0 {
		delete(sessionGrants, key)
		return nil, nil
	}

	remove, err := normalizeGrantedPermissions(permissions)
	if err != nil {
		return nil, err
	}
	removeSet := map[string]bool{}
	for _, permission := range remove {
		removeSet[permission] = true
	}

	current, err := normalizeGrantedPermissions(sessionGrants[key])
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, len(current))
	for _, permission := range current {
		if !removeSet[permission] {
			out = append(out, permission)
		}
	}
	sort.Strings(out)
	if len(out) == 0 {
		delete(sessionGrants, key)
		return nil, nil
	}
	sessionGrants[key] = out
	return slices.Clone(out), nil
}

func SessionGranted(name string) ([]string, error) {
	skill, err := Get(name)
	if err != nil {
		return nil, err
	}

	sessionGrantMu.Lock()
	defer sessionGrantMu.Unlock()

	values, err := normalizeGrantedPermissions(sessionGrants[sessionGrantKey(skill.Spec.Name)])
	if err != nil {
		return nil, err
	}
	sort.Strings(values)
	return slices.Clone(values), nil
}

func snapshotSessionGrants() map[string][]string {
	sessionGrantMu.Lock()
	defer sessionGrantMu.Unlock()

	root, err := RootDir()
	prefix := ""
	if err == nil {
		prefix = root + "|"
	}

	out := make(map[string][]string, len(sessionGrants))
	for key, values := range sessionGrants {
		name := key
		if prefix != "" {
			if !strings.HasPrefix(key, prefix) {
				continue
			}
			name = strings.TrimPrefix(key, prefix)
		}
		out[name] = slices.Clone(values)
	}
	return out
}

func ReadLogs(name string, last int) ([]string, error) {
	n := strings.TrimSpace(name)
	if n == "" {
		return nil, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}
	if skill, err := Get(n); err == nil {
		n = skill.Spec.Name
	} else if !errors.Is(err, ErrSkillNotFound) {
		return nil, err
	}

	path, err := skillLogPath(n)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	lines := []string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if last > 0 && len(lines) > last {
		lines = lines[len(lines)-last:]
	}

	return lines, nil
}

func Enable(name string) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	state, err := readEnabledState()
	if err != nil {
		return err
	}

	state[skill.Spec.Name] = true
	return writeEnabledState(state)
}

func Disable(name string) error {
	skill, err := Get(name)
	if err != nil {
		return err
	}

	state, err := readEnabledState()
	if err != nil {
		return err
	}

	delete(state, skill.Spec.Name)
	return writeEnabledState(state)
}

func Get(name string) (Skill, error) {
	n := strings.TrimSpace(name)
	if n == "" {
		return Skill{}, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}

	all, err := List()
	if err != nil {
		return Skill{}, err
	}

	for _, skill := range all {
		if skill.Spec.Name == n {
			return skill, nil
		}
	}

	return Skill{}, fmt.Errorf("%w: %s", ErrSkillNotFound, n)
}

func Verify(name string) (result VerifyResult, err error) {
	skill, err := Get(name)
	if err != nil {
		return VerifyResult{}, err
	}

	result = VerifyResult{
		Skill:      skill.Spec.Name,
		Provenance: skill.Spec.Provenance,
	}
	defer func() {
		_ = appendAuditEvent(auditEvent{
			Time:    time.Now().UTC().Format(time.RFC3339),
			Action:  "verify",
			Skill:   skill.Spec.Name,
			Source:  skill.Metadata.Source,
			Success: err == nil,
			Details: map[string]any{
				"digest":   result.Digest,
				"verified": result.Verified,
				"signed":   result.Signed,
			},
		})
	}()

	policy, err := loadPolicy()
	if err != nil {
		return VerifyResult{}, err
	}
	result, err = verifyProvenanceForDir(skill.Dir, skill.Spec, policy)
	if err != nil {
		return VerifyResult{}, err
	}

	skill.Metadata.Digest = result.Digest
	skill.Metadata.Verified = result.Verified
	skill.Metadata.Signed = result.Signed
	skill.Metadata.PublicKey = result.PublicKey
	if result.Verified {
		skill.Metadata.VerifiedAt = time.Now().UTC().Format(time.RFC3339)
	}
	if err := writeMetadata(skill.Dir, skill.Metadata); err != nil {
		return VerifyResult{}, err
	}

	return result, nil
}

func Run(ctx context.Context, name string, args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	return RunWithOptions(ctx, name, args, stdin, stdout, stderr, RunOptions{})
}

func RunWithOptions(ctx context.Context, name string, args []string, stdin io.Reader, stdout, stderr io.Writer, opts RunOptions) (err error) {
	skill, err := Get(name)
	if err != nil {
		return err
	}
	started := time.Now()
	defer func() {
		_ = appendAuditEvent(auditEvent{
			Time:    time.Now().UTC().Format(time.RFC3339),
			Action:  "run",
			Skill:   skill.Spec.Name,
			Source:  skill.Metadata.Source,
			Success: err == nil,
			Details: map[string]any{
				"dry_run":  opts.DryRun,
				"duration": time.Since(started).String(),
			},
		})
	}()

	if !skill.Enabled {
		return fmt.Errorf("%w: %s", ErrSkillNotEnabled, skill.Spec.Name)
	}

	grantedPerms, err := grantedPermissions(skill.Spec.Name, opts.GrantedPermissions)
	if err != nil {
		return err
	}
	if err := ensurePermissionsGranted(skill.Spec.Name, skill.Spec.Permissions.Required, grantedPerms); err != nil {
		return err
	}
	policy, err := loadPolicy()
	if err != nil {
		return err
	}
	if err := enforceRunPolicy(policy, skill); err != nil {
		return err
	}

	command := resolveCommand(skill.Dir, skill.Spec.Command)
	runArgs := append(slices.Clone(skill.Spec.Args), args...)
	if err := enforceSandboxPolicy(skill, runArgs); err != nil {
		return err
	}

	if opts.DryRun {
		if stdout == nil {
			stdout = io.Discard
		}
		_, _ = fmt.Fprintf(stdout, "Skill: %s\n", skill.Spec.Name)
		_, _ = fmt.Fprintf(stdout, "Command: %s %s\n", command, strings.Join(runArgs, " "))
		_, _ = fmt.Fprintf(stdout, "Required permissions: %s\n", strings.Join(skill.Spec.Permissions.Required, ", "))
		_, _ = fmt.Fprintf(stdout, "Granted permissions: %s\n", strings.Join(grantedPerms, ", "))
		_, _ = fmt.Fprintf(stdout, "Sandbox timeout: %ds\n", skill.Spec.Sandbox.TimeoutSeconds)
		_, _ = fmt.Fprintf(stdout, "Sandbox max output bytes: %d\n", skill.Spec.Sandbox.MaxOutputBytes)
		_, _ = fmt.Fprintf(stdout, "Sandbox network: %t\n", derefBool(skill.Spec.Sandbox.AllowNetwork, true))
		_, _ = fmt.Fprintf(stdout, "Sandbox filesystem: %t\n", derefBool(skill.Spec.Sandbox.AllowFilesystem, true))
		_, _ = fmt.Fprintf(stdout, "Sandbox allowed paths: %s\n", joinOrDashForRun(skill.Spec.Sandbox.AllowedPaths))
		return nil
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = time.Duration(skill.Spec.Sandbox.TimeoutSeconds) * time.Second
	}
	runCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	cmd := exec.CommandContext(runCtx, command, runArgs...)
	cmd.Dir = skill.Dir
	cmd.Stdin = stdin
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("OLLAMA_SKILL_SANDBOX_ALLOW_NETWORK=%t", derefBool(skill.Spec.Sandbox.AllowNetwork, true)),
		fmt.Sprintf("OLLAMA_SKILL_SANDBOX_ALLOW_FILESYSTEM=%t", derefBool(skill.Spec.Sandbox.AllowFilesystem, true)),
		fmt.Sprintf("OLLAMA_SKILL_SANDBOX_ALLOWED_PATHS=%s", strings.Join(skill.Spec.Sandbox.AllowedPaths, string(os.PathListSeparator))),
		fmt.Sprintf("OLLAMA_SKILL_TIMEOUT_SECONDS=%d", skill.Spec.Sandbox.TimeoutSeconds),
	)

	logWriter, closeLog := openSkillLog(skill.Spec.Name, runArgs)
	if closeLog != nil {
		defer func() {
			closeLog(err == nil)
		}()
	}

	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}
	limit := opts.MaxOutputBytes
	if limit <= 0 {
		limit = skill.Spec.Sandbox.MaxOutputBytes
	}

	var outWriter io.Writer = stdout
	var errWriter io.Writer = stderr
	if limit > 0 {
		capOut := newCappedWriter(outWriter, limit)
		capErr := newCappedWriter(errWriter, limit)
		outWriter = capOut
		errWriter = capErr
	}

	if logWriter != nil {
		cmd.Stdout = io.MultiWriter(outWriter, logWriter)
		cmd.Stderr = io.MultiWriter(errWriter, logWriter)
	} else {
		cmd.Stdout = outWriter
		cmd.Stderr = errWriter
	}

	err = cmd.Run()
	return err
}

func resolveCommand(skillDir, command string) string {
	if command == "" || filepath.IsAbs(command) {
		return command
	}

	if strings.Contains(command, string(filepath.Separator)) ||
		strings.HasPrefix(command, ".") {
		return filepath.Join(skillDir, command)
	}

	return command
}

func isGitLikeSource(source string) bool {
	s := strings.ToLower(strings.TrimSpace(source))
	return strings.HasPrefix(s, "http://") ||
		strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "git@") ||
		strings.HasSuffix(s, ".git") ||
		isGitHubShorthand(s)
}

func normalizeStringList(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		v := strings.TrimSpace(value)
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizePermissions(values []string) ([]string, error) {
	return normalizePermissionsInternal(values, false)
}

func normalizeGrantedPermissions(values []string) ([]string, error) {
	return normalizePermissionsInternal(values, true)
}

func normalizePermissionsInternal(values []string, allowWildcard bool) ([]string, error) {
	values = normalizeStringList(values)
	for _, permission := range values {
		if allowWildcard && permission == "*" {
			continue
		}
		if !validPermission.MatchString(permission) {
			return nil, fmt.Errorf("invalid permission %q", permission)
		}
	}
	return values, nil
}

func grantedPermissions(skillName string, overrides []string) ([]string, error) {
	base, err := Granted(skillName)
	if err != nil {
		return nil, err
	}
	session, err := SessionGranted(skillName)
	if err != nil {
		return nil, err
	}
	base = append(base, session...)
	base = append(base, normalizeStringList(overrides)...)
	base = append(base, splitCommaList(os.Getenv(envSkillAllow))...)
	out, err := normalizeGrantedPermissions(base)
	if err != nil {
		return nil, err
	}
	if slices.Contains(out, "*") {
		return []string{"*"}, nil
	}
	sort.Strings(out)
	return out, nil
}

func splitCommaList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}

	parts := strings.Split(raw, ",")
	return normalizeStringList(parts)
}

func normalizeFields(fields []FieldSpec) error {
	seen := map[string]bool{}
	for i := range fields {
		fields[i].Name = strings.TrimSpace(fields[i].Name)
		fields[i].Type = strings.TrimSpace(strings.ToLower(fields[i].Type))
		fields[i].Description = strings.TrimSpace(fields[i].Description)
		if fields[i].Name == "" {
			return errors.New("schema field name is required")
		}
		if fields[i].Type == "" {
			return fmt.Errorf("schema field %q type is required", fields[i].Name)
		}
		if !validFieldType.MatchString(fields[i].Type) {
			return fmt.Errorf("invalid schema field type %q", fields[i].Type)
		}
		if seen[fields[i].Name] {
			return fmt.Errorf("duplicate schema field %q", fields[i].Name)
		}
		seen[fields[i].Name] = true
	}
	return nil
}

func derefBool(v *bool, fallback bool) bool {
	if v == nil {
		return fallback
	}
	return *v
}

func joinOrDashForRun(values []string) string {
	if len(values) == 0 {
		return "-"
	}
	return strings.Join(values, ", ")
}

func enforceSandboxPolicy(skill Skill, runArgs []string) error {
	allowNetwork := derefBool(skill.Spec.Sandbox.AllowNetwork, true)
	allowFS := derefBool(skill.Spec.Sandbox.AllowFilesystem, true)

	for _, permission := range skill.Spec.Permissions.Required {
		if strings.HasPrefix(permission, "network.") && !allowNetwork {
			return &SandboxError{
				Skill:  skill.Spec.Name,
				Reason: fmt.Sprintf("permission %q requires network but sandbox.allow_network is false", permission),
			}
		}
		if strings.HasPrefix(permission, "filesystem.") && !allowFS {
			return &SandboxError{
				Skill:  skill.Spec.Name,
				Reason: fmt.Sprintf("permission %q requires filesystem but sandbox.allow_filesystem is false", permission),
			}
		}
	}

	if len(skill.Spec.Sandbox.AllowedPaths) == 0 || !allowFS {
		return nil
	}
	if !requiresFilesystem(skill.Spec.Permissions.Required) {
		return nil
	}

	allowed := make([]string, 0, len(skill.Spec.Sandbox.AllowedPaths))
	for _, path := range skill.Spec.Sandbox.AllowedPaths {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		allowed = append(allowed, filepath.Clean(abs))
	}

	for _, arg := range runArgs {
		if !looksLikePathArg(arg) {
			continue
		}
		candidate := arg
		if strings.HasPrefix(candidate, "~") {
			if home, err := os.UserHomeDir(); err == nil {
				candidate = filepath.Join(home, strings.TrimPrefix(candidate, "~"))
			}
		}
		if !filepath.IsAbs(candidate) {
			candidate = filepath.Join(skill.Dir, candidate)
		}
		candidate = filepath.Clean(candidate)
		if !pathWithinAny(candidate, allowed) {
			return &SandboxError{
				Skill:  skill.Spec.Name,
				Reason: fmt.Sprintf("path %q is outside sandbox.allowed_paths", arg),
			}
		}
	}

	return nil
}

func requiresFilesystem(required []string) bool {
	for _, permission := range required {
		if strings.HasPrefix(permission, "filesystem.") {
			return true
		}
	}
	return false
}

func looksLikePathArg(value string) bool {
	if strings.TrimSpace(value) == "" {
		return false
	}
	if strings.HasPrefix(value, "-") {
		return false
	}
	return strings.Contains(value, "/") || strings.HasPrefix(value, ".") || strings.HasPrefix(value, "~")
}

func pathWithinAny(candidate string, allowed []string) bool {
	for _, base := range allowed {
		rel, err := filepath.Rel(base, candidate)
		if err != nil {
			continue
		}
		if rel == "." || (!strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != "..") {
			return true
		}
	}
	return false
}

type cappedWriter struct {
	w     io.Writer
	limit int64
	mu    sync.Mutex
	n     int64
}

func newCappedWriter(w io.Writer, limit int64) *cappedWriter {
	return &cappedWriter{w: w, limit: limit}
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.limit > 0 && w.n+int64(len(p)) > w.limit {
		remaining := w.limit - w.n
		if remaining > 0 {
			n, _ := w.w.Write(p[:remaining])
			w.n += int64(n)
		}
		return 0, fmt.Errorf("%w: output exceeded max_output_bytes=%d", ErrSandboxDenied, w.limit)
	}

	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, err
}

func ensurePermissionsGranted(skillName string, required, granted []string) error {
	if len(required) == 0 {
		return nil
	}

	grantedSet := map[string]bool{}
	hasWildcard := false
	for _, permission := range granted {
		grantedSet[permission] = true
		if permission == "*" {
			hasWildcard = true
		}
	}

	if hasWildcard {
		return nil
	}

	var missing []string
	for _, permission := range required {
		if !grantedSet[permission] {
			missing = append(missing, permission)
		}
	}

	if len(missing) == 0 {
		return nil
	}

	return &PermissionError{
		Skill:   skillName,
		Missing: missing,
	}
}

func readEnabledState() (map[string]bool, error) {
	if err := ensureLayout(); err != nil {
		return nil, err
	}

	path, err := enabledPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]bool{}, nil
	}
	if err != nil {
		return nil, err
	}

	state := map[string]bool{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}

	return state, nil
}

func writeEnabledState(state map[string]bool) error {
	if err := ensureLayout(); err != nil {
		return err
	}

	path, err := enabledPath()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	tmpFile, err := os.CreateTemp(filepath.Dir(path), "enabled-*.json")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpPath, path)
}

func readGrantedState() (map[string][]string, error) {
	if err := ensureLayout(); err != nil {
		return nil, err
	}

	path, err := grantsPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string][]string{}, nil
	}
	if err != nil {
		return nil, err
	}

	state := map[string][]string{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}

	out := make(map[string][]string, len(state))
	for name, values := range state {
		grants, err := normalizeGrantedPermissions(values)
		if err != nil {
			return nil, err
		}
		if slices.Contains(grants, "*") {
			grants = []string{"*"}
		}
		sort.Strings(grants)
		out[name] = grants
	}
	return out, nil
}

func writeGrantedState(state map[string][]string) error {
	if err := ensureLayout(); err != nil {
		return err
	}

	path, err := grantsPath()
	if err != nil {
		return err
	}

	clean := make(map[string][]string, len(state))
	for name, values := range state {
		grants, err := normalizeGrantedPermissions(values)
		if err != nil {
			return err
		}
		if slices.Contains(grants, "*") {
			grants = []string{"*"}
		}
		sort.Strings(grants)
		if len(grants) > 0 {
			clean[name] = grants
		}
	}

	data, err := json.MarshalIndent(clean, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	tmpFile, err := os.CreateTemp(filepath.Dir(path), "grants-*.json")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpPath, path)
}

func readMetadata(skillDir string) (Metadata, error) {
	data, err := os.ReadFile(filepath.Join(skillDir, metadataFile))
	if errors.Is(err, os.ErrNotExist) {
		return Metadata{}, nil
	}
	if err != nil {
		return Metadata{}, err
	}

	var metadata Metadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return Metadata{}, err
	}
	return metadata, nil
}

func writeMetadata(skillDir string, metadata Metadata) error {
	path := filepath.Join(skillDir, metadataFile)
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func loadPolicy() (Policy, error) {
	path, err := policyPath()
	if err != nil {
		return Policy{}, err
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Policy{}, nil
	}
	if err != nil {
		return Policy{}, err
	}

	var policy Policy
	if err := json.Unmarshal(data, &policy); err != nil {
		return Policy{}, fmt.Errorf("parse %s: %w", path, err)
	}

	return normalizePolicy(policy)
}

func normalizePolicy(policy Policy) (Policy, error) {
	var err error

	policy.AllowedSources = normalizeStringList(policy.AllowedSources)
	policy.AllowedPermissions, err = normalizePermissionsInternal(policy.AllowedPermissions, true)
	if err != nil {
		return Policy{}, err
	}
	policy.DeniedPermissions, err = normalizePermissionsInternal(policy.DeniedPermissions, true)
	if err != nil {
		return Policy{}, err
	}
	if policy.MaxTimeoutSeconds < 0 {
		return Policy{}, errors.New("policy max_timeout_seconds cannot be negative")
	}
	if policy.MaxOutputBytes < 0 {
		return Policy{}, errors.New("policy max_output_bytes cannot be negative")
	}

	if len(policy.TrustedPublicKeys) > 0 {
		keys := make([]string, 0, len(policy.TrustedPublicKeys))
		for _, raw := range policy.TrustedPublicKeys {
			key, err := parseEd25519PublicKey(raw)
			if err != nil {
				return Policy{}, fmt.Errorf("invalid trusted_public_keys entry: %w", err)
			}
			keys = append(keys, base64.StdEncoding.EncodeToString(key))
		}
		policy.TrustedPublicKeys = normalizeStringList(keys)
	}

	return policy, nil
}

func enforceInstallPolicy(policy Policy, spec Spec, source string) error {
	if len(policy.AllowedSources) > 0 {
		src := strings.TrimSpace(source)
		allowed := false
		for _, prefix := range policy.AllowedSources {
			if strings.HasPrefix(src, prefix) {
				allowed = true
				break
			}
		}
		if !allowed {
			return &PolicyError{
				Skill:  spec.Name,
				Reason: fmt.Sprintf("source %q is not permitted by policy.allowed_sources", source),
			}
		}
	}

	if err := enforcePolicyPermissions(policy, spec.Name, spec.Permissions.Required); err != nil {
		return err
	}

	if policy.AllowNetwork != nil && !*policy.AllowNetwork {
		if requiresPermissionPrefix(spec.Permissions.Required, "network.") || derefBool(spec.Sandbox.AllowNetwork, true) {
			return &PolicyError{
				Skill:  spec.Name,
				Reason: "network access is not permitted by policy",
			}
		}
	}
	if policy.AllowFilesystem != nil && !*policy.AllowFilesystem {
		if requiresPermissionPrefix(spec.Permissions.Required, "filesystem.") || derefBool(spec.Sandbox.AllowFilesystem, true) {
			return &PolicyError{
				Skill:  spec.Name,
				Reason: "filesystem access is not permitted by policy",
			}
		}
	}
	if policy.MaxTimeoutSeconds > 0 && spec.Sandbox.TimeoutSeconds > policy.MaxTimeoutSeconds {
		return &PolicyError{
			Skill:  spec.Name,
			Reason: fmt.Sprintf("sandbox timeout %ds exceeds policy max_timeout_seconds=%d", spec.Sandbox.TimeoutSeconds, policy.MaxTimeoutSeconds),
		}
	}
	if policy.MaxOutputBytes > 0 && spec.Sandbox.MaxOutputBytes > policy.MaxOutputBytes {
		return &PolicyError{
			Skill:  spec.Name,
			Reason: fmt.Sprintf("sandbox max output bytes %d exceeds policy max_output_bytes=%d", spec.Sandbox.MaxOutputBytes, policy.MaxOutputBytes),
		}
	}
	if policy.RequireSHA256 && strings.TrimSpace(spec.Provenance.SHA256) == "" {
		return &PolicyError{
			Skill:  spec.Name,
			Reason: "provenance.sha256 is required by policy",
		}
	}
	if policy.RequireSignature && (strings.TrimSpace(spec.Provenance.Signature) == "" || strings.TrimSpace(spec.Provenance.PublicKey) == "") {
		return &PolicyError{
			Skill:  spec.Name,
			Reason: "provenance signature/public_key are required by policy",
		}
	}

	return nil
}

func enforceRunPolicy(policy Policy, skill Skill) error {
	if err := enforceInstallPolicy(policy, skill.Spec, skill.Metadata.Source); err != nil {
		return err
	}
	if policy.RequireSHA256 && strings.TrimSpace(skill.Metadata.Digest) == "" {
		return &PolicyError{
			Skill:  skill.Spec.Name,
			Reason: "verified digest metadata is required by policy",
		}
	}
	if policy.RequireSignature && !skill.Metadata.Signed {
		return &PolicyError{
			Skill:  skill.Spec.Name,
			Reason: "verified signature metadata is required by policy",
		}
	}
	if len(policy.TrustedPublicKeys) > 0 && strings.TrimSpace(skill.Metadata.PublicKey) != "" {
		key := strings.TrimSpace(skill.Metadata.PublicKey)
		if !slices.Contains(policy.TrustedPublicKeys, key) {
			return &PolicyError{
				Skill:  skill.Spec.Name,
				Reason: "skill signature key is not trusted by policy",
			}
		}
	}

	return nil
}

func enforcePolicyPermissions(policy Policy, skillName string, required []string) error {
	if len(policy.AllowedPermissions) > 0 {
		allowed := make(map[string]bool, len(policy.AllowedPermissions))
		for _, permission := range policy.AllowedPermissions {
			allowed[permission] = true
		}
		for _, permission := range required {
			if allowed["*"] {
				break
			}
			if !allowed[permission] {
				return &PolicyError{
					Skill:  skillName,
					Reason: fmt.Sprintf("permission %q is not allowed by policy", permission),
				}
			}
		}
	}

	if len(policy.DeniedPermissions) > 0 {
		denied := make(map[string]bool, len(policy.DeniedPermissions))
		for _, permission := range policy.DeniedPermissions {
			denied[permission] = true
		}
		if denied["*"] && len(required) > 0 {
			return &PolicyError{
				Skill:  skillName,
				Reason: "all permissions are denied by policy",
			}
		}
		for _, permission := range required {
			if denied[permission] {
				return &PolicyError{
					Skill:  skillName,
					Reason: fmt.Sprintf("permission %q is denied by policy", permission),
				}
			}
		}
	}

	return nil
}

func requiresPermissionPrefix(required []string, prefix string) bool {
	for _, permission := range required {
		if strings.HasPrefix(permission, prefix) {
			return true
		}
	}
	return false
}

func verifyProvenanceForDir(dir string, spec Spec, policy Policy) (VerifyResult, error) {
	result := VerifyResult{
		Skill:      spec.Name,
		Provenance: spec.Provenance,
	}

	digest, err := computeSkillDigest(dir)
	if err != nil {
		return VerifyResult{}, err
	}
	result.Digest = digest

	if spec.Provenance.SHA256 != "" && !strings.EqualFold(spec.Provenance.SHA256, digest) {
		return VerifyResult{}, &ProvenanceError{
			Skill:  spec.Name,
			Reason: fmt.Sprintf("sha256 mismatch: expected %s got %s", spec.Provenance.SHA256, digest),
		}
	}
	if spec.Provenance.SHA256 != "" {
		result.Verified = true
	}
	if policy.RequireSHA256 && spec.Provenance.SHA256 == "" {
		return VerifyResult{}, &PolicyError{
			Skill:  spec.Name,
			Reason: "provenance.sha256 is required by policy",
		}
	}

	if spec.Provenance.Signature != "" {
		publicKey, err := parseEd25519PublicKey(spec.Provenance.PublicKey)
		if err != nil {
			return VerifyResult{}, &ProvenanceError{
				Skill:  spec.Name,
				Reason: fmt.Sprintf("invalid public key: %v", err),
			}
		}
		signature, err := parseEd25519Signature(spec.Provenance.Signature)
		if err != nil {
			return VerifyResult{}, &ProvenanceError{
				Skill:  spec.Name,
				Reason: fmt.Sprintf("invalid signature: %v", err),
			}
		}
		digestBytes, _ := hex.DecodeString(digest)
		if !ed25519.Verify(publicKey, digestBytes, signature) {
			return VerifyResult{}, &ProvenanceError{
				Skill:  spec.Name,
				Reason: "signature verification failed",
			}
		}
		result.PublicKey = base64.StdEncoding.EncodeToString(publicKey)
		result.Signed = true
		result.Verified = true

		if len(policy.TrustedPublicKeys) > 0 && !slices.Contains(policy.TrustedPublicKeys, result.PublicKey) {
			return VerifyResult{}, &PolicyError{
				Skill:  spec.Name,
				Reason: "skill signature key is not trusted by policy",
			}
		}
	} else if policy.RequireSignature {
		return VerifyResult{}, &PolicyError{
			Skill:  spec.Name,
			Reason: "provenance signature/public_key are required by policy",
		}
	}

	return result, nil
}

func computeSkillDigest(dir string) (string, error) {
	files := make([]string, 0, 32)
	if err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}

		rel = filepath.ToSlash(rel)
		if rel == metadataFile {
			return nil
		}
		files = append(files, rel)
		return nil
	}); err != nil {
		return "", err
	}

	sort.Strings(files)
	hasher := sha256.New()
	for _, rel := range files {
		_, _ = hasher.Write([]byte(rel))
		_, _ = hasher.Write([]byte{0})
		path := filepath.Join(dir, filepath.FromSlash(rel))
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		if rel == skillManifestFile {
			data, err = canonicalManifestBytes(path)
			if err != nil {
				return "", err
			}
		}
		_, _ = hasher.Write(data)
		_, _ = hasher.Write([]byte{0})
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func canonicalManifestBytes(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var spec Spec
	if err := toml.Unmarshal(data, &spec); err != nil {
		return nil, err
	}
	if err := spec.normalize(filepath.Base(filepath.Dir(path))); err != nil {
		return nil, err
	}
	spec.Provenance = ProvenanceSpec{}
	return toml.Marshal(spec)
}

func parseEd25519PublicKey(raw string) (ed25519.PublicKey, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, errors.New("public key is required")
	}

	if strings.HasPrefix(trimmed, "-----BEGIN") {
		block, _ := pem.Decode([]byte(trimmed))
		if block == nil {
			return nil, errors.New("invalid PEM public key")
		}
		publicKeyAny, err := x509.ParsePKIXPublicKey(block.Bytes)
		if err == nil {
			publicKey, ok := publicKeyAny.(ed25519.PublicKey)
			if !ok {
				return nil, errors.New("public key is not ed25519")
			}
			return slices.Clone(publicKey), nil
		}
		if len(block.Bytes) == ed25519.PublicKeySize {
			return ed25519.PublicKey(slices.Clone(block.Bytes)), nil
		}
		return nil, err
	}

	decoded, err := decodeBase64(raw)
	if err != nil {
		return nil, err
	}
	if len(decoded) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("public key length must be %d bytes", ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(decoded), nil
}

func parseEd25519Signature(raw string) ([]byte, error) {
	signature, err := decodeBase64(raw)
	if err != nil {
		return nil, err
	}
	if len(signature) != ed25519.SignatureSize {
		return nil, fmt.Errorf("signature length must be %d bytes", ed25519.SignatureSize)
	}
	return signature, nil
}

func decodeBase64(raw string) ([]byte, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, errors.New("value is empty")
	}

	if out, err := base64.StdEncoding.DecodeString(trimmed); err == nil {
		return out, nil
	}
	if out, err := base64.RawStdEncoding.DecodeString(trimmed); err == nil {
		return out, nil
	}
	return nil, errors.New("invalid base64 value")
}

type auditEvent struct {
	Time    string         `json:"time"`
	Action  string         `json:"action"`
	Skill   string         `json:"skill,omitempty"`
	Source  string         `json:"source,omitempty"`
	Success bool           `json:"success"`
	Details map[string]any `json:"details,omitempty"`
}

func appendAuditEvent(event auditEvent) error {
	if strings.TrimSpace(event.Action) == "" {
		return nil
	}
	if err := ensureLayout(); err != nil {
		return err
	}

	path, err := auditPath()
	if err != nil {
		return err
	}
	if err := rotateAuditLogIfNeeded(path); err != nil {
		return err
	}
	if event.Time == "" {
		event.Time = time.Now().UTC().Format(time.RFC3339)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(event)
}

func rotateAuditLogIfNeeded(path string) error {
	if auditMaxLogBytes <= 0 {
		return nil
	}

	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Size() < auditMaxLogBytes {
		return nil
	}

	rotated := fmt.Sprintf("%s.%s", path, time.Now().UTC().Format("20060102T150405Z"))
	for i := 0; ; i++ {
		candidate := rotated
		if i > 0 {
			candidate = fmt.Sprintf("%s-%d", rotated, i)
		}
		if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
			rotated = candidate
			break
		}
	}
	if err := os.Rename(path, rotated); err != nil {
		return err
	}

	return pruneAuditLogBackups(path, auditMaxLogBackups)
}

func pruneAuditLogBackups(path string, keep int) error {
	if keep <= 0 {
		return nil
	}

	dir := filepath.Dir(path)
	base := filepath.Base(path) + "."
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	backups := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if strings.HasPrefix(entry.Name(), base) {
			backups = append(backups, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(backups)
	slices.Reverse(backups)
	for _, backup := range backups[keep:] {
		if err := os.Remove(backup); err != nil {
			return err
		}
	}
	return nil
}

func ReadAuditLogs(last int) ([]string, error) {
	path, err := auditPath()
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()

	lines := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if last > 0 && len(lines) > last {
		lines = lines[len(lines)-last:]
	}
	return lines, nil
}

func resolveInstallSource(source string) (sourceSpec, error) {
	raw := strings.TrimSpace(source)
	if raw == "" {
		return sourceSpec{}, errors.New("source is required")
	}

	if abs, ok, err := resolveLocalSource(raw); err != nil {
		return sourceSpec{}, err
	} else if ok {
		return sourceSpec{
			raw:       abs,
			localPath: abs,
		}, nil
	}

	base, ref := splitPinnedRef(raw)
	if strings.TrimSpace(ref) != "" {
		if absBase, ok, err := resolveLocalSource(base); err != nil {
			return sourceSpec{}, err
		} else if ok {
			if _, err := os.Stat(filepath.Join(absBase, ".git")); err == nil {
				return sourceSpec{
					raw:        absBase + "@" + ref,
					repository: absBase,
					ref:        ref,
					isGit:      true,
				}, nil
			}
		}
	}

	if !isGitLikeSource(base) && !isGitHubShorthand(base) {
		return sourceSpec{}, fmt.Errorf("source %q is not a local directory or git/GitHub source", source)
	}
	if strings.TrimSpace(ref) == "" {
		return sourceSpec{}, errors.New("git/GitHub installs require a pinned ref (example: owner/repo@tag or https://...git@commit)")
	}

	repository, err := normalizeGitRepository(base)
	if err != nil {
		return sourceSpec{}, err
	}

	return sourceSpec{
		raw:        repository + "@" + ref,
		repository: repository,
		ref:        ref,
		isGit:      true,
	}, nil
}

func resolveLocalSource(source string) (string, bool, error) {
	abs, err := filepath.Abs(source)
	if err != nil {
		return "", false, err
	}
	stat, err := os.Stat(abs)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", false, nil
		}
		return "", false, err
	}
	if !stat.IsDir() {
		return "", false, fmt.Errorf("source %q is not a directory", source)
	}
	return abs, true, nil
}

func splitPinnedRef(source string) (string, string) {
	s := strings.TrimSpace(source)
	idx := strings.LastIndex(s, "@")
	if idx <= 0 {
		return s, ""
	}

	if strings.HasPrefix(s, "git@") && idx == strings.Index(s, "@") {
		return s, ""
	}

	return strings.TrimSpace(s[:idx]), strings.TrimSpace(s[idx+1:])
}

func isGitHubShorthand(source string) bool {
	s := strings.TrimSpace(source)
	if s == "" || strings.Contains(s, "://") || strings.HasPrefix(s, "git@") {
		return false
	}
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") {
		return false
	}
	parts := strings.Split(s, "/")
	if len(parts) != 2 {
		return false
	}
	return validSkillName.MatchString(parts[0]) && validSkillName.MatchString(strings.TrimSuffix(parts[1], ".git"))
}

func normalizeGitRepository(base string) (string, error) {
	s := strings.TrimSpace(base)
	if s == "" {
		return "", errors.New("repository is required")
	}

	if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") || strings.HasPrefix(s, "git@") {
		return s, nil
	}

	if isGitHubShorthand(s) {
		if strings.HasSuffix(s, ".git") {
			return "https://github.com/" + s, nil
		}
		return "https://github.com/" + s + ".git", nil
	}

	return "", fmt.Errorf("unsupported git source %q", base)
}

func checkoutGitSource(repository, ref string) (string, string, error) {
	tmpDir, err := os.MkdirTemp("", "ollama-skill-clone-*")
	if err != nil {
		return "", "", err
	}

	clone := exec.Command("git", "clone", "--quiet", repository, tmpDir)
	if out, err := clone.CombinedOutput(); err != nil {
		os.RemoveAll(tmpDir)
		return "", "", fmt.Errorf("clone skill source: %w: %s", err, strings.TrimSpace(string(out)))
	}

	checkout := exec.Command("git", "-C", tmpDir, "checkout", "--quiet", ref)
	if out, err := checkout.CombinedOutput(); err != nil {
		os.RemoveAll(tmpDir)
		return "", "", fmt.Errorf("checkout pinned ref %q: %w: %s", ref, err, strings.TrimSpace(string(out)))
	}

	rev := exec.Command("git", "-C", tmpDir, "rev-parse", "HEAD")
	out, err := rev.Output()
	if err != nil {
		os.RemoveAll(tmpDir)
		return "", "", fmt.Errorf("resolve pinned ref commit: %w", err)
	}

	return tmpDir, strings.TrimSpace(string(out)), nil
}

func moveToBackup(skillName, currentDir, reason string) (string, error) {
	root, err := backupRootDir()
	if err != nil {
		return "", err
	}

	skillBackupDir := filepath.Join(root, skillName)
	if err := os.MkdirAll(skillBackupDir, 0o755); err != nil {
		return "", err
	}

	stamp := time.Now().UTC().Format("20060102T150405Z")
	safeReason := strings.ReplaceAll(strings.TrimSpace(reason), " ", "-")
	if safeReason == "" {
		safeReason = "snapshot"
	}
	dest := filepath.Join(skillBackupDir, stamp+"-"+safeReason)
	for i := 0; ; i++ {
		candidate := dest
		if i > 0 {
			candidate = fmt.Sprintf("%s-%d", dest, i)
		}
		if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
			dest = candidate
			break
		}
	}

	if err := os.Rename(currentDir, dest); err != nil {
		return "", err
	}
	return dest, nil
}

func listBackups(skillName string) ([]string, error) {
	root, err := backupRootDir()
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(root, skillName)
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	backups := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		backups = append(backups, filepath.Join(dir, entry.Name()))
	}
	sort.Strings(backups)
	slices.Reverse(backups)
	return backups, nil
}

func pruneBackups(skillName string, keep int) error {
	if keep <= 0 {
		return nil
	}
	backups, err := listBackups(skillName)
	if err != nil {
		return err
	}
	if len(backups) <= keep {
		return nil
	}
	for _, backup := range backups[keep:] {
		if err := os.RemoveAll(backup); err != nil {
			return err
		}
	}
	return nil
}

func skillLogPath(skillName string) (string, error) {
	if err := ensureLayout(); err != nil {
		return "", err
	}

	root, err := logRootDir()
	if err != nil {
		return "", err
	}
	if !validSkillName.MatchString(skillName) {
		return "", fmt.Errorf("invalid skill name %q", skillName)
	}
	return filepath.Join(root, skillName+".log"), nil
}

func openSkillLog(skillName string, args []string) (io.Writer, func(success bool)) {
	path, err := skillLogPath(skillName)
	if err != nil {
		return nil, nil
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, nil
	}

	start := time.Now().UTC().Format(time.RFC3339)
	_, _ = fmt.Fprintf(file, "== %s run %s %s ==\n", start, skillName, strings.Join(args, " "))

	return file, func(success bool) {
		status := "ok"
		if !success {
			status = "error"
		}
		_, _ = fmt.Fprintf(file, "== %s status=%s ==\n", time.Now().UTC().Format(time.RFC3339), status)
		_ = file.Close()
	}
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}

		if d.IsDir() {
			// Avoid copying Git metadata into installed skills.
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}

		if !d.Type().IsRegular() {
			return nil
		}

		srcInfo, err := d.Info()
		if err != nil {
			return err
		}

		return copyFile(path, filepath.Join(dst, rel), srcInfo.Mode())
	})
}

func copyFile(src, dst string, mode fs.FileMode) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode.Perm())
	if err != nil {
		return err
	}
	defer dstFile.Close()

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		return err
	}

	return nil
}
