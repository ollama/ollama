// Package compatmigrate converts older Ollama-format GGUF manifests already
// present in a local model store into llama.cpp-compatible manifest-list
// children.
//
// This exists to bridge the llama-server transition without forcing users to
// re-pull large models they already have on disk. Loads wait for a shared
// conversion before starting the runner. Canceling a load does not cancel the
// conversion. Successful conversion replaces the legacy child and reclaims
// its unreferenced blobs.
//
// Once manifest-list publishing has been live for enough releases and stale
// legacy-only local stores are no longer a practical concern, this code should
// be removed and users with unmigrated local artifacts should re-pull.
package compatmigrate

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

var (
	errUnsupportedFamily = errors.New("compat migration unsupported for family")
	errInsufficientSpace = errors.New("insufficient disk space for local compat migration")
)

const (
	// compatMigrationHeadroom (512 MiB) covers the converted projector,
	// config, and manifest blobs plus temp-file slack during conversion.
	compatMigrationHeadroom = 512 << 20

	// compatMigrationMarginDenom pads the estimated converted-model size by
	// 1/4 (25%) to allow for metadata growth and tensor dtype promotions.
	compatMigrationMarginDenom = 4
)

type Migrator interface {
	NeedsMigration(*SourceModel) bool
	Migrate(*SourceModel) (*Result, error)
}

type SourceModel struct {
	Source         model.Name
	Manifest       *manifest.Manifest
	Config         model.ConfigV2
	GGUFPath       string
	GGUF           *gguf.File
	GGUFData       io.ReaderAt
	GGUFDataOffset int64

	ProjectorPath       string
	ProjectorGGUF       *gguf.File
	ProjectorData       io.ReaderAt
	ProjectorDataOffset int64
}

type Result struct {
	ModelKV           outKV
	ModelTensors      []*outTensor
	ProjectorKV       outKV
	ProjectorTensors  []*outTensor
	PreserveProjector bool

	Renderer string
	Parser   string
	Requires string

	ClearRenderer bool
	ClearParser   bool
}

var migratorsByArchitecture = map[string][]Migrator{
	"gemma4":          {gemma4Migrator{}},
	"gemma3":          {embeddingGemmaMigrator{}, gemma3Migrator{}},
	"gemma3n":         {gemma3nMigrator{}},
	"bert":            {snowflakeArcticEmbed2Migrator{}},
	"deepseekocr":     {deepseekOCRMigrator{}},
	"glm4moelite":     {glm47FlashMigrator{}},
	"glmocr":          {glmOCRMigrator{}},
	"gptoss":          {gptossMigrator{}},
	"laguna":          {lagunaMigrator{}},
	"lfm2":            {lfm25ThinkingMigrator{}},
	"llama":           {bakllavaMigrator{}, llama3Migrator{}},
	"llama4":          {llama4Migrator{}},
	"mistral3":        {mistralPixtralMigrator{}},
	"nemotron_h_moe":  {nemotronHMoeMigrator{}},
	"nemotron_h_omni": {nemotron3Migrator{}},
	"olmo3":           {olmo3Migrator{}},
	"qwen35":          {clefMigrator{}, nimbleMigrator{}, qwen35Migrator{}},
	"qwen35moe":       {qwen35Migrator{}},
	"qwen3next":       {qwen3NextMigrator{}},
	"qwen25vl":        {qwen25VLMigrator{}},
	"qwen3vl":         {qwen3VLMigrator{}},
	"qwen3vlmoe":      {qwen3VLMigrator{}},
}

var (
	availableSpaceForPath = availableSpace
	migrationLocks        sync.Map
	migrationInFlight     sync.Map
)

// SetMigratorsForTesting replaces the migration registry for tests.
func SetMigratorsForTesting(migrators map[string][]Migrator) func() {
	previous := migratorsByArchitecture
	migratorsByArchitecture = migrators
	return func() {
		migratorsByArchitecture = previous
	}
}

type localMigration struct {
	done   chan struct{}
	digest string
	err    error
}

// WaitLocalCompatibilityMigration lets a load wait for conversion without
// cancelling shared conversion work when one client disconnects. A nonempty
// result identifies the replacement for the selected manifest, not another
// platform's preferred child.
func WaitLocalCompatibilityMigration(ctx context.Context, name model.Name, selectedDigest string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if !hasCompatibilityMigrators() {
		return "", nil
	}
	if !name.IsFullyQualified() {
		return "", model.Unqualified(name)
	}

	key := name.String() + ":" + selectedDigest
	migration := &localMigration{done: make(chan struct{})}
	if existing, loaded := migrationInFlight.LoadOrStore(key, migration); loaded {
		migration = existing.(*localMigration)
	} else {
		go func() {
			defer migrationInFlight.Delete(key)
			migration.digest, migration.err = ensureLocalCompatibilityMigration(name, selectedDigest)
			if migration.err != nil {
				slog.Warn("local compatibility migration failed", "model", name.DisplayShortest(), "error", migration.err)
			}
			close(migration.done)
		}()
	}
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-migration.done:
		if err := ctx.Err(); err != nil {
			return "", err
		}
		return migration.digest, migration.err
	}
}

func hasCompatibilityMigrators() bool {
	for _, migrators := range migratorsByArchitecture {
		if len(migrators) > 0 {
			return true
		}
	}
	return false
}

func ensureLocalCompatibilityMigration(name model.Name, selectedDigest string) (string, error) {
	if !name.IsFullyQualified() {
		return "", model.Unqualified(name)
	}

	unlock := lockCompatibilityMigration(name.String())
	defer unlock()

	if !manifest.IsDigestReferenceName(name) {
		if digest, err := retireConvertedModel(name, selectedDigest); err != nil || digest != "" {
			return digest, err
		}
	}
	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return "", err
	}

	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return "", err
	}

	source, index, err := migrationSourceFromManifest(&parent, data, selectedDigest)
	if err != nil {
		return "", err
	}

	src, err := loadSourceModelFromManifest(name, source)
	if err != nil {
		return "", err
	}
	defer src.Close()

	migrator := compatibilityMigratorForSource(src)
	if migrator == nil {
		return "", nil
	}
	if manifest.IsDigestReferenceName(name) {
		return "", errors.New("this GGUF requires conversion; use a model tag instead of its original digest")
	}
	if mediaType, ok := unsupportedSourceLayer(source); ok {
		return "", fmt.Errorf("cannot convert legacy GGUF with %s layer; re-pull a compatible model", mediaType)
	}

	// Aliases can share a source blob. Keep their output installation and
	// aborted-output cleanup from racing with another conversion of that blob.
	unlockSource := lockCompatibilityMigration(src.GGUFPath)
	defer unlockSource()

	convertedRef, err := migrateToManifestReference(migrator, src)
	if err != nil {
		return "", err
	}

	// Windows cannot reclaim source files while the converter holds them open.
	if err := src.Close(); err != nil {
		removeConvertedReference(convertedRef)
		return "", err
	}
	if index < 0 {
		parent = manifest.Manifest{SchemaVersion: 2, MediaType: manifest.MediaTypeManifestList, Manifests: []manifest.Manifest{convertedRef}}
	} else {
		parent.Manifests[index] = convertedRef
	}
	if err := replaceConvertedModel(name, data, &parent); err != nil {
		removeConvertedReference(convertedRef)
		return "", err
	}
	slog.Info("completed local compat GGUF migration", "model", name.DisplayShortest())
	return convertedRef.BlobDigest(), nil
}

// unsupportedSourceLayer reports a source layer type the conversion would
// silently drop from the converted child (see copyAncillaryLayers).
func unsupportedSourceLayer(source *manifest.Manifest) (string, bool) {
	for _, layer := range source.Layers {
		switch layer.MediaType {
		case manifest.MediaTypeImageAdapter, manifest.MediaTypeImageEmbed:
			return layer.MediaType, true
		}
	}
	return "", false
}

func lockCompatibilityMigration(key string) func() {
	value, _ := migrationLocks.LoadOrStore(key, &sync.Mutex{})
	mu := value.(*sync.Mutex)
	mu.Lock()
	return mu.Unlock
}

func migrationSourceFromManifest(parent *manifest.Manifest, data []byte, selectedDigest string) (*manifest.Manifest, int, error) {
	if parent.MediaType != manifest.MediaTypeManifestList {
		if selectedDigest != "" && !manifest.SameDigest(selectedDigest, fmt.Sprintf("sha256:%x", sha256.Sum256(data))) {
			return nil, -1, manifest.ErrManifestChanged
		}
		return parent, -1, nil
	}
	for i, child := range parent.Manifests {
		if selectedDigest == "" && child.Format != manifest.FormatGGUF {
			continue
		}
		digest, err := manifest.ChildManifestDigest(child)
		if err != nil {
			return nil, -1, err
		}
		if selectedDigest != "" && !manifest.SameDigest(digest, selectedDigest) {
			continue
		}
		resolved, err := resolveChildManifest(child)
		if err != nil {
			return nil, -1, err
		}
		if selectedDigest == "" && resolved.Format != manifest.FormatGGUF {
			continue
		}
		return resolved, i, nil
	}
	return nil, -1, manifest.ErrManifestChanged
}

// RunnerForManifest reports the runner a GGUF manifest should carry: ggml when
// a compatibility detector fires, llamacpp otherwise. It reads GGUF headers
// only, so it is cheap enough to run whenever a manifest is written.
func RunnerForManifest(source model.Name, mf *manifest.Manifest) (string, error) {
	src, err := loadSourceModelFromManifest(source, mf)
	if err != nil {
		return "", err
	}
	defer src.Close()

	if compatibilityMigratorForSource(src) != nil {
		return manifest.RunnerGGML, nil
	}
	return manifest.RunnerLlamaCPP, nil
}

func compatibilityMigratorForSource(src *SourceModel) Migrator {
	arch := strings.ToLower(strings.TrimSpace(src.GGUF.KeyValue("general.architecture").String()))
	if arch == "" {
		return nil
	}

	for _, migrator := range migratorsByArchitecture[arch] {
		if migrator.NeedsMigration(src) {
			return migrator
		}
	}
	return nil
}

func sourceTensorHasPrefix(src *SourceModel, prefix string) bool {
	for _, tensor := range src.GGUF.TensorInfos() {
		if strings.HasPrefix(tensor.Name, prefix) {
			return true
		}
	}
	return false
}

func sourceTensorExists(src *SourceModel, name string) bool {
	return src.GGUF.TensorInfo(name).Valid()
}

func sourceTensorShape(src *SourceModel, name string) ([]uint64, bool) {
	info := src.GGUF.TensorInfo(name)
	return info.Shape, info.Valid()
}

func rawGGUFKeyExists(g *gguf.File, key string) bool {
	return rawGGUFKeyValue(g, key).Valid()
}

func rawGGUFKeyValue(g *gguf.File, key string) gguf.KeyValue {
	for _, keyValue := range g.KeyValues() {
		if keyValue.Key == key && keyValue.Valid() {
			return keyValue
		}
	}
	return gguf.KeyValue{}
}

func migrateToManifestReference(migrator Migrator, src *SourceModel) (_ manifest.Manifest, err error) {
	required := requiredBytesFromSource(src)
	available, err := availableSpaceForPath(filepath.Dir(src.GGUFPath))
	if err != nil {
		return manifest.Manifest{}, err
	}
	if available < required {
		slog.Warn("cannot convert legacy model due to disk headroom",
			"model", src.Source.DisplayShortest(),
			"available_bytes", available,
			"required_bytes", required,
		)
		return manifest.Manifest{}, errInsufficientSpace
	}

	start := time.Now()
	slog.Info("starting local compat GGUF migration",
		"model", src.Source.DisplayShortest(),
		"required_bytes", required,
	)

	result, err := migrator.Migrate(src)
	if err != nil {
		return manifest.Manifest{}, err
	}

	child, err := convertedManifest(src, result)
	if err != nil {
		return manifest.Manifest{}, err
	}

	// From here on the converted blobs exist in the store; clean them up on
	// any failure so an aborted migration does not strand a multi-GB blob
	// until the next startup prune.
	var childDigest string
	defer func() {
		if err != nil {
			removeConvertedChildBlobs(child, childDigest)
		}
	}()

	data, err := json.Marshal(child)
	if err != nil {
		return manifest.Manifest{}, err
	}
	childDigest, err = manifest.WriteManifestBlob(data)
	if err != nil {
		return manifest.Manifest{}, err
	}
	ref, err := manifest.NewManifestReference(childDigest, manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	if err != nil {
		return manifest.Manifest{}, err
	}
	runner, err := RunnerForManifest(src.Source, child)
	if err != nil {
		return manifest.Manifest{}, err
	}
	if runner != manifest.RunnerLlamaCPP {
		return manifest.Manifest{}, errors.New("converted GGUF still requires compatibility migration")
	}
	slog.Debug("wrote converted GGUF",
		"model", src.Source.DisplayShortest(),
		"duration", time.Since(start),
	)

	return ref, nil
}
