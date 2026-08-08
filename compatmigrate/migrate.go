package compatmigrate

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/fs/ggml"
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
	ModelKV           ggml.KV
	ModelTensors      []*ggml.Tensor
	ProjectorKV       ggml.KV
	ProjectorTensors  []*ggml.Tensor
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
	"qwen35":          {qwen35Migrator{}},
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

func StartLocalCompatibilityMigration(name model.Name) bool {
	if !name.IsFullyQualified() {
		return false
	}
	if !hasCompatibilityMigrators() {
		return false
	}

	key := name.String()
	if _, loaded := migrationInFlight.LoadOrStore(key, struct{}{}); loaded {
		return false
	}

	go func() {
		defer migrationInFlight.Delete(key)

		migrated, err := EnsureLocalCompatibilityMigration(name)
		switch {
		case err != nil:
			slog.Warn("local compatibility migration failed",
				"model", name.DisplayShortest(),
				"error", err,
			)
		case migrated:
			slog.Debug("local compatibility migration completed",
				"model", name.DisplayShortest(),
			)
		}
	}()

	return true
}

func hasCompatibilityMigrators() bool {
	for _, migrators := range migratorsByArchitecture {
		if len(migrators) > 0 {
			return true
		}
	}
	return false
}

func EnsureLocalCompatibilityMigration(name model.Name) (bool, error) {
	if !name.IsFullyQualified() {
		return false, model.Unqualified(name)
	}

	unlock := lockCompatibilityMigration(name)
	defer unlock()

	data, err := manifest.ReadManifestData(name)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	} else if err != nil {
		return false, err
	}

	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return false, err
	}

	source, refs, done, err := migrationSourceFromManifest(&parent)
	if err != nil || done || source == nil {
		return done, err
	}

	if mediaType, ok := unsupportedSourceLayer(source); ok {
		// copyAncillaryLayers cannot carry these layers into the converted
		// child, and the converted child is preferred at load time — migrating
		// would silently change model behavior (e.g. run a LoRA model without
		// its adapter).
		slog.Info("skipping local compatibility migration for unsupported source layer",
			"model", name.DisplayShortest(),
			"media_type", mediaType,
		)
		return false, nil
	}

	src, err := loadSourceModelFromManifest(name, source)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	defer src.Close()

	migrator := compatibilityMigratorForSource(src)
	if migrator == nil {
		return false, nil
	}

	convertedRef, err := migrateToManifestReference(migrator, src)
	if err != nil {
		if errors.Is(err, errUnsupportedFamily) || errors.Is(err, errInsufficientSpace) {
			slog.Info("skipping local compatibility migration",
				"model", name.DisplayShortest(),
				"reason", err,
			)
			return false, nil
		}
		return false, err
	}

	// The conversion can take minutes on a large model. If the named manifest
	// was removed or replaced while it ran (ollama rm / pull / create), writing
	// now would resurrect or revert it. The per-name lock is held for this
	// whole function, so re-checking the raw manifest data closes the race.
	if current, readErr := manifest.ReadManifestData(name); readErr != nil || !bytes.Equal(current, data) {
		slog.Info("discarding local compatibility migration; model changed during conversion",
			"model", name.DisplayShortest(),
		)
		removeConvertedReference(convertedRef)
		return false, nil
	}

	refs = append(refs, convertedRef)
	return writeCompatibilityManifestList(name, source, refs)
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

func lockCompatibilityMigration(name model.Name) func() {
	value, _ := migrationLocks.LoadOrStore(name.String(), &sync.Mutex{})
	mu := value.(*sync.Mutex)
	mu.Lock()
	return mu.Unlock
}

func migrationSourceFromManifest(parent *manifest.Manifest) (*manifest.Manifest, []manifest.Manifest, bool, error) {
	if parent.MediaType != manifest.MediaTypeManifestList {
		child := *parent
		if err := manifest.FillMetadata(&child); err != nil {
			return nil, nil, false, err
		}
		if isRunnerFormat(&child, manifest.RunnerLlamaCPP, manifest.FormatGGUF) && manifestBlobsExist(&child) {
			return nil, nil, true, nil
		}
		if !isRunnerFormat(&child, manifest.RunnerGGML, manifest.FormatGGUF) {
			return nil, nil, false, nil
		}
		ref, err := manifestReferenceForChild(&child)
		if err != nil {
			return nil, nil, false, err
		}
		return &child, []manifest.Manifest{ref}, false, nil
	}

	refs := make([]manifest.Manifest, 0, len(parent.Manifests)+1)
	var source *manifest.Manifest
	for _, child := range parent.Manifests {
		if child.MediaType == manifest.MediaTypeManifestList {
			return nil, nil, false, errors.New("nested manifest lists are not supported")
		}

		resolved, err := resolveChildManifest(child)
		if err != nil {
			if isRunnerFormat(&child, manifest.RunnerLlamaCPP, manifest.FormatGGUF) {
				if !errors.Is(err, os.ErrNotExist) {
					return nil, nil, false, err
				}
				// Drop llamacpp children whose blobs are gone so the
				// migration regenerates them.
				continue
			}
			// Preserve foreign-runner children verbatim; their blobs may
			// legitimately live elsewhere (e.g. not pulled for this platform).
			slog.Warn("keeping unresolvable manifest child during migration scan",
				"runner", child.Runner,
				"digest", child.Digest,
				"error", err,
			)
			refs = append(refs, child)
			continue
		}

		if isRunnerFormat(resolved, manifest.RunnerLlamaCPP, manifest.FormatGGUF) && manifestBlobsExist(resolved) {
			return nil, nil, true, nil
		}
		if isRunnerFormat(resolved, manifest.RunnerLlamaCPP, manifest.FormatGGUF) {
			// Broken llamacpp child (manifest resolves but blobs are missing):
			// drop it so this migration writes a fresh replacement.
			continue
		}

		// The first GGML child with complete blobs is the conversion source.
		if source == nil && isRunnerFormat(resolved, manifest.RunnerGGML, manifest.FormatGGUF) && manifestBlobsExist(resolved) {
			sourceCopy := *resolved
			source = &sourceCopy
		}

		ref, err := manifestReferenceForChild(resolved)
		if err != nil {
			return nil, nil, false, err
		}
		refs = append(refs, ref)
	}
	return source, refs, false, nil
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
		slog.Info("skipping local compat migration due to disk headroom",
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
	if err = writeConvertedLegacyShadow(childDigest, data); err != nil {
		return manifest.Manifest{}, err
	}

	slog.Info("completed local compat GGUF migration",
		"model", src.Source.DisplayShortest(),
		"duration", time.Since(start),
	)

	return ref, nil
}
