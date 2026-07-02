package compatmigrate

import (
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

const compatMigrationHeadroom = 512 << 20

type Migrator interface {
	NeedsMigration(*SourceModel) bool
	Migrate(*SourceModel) (*Result, error)
}

type SourceModel struct {
	Source         model.Name
	Target         model.Name
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

	src, err := loadSourceModelFromManifest(name, name, source)
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
			return false, nil
		}
		return false, err
	}
	refs = append(refs, convertedRef)
	return writeCompatibilityManifestList(name, source, refs)
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
		if err := fillManifestMetadata(&child); err != nil {
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
				continue
			}
			refs = append(refs, child)
			continue
		}

		if isRunnerFormat(resolved, manifest.RunnerLlamaCPP, manifest.FormatGGUF) && manifestBlobsExist(resolved) {
			return nil, nil, true, nil
		}
		if isRunnerFormat(resolved, manifest.RunnerLlamaCPP, manifest.FormatGGUF) {
			continue
		}

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
	for _, tensor := range src.GGUF.TensorInfos() {
		if tensor.Name == name {
			return true
		}
	}
	return false
}

func sourceTensorShape(src *SourceModel, name string) ([]uint64, bool) {
	for _, tensor := range src.GGUF.TensorInfos() {
		if tensor.Name == name {
			return tensor.Shape, true
		}
	}
	return nil, false
}

func rawGGUFKeyExists(g *gguf.File, key string) bool {
	for _, keyValue := range g.KeyValues() {
		if keyValue.Key == key && keyValue.Valid() {
			return true
		}
	}
	return false
}

func migrateToManifestReference(migrator Migrator, src *SourceModel) (manifest.Manifest, error) {
	required := requiredBytesFromSource(src)
	available, err := availableSpaceForPath(filepath.Dir(src.GGUFPath))
	if err != nil {
		return manifest.Manifest{}, err
	}
	if available < required {
		slog.Debug("skipping local compat migration due to disk headroom",
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
		if errors.Is(err, errUnsupportedFamily) || errors.Is(err, errInsufficientSpace) {
			return manifest.Manifest{}, err
		}
		return manifest.Manifest{}, err
	}

	child, err := convertedManifest(src, result)
	if err != nil {
		return manifest.Manifest{}, err
	}

	data, err := json.Marshal(child)
	if err != nil {
		return manifest.Manifest{}, err
	}
	digest, err := manifest.WriteManifestBlob(data)
	if err != nil {
		return manifest.Manifest{}, err
	}
	ref, err := manifest.NewManifestReference(digest, manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	if err != nil {
		return manifest.Manifest{}, err
	}
	if err := writeConvertedLegacyShadow(digest, data); err != nil {
		return manifest.Manifest{}, err
	}

	slog.Info("completed local compat GGUF migration",
		"model", src.Source.DisplayShortest(),
		"duration", time.Since(start),
	)

	return ref, nil
}
