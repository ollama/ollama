package server

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
	"github.com/ollama/ollama/x/transfer"
)

// Blobs newer than this may belong to another process that has not written its
// manifest yet. They become eligible for the normal mark-and-sweep pass later.
const layerPruneGracePeriod = time.Hour

var (
	errCapabilities         = errors.New("does not support")
	errCapabilityCompletion = errors.New("completion")
	errCapabilityTools      = errors.New("tools")
	errCapabilityInsert     = errors.New("insert")
	errCapabilityVision     = errors.New("vision")
	errCapabilityAudio      = errors.New("audio")
	errCapabilityEmbedding  = errors.New("embedding")
	errCapabilityThinking   = errors.New("thinking")
	errCapabilityImage      = errors.New("image generation")
	errInsecureProtocol     = errors.New("insecure protocol http")
)

type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

type Model struct {
	Name               string `json:"name"`
	Config             model.ConfigV2
	ShortName          string
	ModelPath          string
	ModelShardPaths    []string
	DraftPath          string
	DraftShardPaths    []string
	ParentModel        string
	HasChatTemplate    bool
	HasGoTemplate      bool
	PreferChatTemplate bool // set when GGUF chat_template should take precedence over Go TEMPLATE
	AdapterPaths       []string
	ProjectorPaths     []string
	System             string
	License            []string
	Digest             string
	Options            map[string]any
	GenerationDefaults model.GenerationDefaults
	Messages           []api.Message

	Template       *template.Template
	templateDigest string

	// Metadata of the model blob and of each projector, read from their
	// metadata files when the model is loaded.
	metadata          ggufMetadata
	projectorMetadata []ggufMetadata
}

func (m *Model) IsMLX() bool {
	return m.Config.ModelFormat == "safetensors"
}

func (m *Model) isGGUF() bool {
	return m.Config.ModelFormat == "" || m.Config.ModelFormat == "gguf"
}

func (m *Model) modelPaths() []string {
	if m == nil || m.ModelPath == "" {
		return nil
	}
	paths := make([]string, 1, len(m.ModelShardPaths)+1)
	paths[0] = m.ModelPath
	return append(paths, m.ModelShardPaths...)
}

func generationDefaultsFromMetadata(md ggufMetadata) model.GenerationDefaults {
	return model.ParseGGUFGenerationDefaults(
		func(key string) (int64, bool) {
			n, ok := md.number(key)
			if !ok {
				return 0, false
			}
			if value, err := n.Int64(); err == nil {
				return value, true
			}
			value, err := n.Float64()
			if err != nil {
				return 0, false
			}
			return int64(value), true
		},
		func(key string) (float64, bool) {
			n, ok := md.number(key)
			if !ok {
				return 0, false
			}
			value, err := n.Float64()
			return value, err == nil
		},
	)
}

func appendCapability(capabilities []model.Capability, capability model.Capability) []model.Capability {
	if slices.Contains(capabilities, capability) {
		return capabilities
	}
	return append(capabilities, capability)
}

type templateCapabilitySource int

const (
	templateCapabilitySelected templateCapabilitySource = iota
	templateCapabilityGo
	templateCapabilityChat
)

// Capabilities returns the capabilities that the model supports
func (m *Model) Capabilities() []model.Capability {
	capabilities := m.capabilitiesForTemplate(templateCapabilitySelected)
	if len(capabilities) == 0 {
		slog.Warn("unknown capabilities for model", "model", m.Name)
	}

	return capabilities
}

func (m *Model) capabilitiesForTemplate(source templateCapabilitySource) []model.Capability {
	capabilities := []model.Capability{}
	var modelArch string

	capabilities = m.configCapabilities(capabilities)
	capabilities, modelArch = m.ggufCapabilities(capabilities, source)
	capabilities = m.projectorCapabilities(capabilities)
	capabilities = m.templateCapabilities(capabilities, source)
	capabilities = m.parserCapabilities(capabilities)
	capabilities = m.modelFamilyCapabilities(capabilities)
	capabilities = m.filterUnsupportedCapabilities(capabilities, modelArch)

	return capabilities
}

func (m *Model) configCapabilities(capabilities []model.Capability) []model.Capability {
	for _, c := range m.Config.Capabilities {
		capabilities = appendCapability(capabilities, model.Capability(c))
	}
	return capabilities
}

func (m *Model) ggufCapabilities(capabilities []model.Capability, source templateCapabilitySource) ([]model.Capability, string) {
	if m.ModelPath == "" || !m.isGGUF() {
		return capabilities, ""
	}

	switch source {
	case templateCapabilitySelected:
		if !usesOllamaRenderedChat(m) {
			capabilities = chatTemplateCapabilities(capabilities, m.metadata.String("tokenizer.chat_template"))
		}
	case templateCapabilityChat:
		capabilities = chatTemplateCapabilities(capabilities, m.metadata.String("tokenizer.chat_template"))
	}
	if m.metadata.Valid("pooling_type") {
		capabilities = appendCapability(capabilities, model.CapabilityEmbedding)
	} else {
		// If no embedding is specified, we assume the model supports completion.
		capabilities = appendCapability(capabilities, model.CapabilityCompletion)
	}
	if m.metadata.Valid("vision.block_count") {
		capabilities = appendCapability(capabilities, model.CapabilityVision)
	}
	if m.metadata.Valid("audio.block_count") {
		capabilities = appendCapability(capabilities, model.CapabilityAudio)
	}

	return capabilities, m.metadata.String("general.architecture")
}

func chatTemplateCapabilities(capabilities []model.Capability, chatTemplate string) []model.Capability {
	if chatTemplate == "" {
		return capabilities
	}

	if chatTemplateHasToolSupport(chatTemplate) {
		capabilities = appendCapability(capabilities, model.CapabilityTools)
	}
	if thinking.TemplateSupportsThinking(chatTemplate) {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func chatTemplateHasToolSupport(chatTemplate string) bool {
	return strings.Contains(chatTemplate, "tools") || strings.Contains(chatTemplate, "tool_call")
}

func chatTemplateHasToolRoundTrip(chatTemplate string) bool {
	if !chatTemplateHasToolSupport(chatTemplate) {
		return false
	}

	toolCalls := strings.Contains(chatTemplate, "tool_calls") || strings.Contains(chatTemplate, "assistant_tool_call")
	return toolCalls && (strings.Contains(chatTemplate, "tool_response") ||
		strings.Contains(chatTemplate, "tool_results") ||
		strings.Contains(chatTemplate, "role'] == 'tool'") ||
		strings.Contains(chatTemplate, `role'] == "tool"`) ||
		strings.Contains(chatTemplate, `role"] == 'tool'`) ||
		strings.Contains(chatTemplate, `role"] == "tool"`) ||
		strings.Contains(chatTemplate, `message.role == 'tool'`) ||
		strings.Contains(chatTemplate, `message.role == "tool"`) ||
		strings.Contains(chatTemplate, "ipython"))
}

func goTemplateCapabilities(t *template.Template) []model.Capability {
	if t == nil {
		return nil
	}

	v, err := t.Vars()
	if err != nil {
		slog.Warn("model template contains errors", "error", err)
		return nil
	}

	var capabilities []model.Capability
	if slices.Contains(v, "tools") {
		capabilities = appendCapability(capabilities, model.CapabilityTools)
	}
	if slices.Contains(v, "suffix") {
		capabilities = appendCapability(capabilities, model.CapabilityInsert)
	}

	openingTag, closingTag := thinking.InferTags(t.Template)
	if openingTag != "" && closingTag != "" {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func goTemplateHasToolRoundTrip(t *template.Template) bool {
	if t == nil {
		return false
	}

	v, err := t.Vars()
	if err != nil || !slices.Contains(v, "tools") || !slices.Contains(v, "toolcalls") {
		return false
	}

	raw := t.String()
	return strings.Contains(raw, `eq .Role "tool"`) ||
		strings.Contains(raw, "tool_response") ||
		strings.Contains(raw, "TOOL_RESULTS")
}

func hasMoreCapabilities(candidate, current []model.Capability) bool {
	return len(candidate) > len(current)
}

func sameCapabilities(candidate, current []model.Capability) bool {
	if len(candidate) != len(current) {
		return false
	}
	for _, c := range candidate {
		if !slices.Contains(current, c) {
			return false
		}
	}
	return true
}

func shouldPreferChatTemplate(chatTemplate string, chatTemplateCaps []model.Capability, goTemplate *template.Template, goTemplateCaps []model.Capability) bool {
	if hasMoreCapabilities(chatTemplateCaps, goTemplateCaps) {
		return !goTemplateHasToolRoundTrip(goTemplate) || chatTemplateHasToolRoundTrip(chatTemplate)
	}

	if !sameCapabilities(chatTemplateCaps, goTemplateCaps) ||
		!slices.Contains(chatTemplateCaps, model.CapabilityTools) ||
		!slices.Contains(goTemplateCaps, model.CapabilityTools) {
		return false
	}

	return chatTemplateHasToolRoundTrip(chatTemplate) && !goTemplateHasToolRoundTrip(goTemplate)
}

func goTemplateEnvSet() bool {
	return envconfig.GoTemplate(true) == envconfig.GoTemplate(false)
}

func capabilityNames(capabilities []model.Capability) []string {
	names := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		names = append(names, string(capability))
	}

	return names
}

func selectedTemplateSource(m *Model, usesHarmony bool) string {
	switch {
	case m.Config.Renderer != "" && m.Config.Parser != "":
		return "renderer_parser"
	case m.Config.Renderer != "":
		return "renderer"
	case m.Config.Parser != "":
		return "parser"
	case usesHarmony:
		return "harmony"
	case shouldUseGoTemplate(m):
		return "go_template"
	case m.HasChatTemplate:
		return "gguf_chat_template"
	default:
		return "none"
	}
}

func capabilityLogValue(present bool, capabilities []model.Capability) any {
	if !present {
		return "null"
	}

	return capabilityNames(capabilities)
}

func (m *Model) templateSelectionCapabilities(usesHarmony bool) (goTemplate, chatTemplate, harmony, rendererParser []model.Capability) {
	if m.HasGoTemplate {
		goTemplate = m.capabilitiesForTemplate(templateCapabilityGo)
	}
	if m.HasChatTemplate {
		chatTemplate = m.capabilitiesForTemplate(templateCapabilityChat)
	}
	if usesHarmony {
		harmony = m.capabilitiesForTemplate(templateCapabilitySelected)
	}
	if m.Config.Renderer != "" || m.Config.Parser != "" {
		rendererParser = m.capabilitiesForTemplate(templateCapabilitySelected)
	}

	return goTemplate, chatTemplate, harmony, rendererParser
}

func logTemplateSelection(m *Model) {
	usesHarmony := m.Template != nil && shouldUseHarmony(m)
	goTemplateCapabilities, chatTemplateCapabilities, harmonyCapabilities, rendererParserCapabilities := m.templateSelectionCapabilities(usesHarmony)

	slog.Info("template selection",
		"model", m.Name,
		"selected", selectedTemplateSource(m, usesHarmony),
		"renderer", m.Config.Renderer,
		"parser", m.Config.Parser,
		"go_template", capabilityLogValue(m.HasGoTemplate, goTemplateCapabilities),
		"chat_template", capabilityLogValue(m.HasChatTemplate, chatTemplateCapabilities),
		"harmony", capabilityLogValue(usesHarmony, harmonyCapabilities),
		"renderer_parser", capabilityLogValue(m.Config.Renderer != "" || m.Config.Parser != "", rendererParserCapabilities),
	)
}

func (m *Model) projectorCapabilities(capabilities []model.Capability) []model.Capability {
	if len(m.ProjectorPaths) == 0 {
		return capabilities
	}

	capabilities = appendCapability(capabilities, model.CapabilityVision)
	for _, md := range m.projectorMetadata {
		if projectorHasAudio(md) && !projectorSuppressesAudioCapability(md) {
			capabilities = appendCapability(capabilities, model.CapabilityAudio)
		}
	}

	return capabilities
}

func (m *Model) templateCapabilities(capabilities []model.Capability, source templateCapabilitySource) []model.Capability {
	switch source {
	case templateCapabilitySelected:
		if m.HasGoTemplate && !shouldUseGoTemplate(m) {
			return capabilities
		}
	case templateCapabilityGo:
		if !m.HasGoTemplate {
			return capabilities
		}
	case templateCapabilityChat:
		return capabilities
	}

	for _, capability := range goTemplateCapabilities(m.Template) {
		capabilities = appendCapability(capabilities, capability)
	}

	return capabilities
}

func (m *Model) parserCapabilities(capabilities []model.Capability) []model.Capability {
	builtinParser := parsers.ParserForName(m.Config.Parser)
	if builtinParser == nil {
		return capabilities
	}

	if builtinParser.HasToolSupport() {
		capabilities = appendCapability(capabilities, model.CapabilityTools)
	}
	if builtinParser.HasThinkingSupport() {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func (m *Model) modelFamilyCapabilities(capabilities []model.Capability) []model.Capability {
	isGptoss := slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily)
	if isGptoss {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func (m *Model) filterUnsupportedCapabilities(capabilities []model.Capability, modelArch string) []model.Capability {
	if suppressAudioCapability(m, modelArch) {
		capabilities = slices.DeleteFunc(capabilities, func(c model.Capability) bool {
			return c == model.CapabilityAudio
		})
	}

	return capabilities
}

func suppressAudioCapability(m *Model, arch string) bool {
	if m.Config.ModelFormat == "safetensors" && m.Config.Renderer == "glimmer" {
		return true
	}
	if isNemotronSafetensors(m) {
		return true
	}

	if arch == "nemotron_h_omni" ||
		m.Config.ModelFamily == "nemotron_h_omni" ||
		slices.Contains(m.Config.ModelFamilies, "nemotron_h_omni") {
		// TODO: expose Nemotron3 audio once llama.cpp can skip or load the audio projector safely.
		return true
	}

	return false
}

func isNemotronSafetensors(m *Model) bool {
	return isNemotronSafetensorsConfig(m.Config)
}

func isNemotronSafetensorsConfig(cfg model.ConfigV2) bool {
	return cfg.ModelFormat == "safetensors" &&
		(cfg.Parser == "nemotron-3-nano" ||
			cfg.Renderer == "nemotron-3-nano" ||
			cfg.Parser == "nemotron-3.5-nano" ||
			cfg.Renderer == "nemotron-3.5-nano" ||
			cfg.ModelFamily == "nemotron_h_omni" ||
			slices.Contains(cfg.ModelFamilies, "nemotron_h_omni"))
}

func projectorHasAudio(md ggufMetadata) bool {
	// read directly: Keys reports qualified keys, the accessors qualify theirs
	for _, key := range md.Keys() {
		if key == "has_audio_encoder" || strings.HasSuffix(key, ".has_audio_encoder") {
			if b, ok := md.KV[key].(bool); ok && b {
				return true
			}
		}
	}

	return false
}

func projectorSuppressesAudioCapability(md ggufMetadata) bool {
	switch md.String("vision.projector_type") {
	case "gemma3nv":
		return true
	}

	return false
}

// CheckCapabilities checks if the model has the specified capabilities returning an error describing
// any missing or unknown capabilities
func (m *Model) CheckCapabilities(want ...model.Capability) error {
	available := m.Capabilities()
	var errs []error

	// Map capabilities to their corresponding error
	capToErr := map[model.Capability]error{
		model.CapabilityCompletion: errCapabilityCompletion,
		model.CapabilityTools:      errCapabilityTools,
		model.CapabilityInsert:     errCapabilityInsert,
		model.CapabilityVision:     errCapabilityVision,
		model.CapabilityAudio:      errCapabilityAudio,
		model.CapabilityEmbedding:  errCapabilityEmbedding,
		model.CapabilityThinking:   errCapabilityThinking,
		model.CapabilityImage:      errCapabilityImage,
	}

	for _, cap := range want {
		err, ok := capToErr[cap]
		if !ok {
			slog.Error("unknown capability", "capability", cap)
			return fmt.Errorf("unknown capability: %s", cap)
		}

		if !slices.Contains(available, cap) {
			errs = append(errs, err)
		}
	}

	var err error
	if len(errs) > 0 {
		err = fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))
	}

	if slices.Contains(errs, errCapabilityThinking) {
		if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
			// append a message to the existing error
			return fmt.Errorf("%w. Pull the model again to get the latest version with full thinking support", err)
		}
	}

	return err
}

func (m *Model) String() string {
	var modelfile parser.Modelfile

	modelfile.Commands = append(modelfile.Commands, parser.Command{
		Name: "model",
		Args: m.ModelPath,
	})

	for _, adapter := range m.AdapterPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "adapter",
			Args: adapter,
		})
	}

	if m.DraftPath != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "draft",
			Args: m.DraftPath,
		})
	}

	for _, projector := range m.ProjectorPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "model",
			Args: projector,
		})
	}

	if m.Template != nil {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "template",
			Args: m.Template.String(),
		})
	}

	if m.System != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "system",
			Args: m.System,
		})
	}

	if m.Config.Renderer != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "renderer",
			Args: m.Config.Renderer,
		})
	}

	if m.Config.Parser != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "parser",
			Args: m.Config.Parser,
		})
	}

	for k, v := range m.Options {
		switch v := v.(type) {
		case []any:
			for _, s := range v {
				modelfile.Commands = append(modelfile.Commands, parser.Command{
					Name: k,
					Args: fmt.Sprintf("%v", s),
				})
			}
		default:
			modelfile.Commands = append(modelfile.Commands, parser.Command{
				Name: k,
				Args: fmt.Sprintf("%v", v),
			})
		}
	}

	for _, license := range m.License {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "license",
			Args: license,
		})
	}

	for _, msg := range m.Messages {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "message",
			Args: fmt.Sprintf("%s: %s", msg.Role, msg.Content),
		})
	}

	return modelfile.String()
}

func GetManifest(n model.Name) (*manifest.Manifest, string, error) {
	fp := n.Filepath()

	f, err := os.Open(fp)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()

	sha256sum := sha256.New()

	var manifestFile manifest.Manifest
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&manifestFile); err != nil {
		return nil, "", err
	}

	return &manifestFile, hex.EncodeToString(sha256sum.Sum(nil)), nil
}

// ggufSplitInfo reports the zero-based index and total count of a split GGUF,
// if the file carries split metadata. These keys are stored unprefixed by
// llama.cpp's gguf-split, but gguf.File.KeyValue implicitly prefixes
// non-general/tokenizer keys with the architecture, so scan the raw key
// values instead.
func ggufSplitInfo(f *gguf.File) (no, count uint64, ok bool) {
	for _, kv := range f.KeyValues() {
		switch kv.Key {
		case "split.no":
			no, ok = kv.Uint(), true
		case "split.count":
			count = kv.Uint()
		}
	}
	return no, count, ok
}

// splitGGUFLinkDir materializes a directory of hard links to the shard blobs,
// named with llama.cpp's `-00001-of-0000N.gguf` convention, and returns the
// path of the first shard.
//
// llama.cpp is handed only the first shard and derives the rest by rewriting
// the index in its filename. Ollama stores blobs content-addressed as
// sha256-<hex>, which carries no such pattern, so the names have to be
// recreated before the model can be loaded.
//
// The links are reused across loads: they are cheap (no data is copied, both
// names refer to the same inode) and stable, so a model that is loaded,
// evicted, and loaded again does not rebuild them.
func splitGGUFLinkDir(shards []manifest.Layer) (string, error) {
	if len(shards) == 0 {
		return "", errors.New("no split GGUF shards")
	}

	blobs, err := manifest.BlobsPath("")
	if err != nil {
		return "", err
	}

	// Key the directory on the first shard's digest so distinct models (and
	// distinct quantizations of the same model) never collide.
	dir := filepath.Join(blobs, "splits", strings.TrimPrefix(shards[0].Digest, "sha256:"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	base := splitGGUFBaseName(shards[0].Name)
	first := ""
	for i, shard := range shards {
		blobPath, err := manifest.BlobsPath(shard.Digest)
		if err != nil {
			return "", err
		}

		linkPath := filepath.Join(dir, parser.ShardName(base, i+1, len(shards)))
		if i == 0 {
			first = linkPath
		}

		if _, err := os.Stat(linkPath); err == nil {
			continue
		}
		if err := os.Link(blobPath, linkPath); err != nil {
			// A cross-device or filesystem-level failure is not fatal on its
			// own, but without every shard linked llama.cpp cannot load the
			// model, so surface it.
			return "", fmt.Errorf("linking split GGUF shard %d: %w", i+1, err)
		}
	}

	return first, nil
}

// splitGGUFBaseName recovers the model-name prefix from a stored shard
// filename, falling back to a fixed name for manifests written before shard
// names were recorded. The prefix only has to be consistent across the set.
func splitGGUFBaseName(name string) string {
	if base, _, _, ok := parser.ParseShardName(path.Base(name)); ok {
		return base
	}
	return "model"
}

// resolveSplitGGUF inspects the model layers and, when they hold a split GGUF,
// returns the path llama.cpp should be given: the first shard, under a name
// from which it can derive the others. It returns "" when the layers are not
// a split set, in which case the caller keeps the plain blob path.
func resolveSplitGGUF(modelLayers []manifest.Layer) (string, error) {
	if len(modelLayers) < 2 {
		return "", nil
	}

	type shard struct {
		layer manifest.Layer
		no    uint64
	}

	var (
		shards []shard
		count  uint64
	)
	for _, layer := range modelLayers {
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return "", err
		}
		f, err := gguf.Open(blobPath)
		if err != nil {
			// Not readable as GGUF: not a split set we can reconstruct.
			return "", nil
		}
		no, total, ok := ggufSplitInfo(f)
		f.Close()
		if !ok {
			return "", nil
		}
		if count == 0 {
			count = total
		} else if count != total {
			return "", fmt.Errorf("inconsistent split GGUF counts: %d and %d", count, total)
		}
		shards = append(shards, shard{layer: layer, no: no})
	}

	if int(count) != len(shards) {
		return "", fmt.Errorf("split GGUF has %d shards, expected %d", len(shards), count)
	}

	slices.SortFunc(shards, func(a, b shard) int { return cmp.Compare(a.no, b.no) })
	for i, s := range shards {
		if s.no != uint64(i) {
			return "", fmt.Errorf("split GGUF is missing shard %d", i)
		}
	}

	ordered := make([]manifest.Layer, len(shards))
	for i, s := range shards {
		ordered[i] = s.layer
	}
	return splitGGUFLinkDir(ordered)
}

func GetModel(name string) (*Model, error) {
	n := model.ParseName(name)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Name:      n.String(),
		ShortName: n.DisplayShortest(),
		Digest:    mf.Digest(),
		Template:  template.DefaultTemplate,
	}

	if mf.Config.Digest != "" {
		filename, err := manifest.BlobsPath(mf.Config.Digest)
		if err != nil {
			return nil, err
		}

		configFile, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer configFile.Close()

		if err := json.NewDecoder(configFile).Decode(&m.Config); err != nil {
			return nil, err
		}
		m.GenerationDefaults = m.Config.GenerationDefaults
	}

	modelHasPooling := false
	ggufChatTemplate := ""
	modelLayerSeen := false
	var modelLayers []manifest.Layer
	for _, layer := range mf.Layers {
		// Nothing below reads a tensor layer, and resolving a path costs a
		// syscall each. Named rather than allowlisting the types below, so a new
		// layer type is slower here instead of silently unread.
		if layer.MediaType == manifest.MediaTypeImageTensor {
			continue
		}

		filename, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			// A model may be split across multiple GGUF layers (e.g. downloaded
			// as model-00001-of-00005.gguf, ...). llama.cpp must be pointed at
			// the first split (split.no == 0); it discovers and loads the rest
			// itself. Inspect each candidate's split metadata rather than
			// assuming manifest order, and don't let a later split overwrite
			// the first one we already selected.
			modelLayers = append(modelLayers, layer)
			useLayer := !modelLayerSeen
			if m.isGGUF() {
				md, err := readGGUFMetadata(layer.Digest)
				if err != nil {
					slog.Error("couldn't read model metadata", "error", err)
					break
				}
				if splitNo, ok := ggufMetadataSplitNo(md); ok {
					useLayer = splitNo == 0
				} else {
					useLayer = true
				}
				if useLayer {
					m.metadata = md
					ggufChatTemplate = md.String("tokenizer.chat_template")
					m.HasChatTemplate = ggufChatTemplate != ""
					modelHasPooling = md.Valid("pooling_type")
					m.GenerationDefaults = generationDefaultsFromMetadata(md)
				}
			}
			if useLayer {
				m.ModelPath = filename
				m.ParentModel = layer.From
				modelLayerSeen = true
			} else {
				m.ModelShardPaths = append(m.ModelShardPaths, filename)
			}
		case manifest.MediaTypeImageDraft:
			if m.DraftPath == "" {
				m.DraftPath = filename
			} else {
				m.DraftShardPaths = append(m.DraftShardPaths, filename)
			}
		case "application/vnd.ollama.image.embed":
			// Deprecated in versions  > 0.1.2
			// TODO: remove this warning in a future version
			slog.Info("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			m.AdapterPaths = append(m.AdapterPaths, filename)
		case "application/vnd.ollama.image.projector":
			m.ProjectorPaths = append(m.ProjectorPaths, filename)
			if md, err := readGGUFMetadata(layer.Digest); err != nil {
				slog.Error("couldn't read projector metadata", "error", err)
			} else {
				m.projectorMetadata = append(m.projectorMetadata, md)
			}
		case "application/vnd.ollama.image.prompt",
			"application/vnd.ollama.image.template":
			m.HasGoTemplate = true
			m.templateDigest = layer.Digest
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.Template, err = template.Parse(string(bts))
			if err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			m.System = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// parse model options parameters into a map so that we can see which fields have been specified explicitly
			if err = json.NewDecoder(params).Decode(&m.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.messages":
			msgs, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer msgs.Close()

			if err = json.NewDecoder(msgs).Decode(&m.Messages); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			m.License = append(m.License, string(bts))
		}
	}

	// A split GGUF cannot be loaded from its blob path: llama.cpp derives the
	// remaining shards from the first shard's filename. Point ModelPath at a
	// correctly named link instead so the whole set resolves.
	if m.isGGUF() && len(modelLayers) > 1 {
		splitPath, err := resolveSplitGGUF(modelLayers)
		if err != nil {
			return nil, err
		}
		if splitPath != "" {
			m.ModelPath = splitPath
		}
	}

	ggufCaps := chatTemplateCapabilities(nil, ggufChatTemplate)
	goCaps := goTemplateCapabilities(m.Template)
	usesHarmony := m.Template != nil && shouldUseHarmony(m)
	if !goTemplateEnvSet() && m.HasGoTemplate && ggufChatTemplate != "" && m.Config.Renderer == "" && m.Config.Parser == "" && !usesHarmony && shouldPreferChatTemplate(ggufChatTemplate, ggufCaps, m.Template, goCaps) {
		m.PreferChatTemplate = true
	}

	if m.ModelPath != "" && m.isGGUF() && !modelHasPooling && !m.HasChatTemplate && (!m.HasGoTemplate || !envconfig.GoTemplate(true)) && m.Config.Renderer == "" && m.Config.Parser == "" && !usesHarmony {
		slog.Warn("model is missing tokenizer.chat_template and Go TEMPLATE support is unavailable; chat responses may be poorly formatted", "model", m.Name, "env", "OLLAMA_GO_TEMPLATE=1")
	}

	return m, nil
}

func CopyModel(src, dst model.Name) error {
	if !dst.IsFullyQualified() {
		return model.Unqualified(dst)
	}
	if !src.IsFullyQualified() {
		return model.Unqualified(src)
	}

	if src.Filepath() == dst.Filepath() {
		return nil
	}

	manifests, err := manifest.Path()
	if err != nil {
		return err
	}

	dstpath := filepath.Join(manifests, dst.Filepath())
	if err := os.MkdirAll(filepath.Dir(dstpath), 0o755); err != nil {
		return err
	}

	srcpath := filepath.Join(manifests, src.Filepath())
	srcfile, err := os.Open(srcpath)
	if err != nil {
		return err
	}
	defer srcfile.Close()

	dstfile, err := os.Create(dstpath)
	if err != nil {
		return err
	}
	defer dstfile.Close()

	_, err = io.Copy(dstfile, srcfile)
	return err
}

func deleteUnusedLayers(deleteMap map[string]struct{}) error {
	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	for _, manifest := range manifests {
		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := manifest.BlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if err := os.Remove(fp); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
			continue
		}
		// Split GGUF shards are also hard-linked under blobs/splits/. Removing
		// the blob alone would leave those links holding the data, so the
		// space would never be reclaimed.
		removeSplitGGUFLinkDir(k)
		removeGGUFMetadata(k)
	}

	return nil
}

// removeSplitGGUFLinkDir drops the link directory keyed on the given digest,
// if one exists. Only the first shard's digest names a directory; calling this
// for other digests is a no-op.
func removeSplitGGUFLinkDir(digest string) {
	blobs, err := manifest.BlobsPath("")
	if err != nil {
		return
	}

	dir := filepath.Join(blobs, "splits", strings.TrimPrefix(digest, "sha256:"))
	if err := os.RemoveAll(dir); err != nil {
		slog.Info(fmt.Sprintf("couldn't remove split GGUF links '%s': %v", dir, err))
	}
}

func PruneLayers() error {
	deleteMap := make(map[string]struct{})
	p, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't read dir '%s': %v", p, err))
		return err
	}

	for _, blob := range blobs {
		if blob.IsDir() {
			continue
		}

		info, err := blob.Info()
		if err != nil {
			slog.Error("couldn't stat blob", "blob", blob.Name(), "error", err)
			continue
		}
		if time.Since(info.ModTime()) < layerPruneGracePeriod {
			continue
		}

		name := blob.Name()
		name = strings.ReplaceAll(name, "-", ":")

		_, err = manifest.BlobsPath(name)
		if err != nil {
			if errors.Is(err, manifest.ErrInvalidDigestFormat) {
				// remove invalid blobs (e.g. partial downloads)
				if err := os.Remove(filepath.Join(p, blob.Name())); err != nil {
					slog.Error("couldn't remove blob", "blob", blob.Name(), "error", err)
				}
			}

			continue
		}

		deleteMap[name] = struct{}{}
	}

	slog.Info(fmt.Sprintf("total blobs: %d", len(deleteMap)))

	if err := deleteUnusedLayers(deleteMap); err != nil {
		slog.Error(fmt.Sprintf("couldn't remove unused layers: %v", err))
		return nil
	}
	pruneGGUFMetadata()

	slog.Info(fmt.Sprintf("total unused blobs removed: %d", len(deleteMap)))

	pruneSplitGGUFLinkDirs()

	return nil
}

// pruneSplitGGUFLinkDirs removes split link directories whose first shard is
// gone. Each directory is named for that shard's digest, so its absence means
// the model it belonged to is no longer present.
func pruneSplitGGUFLinkDirs() {
	blobs, err := manifest.BlobsPath("")
	if err != nil {
		return
	}

	root := filepath.Join(blobs, "splits")
	entries, err := os.ReadDir(root)
	if err != nil {
		return
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if _, err := os.Stat(filepath.Join(blobs, "sha256-"+entry.Name())); err == nil {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		if err := os.RemoveAll(dir); err != nil {
			slog.Info(fmt.Sprintf("couldn't remove split GGUF links '%s': %v", dir, err))
		}
	}
}

func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}

func PushModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	n := model.ParseName(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		// Read raw manifest JSON to preserve tensor metadata fields
		manifestPath, err := manifest.PathForName(n)
		if err != nil {
			return err
		}
		manifestJSON, err := os.ReadFile(manifestPath)
		if err != nil {
			return err
		}
		if err := pushWithTransfer(ctx, n, layers, manifestJSON, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	for _, layer := range layers {
		if err := uploadBlob(ctx, n, layer, regOpts, fn); err != nil {
			slog.Info(fmt.Sprintf("error uploading blob: %v", err))
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := n.BaseURL()
	requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	manifestJSON, err := json.Marshal(mf)
	if err != nil {
		return err
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func PullModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	n := model.ParseName(name)

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})
	existingMf, err := manifest.ParseNamedManifest(n)
	if errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		slog.Warn("pulling model with bad existing manifest", "name", name, "error", err)
	} else {
		for _, l := range existingMf.Layers {
			deleteMap[l.Digest] = struct{}{}
		}
		if existingMf.Config.Digest != "" {
			deleteMap[existingMf.Config.Digest] = struct{}{}
		}
	}

	if n.ProtocolScheme == "http" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	mf, manifestData, err := pullModelManifest(ctx, n, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}
	if hasTensorLayers(mf.Layers) {
		if err := mlx.CheckInit(); err != nil {
			slog.Debug("MLX is unavailable for safetensors model pull", "error", err)
			return errors.New("this model requires MLX support, but the MLX runtime is not available")
		}
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	// Use fast transfer for models with tensor layers (many small blobs)
	if hasTensorLayers(layers) {
		if err := pullWithTransfer(ctx, n, layers, manifestData, regOpts, fn); err != nil {
			return err
		}
		fn(api.ProgressResponse{Status: "success"})
		return nil
	}

	skipVerify := make(map[string]bool)
	isHF := isHuggingFaceRegistry(n.Host)

	for i, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			n:       n,
			digest:  layer.Digest,
			regOpts: regOpts,
			fn:      fn,
		})
		if err != nil {
			return err
		}

		// For HuggingFace downloads, replace the HF path with the real digest
		if isHF && strings.HasPrefix(layer.Digest, "hf:") {
			if realDigest, ok := hfDigestMap.Load(layer.Digest); ok {
				layer.Digest = realDigest.(string)
				layers[i].Digest = realDigest.(string)
				// Update the manifest layers
				for j := range mf.Layers {
					if strings.HasPrefix(mf.Layers[j].Digest, "hf:") {
						if rd, ok := hfDigestMap.Load(mf.Layers[j].Digest); ok {
							mf.Layers[j].Digest = rd.(string)
						}
					}
				}
			}
		}

		// If any download of a given digest was not a cache hit,
		// always verify it. Without this guard, a config entry
		// sharing a digest with a layer can overwrite the layer's
		// false (needs verification) with true (cache hit), since
		// the blob now exists on disk from the first download.
		if existing, ok := skipVerify[layer.Digest]; !ok {
			skipVerify[layer.Digest] = cacheHit
		} else {
			skipVerify[layer.Digest] = existing && cacheHit
		}
		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if skipVerify[layer.Digest] {
			continue
		}
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				fp, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					slog.Info(fmt.Sprintf("couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
				removeGGUFMetadata(layer.Digest)
			}
			return err
		}
	}

	for _, layer := range layers {
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, mf.Config.Digest)

	fn(api.ProgressResponse{Status: "writing manifest"})

	fp, err := manifest.PathForName(n)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	if isHF {
		manifestData, err = json.Marshal(mf)
		if err != nil {
			return fmt.Errorf("failed to marshal updated manifest: %w", err)
		}
	}

	err = os.WriteFile(fp, manifestData, 0o644)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't write to %s", fp))
		return err
	}

	slog.Debug("manifest written", "path", fp, "sha256", fmt.Sprintf("%x", sha256.Sum256(manifestData)), "size", len(manifestData))

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

// hasTensorLayers checks if any layer has tensor media type.
func hasTensorLayers(layers []manifest.Layer) bool {
	for _, layer := range layers {
		if layer.MediaType == manifest.MediaTypeImageTensor {
			return true
		}
	}
	return false
}

// pullWithTransfer uses the simplified x/transfer package for downloading blobs.
func pullWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, manifestData []byte, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
		}
	}

	destDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pulling model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	if err := transfer.Download(ctx, transfer.DownloadOptions{
		Blobs:             blobs,
		BaseURL:           baseURL,
		DestDir:           destDir,
		Repository:        n.DisplayNamespaceModel(),
		BodyConcurrency:   max(1, int(envconfig.MaxTransferStreams())),
		Progress:          progress,
		Token:             regOpts.Token,
		GetToken:          getToken,
		Logger:            slog.Default(),
		AllowPrivateHosts: regOpts != nil && regOpts.Insecure,
	}); err != nil {
		return err
	}

	// Write manifest
	fn(api.ProgressResponse{Status: "writing manifest"})

	fp, err := manifest.PathForName(n)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	if err := os.WriteFile(fp, manifestData, 0o644); err != nil {
		return err
	}

	slog.Debug("manifest written", "path", fp, "sha256", fmt.Sprintf("%x", sha256.Sum256(manifestData)), "size", len(manifestData))
	return nil
}

// pushWithTransfer uses the simplified x/transfer package for uploading blobs and manifest.
func pushWithTransfer(ctx context.Context, n model.Name, layers []manifest.Layer, manifestJSON []byte, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	blobs := make([]transfer.Blob, len(layers))
	for i, layer := range layers {
		blobs[i] = transfer.Blob{
			Digest: layer.Digest,
			Size:   layer.Size,
			From:   layer.From,
		}
	}

	srcDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	base := n.BaseURL()
	if base.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		base.Scheme = "http"
	}
	baseURL := base.String()

	var totalSize int64
	for _, blob := range blobs {
		totalSize += blob.Size
	}

	progress := func(completed, total int64) {
		fn(api.ProgressResponse{
			Status:    "pushing model",
			Digest:    "sha256:model",
			Total:     total,
			Completed: completed,
		})
	}

	getToken := func(ctx context.Context, challenge transfer.AuthChallenge) (string, error) {
		return getAuthorizationToken(ctx, registryChallenge{
			Realm:   challenge.Realm,
			Service: challenge.Service,
			Scope:   challenge.Scope,
		}, base.Host)
	}

	return transfer.Upload(ctx, transfer.UploadOptions{
		Blobs:             blobs,
		BaseURL:           baseURL,
		SrcDir:            srcDir,
		BodyConcurrency:   max(1, int(envconfig.MaxTransferStreams())),
		Progress:          progress,
		Token:             regOpts.Token,
		GetToken:          getToken,
		Logger:            slog.Default(),
		Manifest:          manifestJSON,
		ManifestRef:       n.Tag,
		Repository:        n.DisplayNamespaceModel(),
		AllowPrivateHosts: regOpts != nil && regOpts.Insecure,
	})
}

func pullModelManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, []byte, error) {
	// Check if this is a HuggingFace registry
	if isHuggingFaceRegistry(n.Host) {
		return pullHuggingFaceManifest(ctx, n, regOpts)
	}
	requestURL := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	var m manifest.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, nil, err
	}

	return &m, data, err
}

// isHuggingFaceRegistry checks if the registry is HuggingFace
func isHuggingFaceRegistry(registry string) bool {
	return registry == "hf.co" || registry == "huggingface.co"
}

// HFFileInfo represents a file in HuggingFace's file tree
type HFFileInfo struct {
	Type string `json:"type"`
	OID  string `json:"oid"`
	Size int64  `json:"size"`
	Path string `json:"path"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs,omitempty"`
}

// pullHuggingFaceManifest pulls a model manifest from HuggingFace
func pullHuggingFaceManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, []byte, error) {
	// For HuggingFace, the tag might be "main" or could include a subdirectory like "BF16"
	// We'll use "main" as the revision and the tag as the subdirectory filter
	revision := "main"
	subdirFilter := n.Tag

	// Query HuggingFace API for file tree (always use main revision, recursive)
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s?recursive=true", n.DisplayNamespaceModel(), revision)

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("creating HuggingFace API request: %w", err)
	}

	if regOpts != nil && regOpts.Token != "" {
		req.Header.Set("Authorization", "Bearer "+regOpts.Token)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("querying HuggingFace API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil, fmt.Errorf("model not found on HuggingFace")
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("HuggingFace API error (%d): %s", resp.StatusCode, string(body))
	}

	var files []HFFileInfo
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, nil, fmt.Errorf("decoding HuggingFace API response: %w", err)
	}

	// Find GGUF files matching the tag/subdirectory
	var ggufFiles []HFFileInfo
	subdirLower := strings.ToLower(subdirFilter)
	for _, file := range files {
		if file.Type == "file" && strings.HasSuffix(file.Path, ".gguf") {
			pathLower := strings.ToLower(file.Path)
			// Match if:
			// 1. Path starts with "tag/" (directory match)
			// 2. Filename is exactly "tag.gguf" or "anything-tag.gguf" or "tag-anything.gguf"
			//    But NOT "tag_anything.gguf" to avoid Q6_K matching Q6_K_XL
			if strings.HasPrefix(pathLower, subdirLower+"/") ||
				strings.Contains(pathLower, "/"+subdirLower+"/") ||
				strings.HasSuffix(pathLower, "-"+subdirLower+".gguf") ||
				strings.Contains(pathLower, "-"+subdirLower+"-") {
				ggufFiles = append(ggufFiles, file)
			}
		}
	}

	if len(ggufFiles) == 0 {
		return nil, nil, fmt.Errorf("no GGUF files found for tag %s", subdirFilter)
	}

	// Check if these are split GGUF files
	shardSets, singles := parser.GroupGGUFShards(extractPaths(ggufFiles))

	var mf manifest.Manifest
	mf.SchemaVersion = 2
	mf.MediaType = "application/vnd.docker.distribution.manifest.v2+json"

	// Handle split GGUF files
	if len(shardSets) > 0 {
		slog.Info("detected split GGUF files", "shards", len(shardSets[0].Shards))

		// Use the first (and should be only) shard set
		for _, shardPath := range shardSets[0].Shards {
			// Find the file info for this shard
			var fileInfo *HFFileInfo
			for i := range ggufFiles {
				if strings.HasSuffix(ggufFiles[i].Path, filepath.Base(shardPath)) {
					fileInfo = &ggufFiles[i]
					break
				}
			}

			if fileInfo == nil {
				return nil, nil, fmt.Errorf("shard file info not found: %s", shardPath)
			}

			// Create a layer for this shard. The original filename is kept so
			// the split set can be reconstructed at load time: llama.cpp
			// derives the remaining shards from the first shard's name, which
			// the content-addressed blob path does not preserve.
			layer := manifest.Layer{
				MediaType: "application/vnd.ollama.image.model",
				Size:      fileInfo.Size,
				Digest:    "", // Will be computed during download
				Name:      filepath.Base(fileInfo.Path),
			}

			// Store the HuggingFace download URL in the layer
			// We'll use the digest field temporarily to store the download path
			layer.Digest = fmt.Sprintf("hf:%s/%s/%s", n.DisplayNamespaceModel(), n.Tag, fileInfo.Path)

			mf.Layers = append(mf.Layers, layer)
		}
	} else if len(singles) > 0 {
		// Single GGUF file
		slog.Info("detected single GGUF file", "file", singles[0])

		var fileInfo *HFFileInfo
		for i := range ggufFiles {
			if strings.HasSuffix(ggufFiles[i].Path, filepath.Base(singles[0])) {
				fileInfo = &ggufFiles[i]
				break
			}
		}

		if fileInfo == nil {
			return nil, nil, fmt.Errorf("GGUF file info not found")
		}

		layer := manifest.Layer{
			MediaType: "application/vnd.ollama.image.model",
			Size:      fileInfo.Size,
			Digest:    fmt.Sprintf("hf:%s/%s/%s", n.DisplayNamespaceModel(), n.Tag, fileInfo.Path),
		}

		mf.Layers = append(mf.Layers, layer)
	}

	// Try to build a params layer from HuggingFace tokenizer/generation config
	params := fetchHFParams(ctx, n, regOpts)
	slog.Info("fetchHFParams result", "model", n.DisplayNamespaceModel(), "params", params)
	if len(params) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(params); err != nil {
			slog.Warn("failed to encode HuggingFace params", "err", err)
		} else if layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params"); err != nil {
			slog.Warn("failed to create HuggingFace params layer", "err", err)
		} else {
			slog.Info("adding params layer from HuggingFace config", "params", params)
			mf.Layers = append(mf.Layers, layer)
		}
	}

	mfJSON, err := json.Marshal(mf)
	if err != nil {
		return nil, nil, fmt.Errorf("marshaling HuggingFace manifest: %w", err)
	}

	return &mf, mfJSON, nil
}

// fetchHFParams fetches params from HuggingFace for a GGUF model.
// It first tries an Ollama-format "params" file in the repo (many GGUF repos include one),
// then falls back to extracting stop tokens and sampling params from generation_config.json
// and tokenizer_config.json, trying both the GGUF repo and the base model repo.
func fetchHFParams(ctx context.Context, n model.Name, regOpts *registryOptions) map[string]any {
	client := &http.Client{}

	get := func(u string) (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
		if err != nil {
			return nil, err
		}
		if regOpts != nil && regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, nil
		}
		return resp, nil
	}

	repo := n.DisplayNamespaceModel()

	// 1. Try Ollama-format "params" file — many GGUF repos ship this directly
	if resp, err := get(fmt.Sprintf("https://huggingface.co/%s/resolve/main/params", repo)); err == nil && resp != nil {
		defer resp.Body.Close()
		var p map[string]any
		if json.NewDecoder(resp.Body).Decode(&p) == nil && len(p) > 0 {
			slog.Debug("using HuggingFace params file", "repo", repo)
			return p
		}
	}

	// 2. Fall back: try generation_config.json and tokenizer_config.json
	// These live in the base model repo, not the GGUF repo — strip known suffixes
	parts := strings.SplitN(repo, "/", 2)
	baseRepo := repo
	if len(parts) == 2 {
		for _, suffix := range []string{"-GGUF", "-gguf"} {
			if trimmed := strings.TrimSuffix(parts[1], suffix); trimmed != parts[1] {
				baseRepo = parts[0] + "/" + trimmed
				break
			}
		}
	}

	fetchJSON := func(r, filename string) map[string]json.RawMessage {
		resp, err := get(fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", r, filename))
		if err != nil || resp == nil {
			return nil
		}
		defer resp.Body.Close()
		var m map[string]json.RawMessage
		if json.NewDecoder(resp.Body).Decode(&m) != nil {
			return nil
		}
		return m
	}

	params := make(map[string]any)

	// generation_config.json: sampling params + EOS token IDs
	var eosIDs []int
	if cfg := fetchJSON(baseRepo, "generation_config.json"); cfg != nil {
		if raw, ok := cfg["eos_token_id"]; ok {
			var single int
			var multi []int
			if json.Unmarshal(raw, &multi) == nil {
				eosIDs = multi
			} else if json.Unmarshal(raw, &single) == nil {
				eosIDs = []int{single}
			}
		}
		floatParam := func(key, ollamaKey string) {
			if raw, ok := cfg[key]; ok {
				var v float32
				if json.Unmarshal(raw, &v) == nil {
					params[ollamaKey] = v
				}
			}
		}
		intParam := func(key, ollamaKey string) {
			if raw, ok := cfg[key]; ok {
				var v int
				if json.Unmarshal(raw, &v) == nil {
					params[ollamaKey] = v
				}
			}
		}
		floatParam("temperature", "temperature")
		floatParam("top_p", "top_p")
		floatParam("repetition_penalty", "repeat_penalty")
		intParam("top_k", "top_k")
		intParam("max_new_tokens", "num_predict")
	}

	// tokenizer_config.json: stop token strings
	seen := make(map[string]bool)
	var stopTokens []string
	addStop := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			stopTokens = append(stopTokens, s)
		}
	}

	if cfg := fetchJSON(baseRepo, "tokenizer_config.json"); cfg != nil {
		if raw, ok := cfg["eos_token"]; ok {
			var s string
			var mm map[string]any
			if json.Unmarshal(raw, &s) == nil {
				addStop(s)
			} else if json.Unmarshal(raw, &mm) == nil {
				if content, ok := mm["content"].(string); ok {
					addStop(content)
				}
			}
		}
		if raw, ok := cfg["additional_special_tokens"]; ok {
			var tokens []string
			if json.Unmarshal(raw, &tokens) == nil {
				for _, t := range tokens {
					tl := strings.ToLower(t)
					if strings.Contains(tl, "end") || strings.Contains(tl, "eot") || strings.Contains(tl, "im_end") {
						addStop(t)
					}
				}
			}
		}
	}

	// If EOS IDs found but no string token yet, resolve via tokenizer.json
	if len(eosIDs) > 0 && len(stopTokens) == 0 {
		if tj := fetchJSON(baseRepo, "tokenizer.json"); tj != nil {
			type addedToken struct {
				ID      int    `json:"id"`
				Content string `json:"content"`
			}
			if raw, ok := tj["added_tokens"]; ok {
				var added []addedToken
				if json.Unmarshal(raw, &added) == nil {
					idToContent := make(map[int]string, len(added))
					for _, at := range added {
						idToContent[at.ID] = at.Content
					}
					for _, id := range eosIDs {
						addStop(idToContent[id])
					}
				}
			}
		}
	}

	if len(stopTokens) > 0 {
		params["stop"] = stopTokens
	}

	return params
}

// extractPaths extracts file paths from HFFileInfo slice
func extractPaths(files []HFFileInfo) []string {
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.Path
	}
	return paths
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

var errUnauthorized = errors.New("unauthorized: access denied")

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *registryOptions) (*http.Response, error) {
	for range 2 {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.Info(fmt.Sprintf("request failed: %v", err))
			}

			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			resp.Body.Close()

			// Handle authentication error with one retry
			challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
			token, err := getAuthorizationToken(ctx, challenge, requestURL.Host)
			if err != nil {
				return nil, err
			}
			regOpts.Token = token
			if body != nil {
				_, err = body.Seek(0, io.SeekStart)
				if err != nil {
					return nil, err
				}
			}
		case resp.StatusCode == http.StatusNotFound:
			resp.Body.Close()
			return nil, os.ErrNotExist
		case resp.StatusCode >= http.StatusBadRequest:
			defer resp.Body.Close()
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
			}
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
		default:
			return resp, nil
		}
	}

	return nil, errUnauthorized
}

// testMakeRequestDialContext specifies the dial function for the http client in
// makeRequest. It can be used to resolve hosts in model names to local
// addresses for testing. For example, the model name ("example.com/my/model")
// can be directed to push/pull from "127.0.0.1:1234".
//
// This is not safe to set across goroutines. It should be set in
// the main test goroutine, and not by tests marked to run in parallel with
// t.Parallel().
//
// It should be cleared after use, otherwise it will affect other tests.
//
// Ideally we would have some set this up the stack, but the code is not
// structured in a way that makes this easy, so this will have to do for now.
var testMakeRequestDialContext func(ctx context.Context, network, addr string) (net.Conn, error)

var errBlockedRedirect = errors.New("blocked redirect to a different host")

// isAllowedHost reports whether host may receive cross-host redirects.
var allowedRedirectHosts = []string{"ollama.com", "ollama.ai", "hf.co", "huggingface.co"}

func isAllowedHost(host string) bool {
	for _, h := range allowedRedirectHosts {
		if host == h || strings.HasSuffix(host, "."+h) {
			return true
		}
	}
	return false
}

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *registryOptions) (*http.Response, error) {
	if requestURL.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "http"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		} else if regOpts.Username != "" && regOpts.Password != "" {
			req.SetBasicAuth(regOpts.Username, regOpts.Password)
		}
	}

	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	var checkRedirect func(req *http.Request, via []*http.Request) error
	if regOpts != nil {
		checkRedirect = regOpts.CheckRedirect
	}
	if checkRedirect == nil {
		insecure := regOpts != nil && regOpts.Insecure
		// Default redirect policy: same-host only, so a registry can't steer
		// manifest or blob requests at internal addresses. CDN-backed
		// registries redirect among their own hosts, allowed via
		// isAllowedHost. --insecure opts out for trusted LAN/local registries.
		checkRedirect = func(req *http.Request, via []*http.Request) error {
			if len(via) > 10 {
				return errMaxRedirectsExceeded
			}
			if insecure || req.URL.Host == via[0].URL.Host {
				return nil
			}
			if isAllowedHost(via[0].URL.Hostname()) && isAllowedHost(req.URL.Hostname()) {
				return nil
			}
			return errBlockedRedirect
		}
	}

	c := &http.Client{
		CheckRedirect: checkRedirect,
	}
	if testMakeRequestDialContext != nil {
		tr := http.DefaultTransport.(*http.Transport).Clone()
		tr.DialContext = testMakeRequestDialContext
		c.Transport = tr
	}
	return c.Do(req)
}

func getValue(header, key string) string {
	startIdx := strings.Index(header, key+"=")
	if startIdx == -1 {
		return ""
	}

	// Move the index to the starting quote after the key.
	startIdx += len(key) + 2
	endIdx := startIdx

	for endIdx < len(header) {
		if header[endIdx] == '"' {
			if endIdx+1 < len(header) && header[endIdx+1] != ',' { // If the next character isn't a comma, continue
				endIdx++
				continue
			}
			break
		}
		endIdx++
	}
	return header[startIdx:endIdx]
}

func parseRegistryChallenge(authStr string) registryChallenge {
	authStr = strings.TrimPrefix(authStr, "Bearer ")

	return registryChallenge{
		Realm:   getValue(authStr, "realm"),
		Service: getValue(authStr, "service"),
		Scope:   getValue(authStr, "scope"),
	}
}

var errDigestMismatch = errors.New("digest mismatch, file must be downloaded again")

func verifyBlob(digest string) error {
	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		return fmt.Errorf("%w: want %s, got %s", errDigestMismatch, digest, fileDigest)
	}

	return nil
}
