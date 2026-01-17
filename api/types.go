package api

import (
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/types/model"
)

// StatusError is an error with an HTTP status code and message.
type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type AuthorizationError struct {
	StatusCode int
	Status     string
	SigninURL  string `json:"signin_url"`
}

func (e AuthorizationError) Error() string {
	if e.Status != "" {
		return e.Status
	}
	return "something went wrong, please see the ollama server logs for details"
}

// ImageData represents the raw binary data of an image file.
type ImageData []byte

// GenerateRequest describes a request sent by [Client.Generate]. While you
// have to specify the Model and Prompt fields, all the other fields have
// reasonable defaults for basic uses.
type GenerateRequest struct {
	// Model is the model name; it should be a name familiar to Ollama from
	// the library at https://ollama.com/library
	Model string `json:"model"`

	// Prompt is the textual prompt to send to the model.
	Prompt string `json:"prompt"`

	// Suffix is the text that comes after the inserted text.
	Suffix string `json:"suffix"`

	// System overrides the model's default system message/prompt.
	System string `json:"system"`

	// Template overrides the model's default prompt template.
	Template string `json:"template"`

	// Context is the context parameter returned from a previous call to
	// [Client.Generate]. It can be used to keep a short conversational memory.
	Context []int `json:"context,omitempty"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Raw set to true means that no formatting will be applied to the prompt.
	Raw bool `json:"raw,omitempty"`

	// Format specifies the format to return a response in.
	Format json.RawMessage `json:"format,omitempty"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Images is an optional list of raw image bytes accompanying this
	// request, for multimodal models.
	Images []ImageData `json:"images,omitempty"`

	// Options lists model-specific options. For example, temperature can be
	// set through this field, if the model supports it.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding. Can be a boolean (true/false) or a string ("high", "medium", "low")
	// for supported models. Needs to be a pointer so we can distinguish between false
	// (request that thinking _not_ be used) and unset (use the old behavior
	// before this option was introduced)
	Think *ThinkValue `json:"think,omitempty"`

	// Truncate is a boolean that, when set to true, truncates the chat history messages
	// if the rendered prompt exceeds the context length limit.
	Truncate *bool `json:"truncate,omitempty"`

	// Shift is a boolean that, when set to true, shifts the chat history
	// when hitting the context length limit instead of erroring.
	Shift *bool `json:"shift,omitempty"`

	// DebugRenderOnly is a debug option that, when set to true, returns the rendered
	// template instead of calling the model.
	DebugRenderOnly bool `json:"_debug_render_only,omitempty"`

	// Logprobs specifies whether to return log probabilities of the output tokens.
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number of most likely tokens to return at each token position,
	// each with an associated log probability. Only applies when Logprobs is true.
	// Valid values are 0-20. Default is 0 (only return the selected token's logprob).
	TopLogprobs int `json:"top_logprobs,omitempty"`

	// Experimental: Image generation fields (may change or be removed)

	// Width is the width of the generated image in pixels.
	// Only used for image generation models.
	Width int32 `json:"width,omitempty"`

	// Height is the height of the generated image in pixels.
	// Only used for image generation models.
	Height int32 `json:"height,omitempty"`

	// Steps is the number of diffusion steps for image generation.
	// Only used for image generation models.
	Steps int32 `json:"steps,omitempty"`
}

// ChatRequest describes a request sent by [Client.Chat].
type ChatRequest struct {
	// Model is the model name, as in [GenerateRequest].
	Model string `json:"model"`

	// Messages is the messages of the chat - can be used to keep a chat memory.
	Messages []Message `json:"messages"`

	// Stream enables streaming of returned responses; true by default.
	Stream *bool `json:"stream,omitempty"`

	// Format is the format to return the response in (e.g. "json").
	Format json.RawMessage `json:"format,omitempty"`

	// KeepAlive controls how long the model will stay loaded into memory
	// following the request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Tools is an optional list of tools the model has access to.
	Tools `json:"tools,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding. Can be a boolean (true/false) or a string ("high", "medium", "low")
	// for supported models.
	Think *ThinkValue `json:"think,omitempty"`

	// Truncate is a boolean that, when set to true, truncates the chat history messages
	// if the rendered prompt exceeds the context length limit.
	Truncate *bool `json:"truncate,omitempty"`

	// Shift is a boolean that, when set to true, shifts the chat history
	// when hitting the context length limit instead of erroring.
	Shift *bool `json:"shift,omitempty"`

	// DebugRenderOnly is a debug option that, when set to true, returns the rendered
	// template instead of calling the model.
	DebugRenderOnly bool `json:"_debug_render_only,omitempty"`

	// Logprobs specifies whether to return log probabilities of the output tokens.
	Logprobs bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number of most likely tokens to return at each token position,
	// each with an associated log probability. Only applies when Logprobs is true.
	// Valid values are 0-20. Default is 0 (only return the selected token's logprob).
	TopLogprobs int `json:"top_logprobs,omitempty"`
}

type Tools []Tool

func (t Tools) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

func (t Tool) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// Message is a single message in a chat sequence. The message contains the
// role ("system", "user", or "assistant"), the content and an optional list
// of images.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking   string      `json:"thinking,omitempty"`
	Images     []ImageData `json:"images,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolName   string      `json:"tool_name,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	type Alias Message
	var a Alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}

	*m = Message(a)
	m.Role = strings.ToLower(m.Role)
	return nil
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Index     int                       `json:"index"`
	Name      string                    `json:"name"`
	Arguments ToolCallFunctionArguments `json:"arguments"`
}

// ToolCallFunctionArguments holds tool call arguments in insertion order.
type ToolCallFunctionArguments struct {
	om *orderedmap.Map[string, any]
}

// NewToolCallFunctionArguments creates a new empty ToolCallFunctionArguments.
func NewToolCallFunctionArguments() ToolCallFunctionArguments {
	return ToolCallFunctionArguments{om: orderedmap.New[string, any]()}
}

// Get retrieves a value by key.
func (t *ToolCallFunctionArguments) Get(key string) (any, bool) {
	if t == nil || t.om == nil {
		return nil, false
	}
	return t.om.Get(key)
}

// Set sets a key-value pair, preserving insertion order.
func (t *ToolCallFunctionArguments) Set(key string, value any) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, any]()
	}
	t.om.Set(key, value)
}

// Len returns the number of arguments.
func (t *ToolCallFunctionArguments) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all key-value pairs in insertion order.
func (t *ToolCallFunctionArguments) All() iter.Seq2[string, any] {
	if t == nil || t.om == nil {
		return func(yield func(string, any) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolCallFunctionArguments) ToMap() map[string]any {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t *ToolCallFunctionArguments) String() string {
	if t == nil || t.om == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t.om)
	return string(bts)
}

func (t *ToolCallFunctionArguments) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, any]()
	return json.Unmarshal(data, t.om)
}

func (t ToolCallFunctionArguments) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(t.om)
}

type Tool struct {
	Type     string       `json:"type"`
	Items    any          `json:"items,omitempty"`
	Function ToolFunction `json:"function"`
}

// PropertyType can be either a string or an array of strings
type PropertyType []string

// UnmarshalJSON implements the json.Unmarshaler interface
func (pt *PropertyType) UnmarshalJSON(data []byte) error {
	// Try to unmarshal as a string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*pt = []string{s}
		return nil
	}

	// If that fails, try to unmarshal as an array of strings
	var a []string
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*pt = a
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (pt PropertyType) MarshalJSON() ([]byte, error) {
	if len(pt) == 1 {
		// If there's only one type, marshal as a string
		return json.Marshal(pt[0])
	}
	// Otherwise marshal as an array
	return json.Marshal([]string(pt))
}

// String returns a string representation of the PropertyType
func (pt PropertyType) String() string {
	if len(pt) == 0 {
		return ""
	}
	if len(pt) == 1 {
		return pt[0]
	}
	return fmt.Sprintf("%v", []string(pt))
}

// ToolPropertiesMap holds tool properties in insertion order.
type ToolPropertiesMap struct {
	om *orderedmap.Map[string, ToolProperty]
}

// NewToolPropertiesMap creates a new empty ToolPropertiesMap.
func NewToolPropertiesMap() *ToolPropertiesMap {
	return &ToolPropertiesMap{om: orderedmap.New[string, ToolProperty]()}
}

// Get retrieves a property by name.
func (t *ToolPropertiesMap) Get(key string) (ToolProperty, bool) {
	if t == nil || t.om == nil {
		return ToolProperty{}, false
	}
	return t.om.Get(key)
}

// Set sets a property, preserving insertion order.
func (t *ToolPropertiesMap) Set(key string, value ToolProperty) {
	if t == nil {
		return
	}
	if t.om == nil {
		t.om = orderedmap.New[string, ToolProperty]()
	}
	t.om.Set(key, value)
}

// Len returns the number of properties.
func (t *ToolPropertiesMap) Len() int {
	if t == nil || t.om == nil {
		return 0
	}
	return t.om.Len()
}

// All returns an iterator over all properties in insertion order.
func (t *ToolPropertiesMap) All() iter.Seq2[string, ToolProperty] {
	if t == nil || t.om == nil {
		return func(yield func(string, ToolProperty) bool) {}
	}
	return t.om.All()
}

// ToMap returns a regular map (order not preserved).
func (t *ToolPropertiesMap) ToMap() map[string]ToolProperty {
	if t == nil || t.om == nil {
		return nil
	}
	return t.om.ToMap()
}

func (t ToolPropertiesMap) MarshalJSON() ([]byte, error) {
	if t.om == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.om)
}

func (t *ToolPropertiesMap) UnmarshalJSON(data []byte) error {
	t.om = orderedmap.New[string, ToolProperty]()
	return json.Unmarshal(data, t.om)
}

type ToolProperty struct {
	AnyOf       []ToolProperty     `json:"anyOf,omitempty"`
	Type        PropertyType       `json:"type,omitempty"`
	Items       any                `json:"items,omitempty"`
	Description string             `json:"description,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	Properties  *ToolPropertiesMap `json:"properties,omitempty"`
}

// ToTypeScriptType converts a ToolProperty to a TypeScript type string
func (tp ToolProperty) ToTypeScriptType() string {
	if len(tp.AnyOf) > 0 {
		var types []string
		for _, anyOf := range tp.AnyOf {
			types = append(types, anyOf.ToTypeScriptType())
		}
		return strings.Join(types, " | ")
	}

	if len(tp.Type) == 0 {
		return "any"
	}

	if len(tp.Type) == 1 {
		return mapToTypeScriptType(tp.Type[0])
	}

	var types []string
	for _, t := range tp.Type {
		types = append(types, mapToTypeScriptType(t))
	}
	return strings.Join(types, " | ")
}

// mapToTypeScriptType maps JSON Schema types to TypeScript types
func mapToTypeScriptType(jsonType string) string {
	switch jsonType {
	case "string":
		return "string"
	case "number", "integer":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		return "any[]"
	case "object":
		return "Record<string, any>"
	case "null":
		return "null"
	default:
		return "any"
	}
}

type ToolFunctionParameters struct {
	Type       string             `json:"type"`
	Defs       any                `json:"$defs,omitempty"`
	Items      any                `json:"items,omitempty"`
	Required   []string           `json:"required,omitempty"`
	Properties *ToolPropertiesMap `json:"properties"`
}

func (t *ToolFunctionParameters) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  ToolFunctionParameters `json:"parameters"`
}

func (t *ToolFunction) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// TokenLogprob represents log probability information for a single token alternative.
type TokenLogprob struct {
	// Token is the text representation of the token.
	Token string `json:"token"`

	// Logprob is the log probability of this token.
	Logprob float64 `json:"logprob"`

	// Bytes contains the raw byte representation of the token
	Bytes []int `json:"bytes,omitempty"`
}

// Logprob contains log probability information for a generated token.
type Logprob struct {
	TokenLogprob

	// TopLogprobs contains the most likely tokens and their log probabilities
	// at this position, if requested via TopLogprobs parameter.
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

// ChatResponse is the response returned by [Client.Chat]. Its fields are
// similar to [GenerateResponse].
type ChatResponse struct {
	// Model is the model name that generated the response.
	Model string `json:"model"`

	// RemoteModel is the name of the upstream model that generated the response.
	RemoteModel string `json:"remote_model,omitempty"`

	// RemoteHost is the URL of the upstream Ollama host that generated the response.
	RemoteHost string `json:"remote_host,omitempty"`

	// CreatedAt is the timestamp of the response.
	CreatedAt time.Time `json:"created_at"`

	// Message contains the message or part of a message from the model.
	Message Message `json:"message"`

	// Done specifies if the response is complete.
	Done bool `json:"done"`

	// DoneReason is the reason the model stopped generating text.
	DoneReason string `json:"done_reason,omitempty"`

	DebugInfo *DebugInfo `json:"_debug_info,omitempty"`

	// Logprobs contains log probability information for the generated tokens,
	// if requested via the Logprobs parameter.
	Logprobs []Logprob `json:"logprobs,omitempty"`

	Metrics
}

// DebugInfo contains debug information for template rendering
type DebugInfo struct {
	RenderedTemplate string `json:"rendered_template"`
	ImageCount       int    `json:"image_count,omitempty"`
}

type Metrics struct {
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

// Options specified in [GenerateRequest].  If you add a new option here, also
// add it to the API docs.
type Options struct {
	Runner

	// Predict options used at runtime
	NumKeep          int      `json:"num_keep,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// Runner options which must be set when the model is loaded into memory
type Runner struct {
	NumCtx    int   `json:"num_ctx,omitempty"`
	NumBatch  int   `json:"num_batch,omitempty"`
	NumGPU    int   `json:"num_gpu,omitempty"`
	MainGPU   int   `json:"main_gpu,omitempty"`
	UseMMap   *bool `json:"use_mmap,omitempty"`
	NumThread int   `json:"num_thread,omitempty"`
}

// EmbedRequest is the request passed to [Client.Embed].
type EmbedRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Input is the input to embed.
	Input any `json:"input"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Truncate truncates the input to fit the model's max sequence length.
	Truncate *bool `json:"truncate,omitempty"`

	// Dimensions truncates the output embedding to the specified dimension.
	Dimensions int `json:"dimensions,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbedResponse is the response from [Client.Embed].
type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`

	TotalDuration   time.Duration `json:"total_duration,omitempty"`
	LoadDuration    time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
}

// EmbeddingRequest is the request passed to [Client.Embeddings].
type EmbeddingRequest struct {
	// Model is the model name.
	Model string `json:"model"`

	// Prompt is the textual prompt to embed.
	Prompt string `json:"prompt"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbeddingResponse is the response from [Client.Embeddings].
type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// CreateRequest is the request passed to [Client.Create].
type CreateRequest struct {
	// Model is the model name to create.
	Model string `json:"model"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Quantize is the quantization format for the model; leave blank to not change the quantization level.
	Quantize string `json:"quantize,omitempty"`

	// From is the name of the model or file to use as the source.
	From string `json:"from,omitempty"`

	// RemoteHost is the URL of the upstream ollama API for the model (if any).
	RemoteHost string `json:"remote_host,omitempty"`

	// Files is a map of files include when creating the model.
	Files map[string]string `json:"files,omitempty"`

	// Adapters is a map of LoRA adapters to include when creating the model.
	Adapters map[string]string `json:"adapters,omitempty"`

	// Template is the template used when constructing a request to the model.
	Template string `json:"template,omitempty"`

	// License is a string or list of strings for licenses.
	License any `json:"license,omitempty"`

	// System is the system prompt for the model.
	System string `json:"system,omitempty"`

	// Parameters is a map of hyper-parameters which are applied to the model.
	Parameters map[string]any `json:"parameters,omitempty"`

	// Messages is a list of messages added to the model before chat and generation requests.
	Messages []Message `json:"messages,omitempty"`

	Renderer string `json:"renderer,omitempty"`
	Parser   string `json:"parser,omitempty"`

	// Requires is the minimum version of Ollama required by the model.
	Requires string `json:"requires,omitempty"`

	// Info is a map of additional information for the model
	Info map[string]any `json:"info,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
	// Deprecated: use Quantize instead
	Quantization string `json:"quantization,omitempty"`
}

// DeleteRequest is the request passed to [Client.Delete].
type DeleteRequest struct {
	Model string `json:"model"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowRequest is the request passed to [Client.Show].
type ShowRequest struct {
	Model  string `json:"model"`
	System string `json:"system"`

	// Template is deprecated
	Template string `json:"template"`
	Verbose  bool   `json:"verbose"`

	Options map[string]any `json:"options"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowResponse is the response returned from [Client.Show].
type ShowResponse struct {
	License       string             `json:"license,omitempty"`
	Modelfile     string             `json:"modelfile,omitempty"`
	Parameters    string             `json:"parameters,omitempty"`
	Template      string             `json:"template,omitempty"`
	System        string             `json:"system,omitempty"`
	Renderer      string             `json:"renderer,omitempty"`
	Parser        string             `json:"parser,omitempty"`
	Details       ModelDetails       `json:"details,omitempty"`
	Messages      []Message          `json:"messages,omitempty"`
	RemoteModel   string             `json:"remote_model,omitempty"`
	RemoteHost    string             `json:"remote_host,omitempty"`
	ModelInfo     map[string]any     `json:"model_info,omitempty"`
	ProjectorInfo map[string]any     `json:"projector_info,omitempty"`
	Tensors       []Tensor           `json:"tensors,omitempty"`
	Capabilities  []model.Capability `json:"capabilities,omitempty"`
	ModifiedAt    time.Time          `json:"modified_at,omitempty"`
	Requires      string             `json:"requires,omitempty"`
}

// CopyRequest is the request passed to [Client.Copy].
type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

// PullRequest is the request passed to [Client.Pull].
type PullRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"` // Deprecated: ignored
	Username string `json:"username"`           // Deprecated: ignored
	Password string `json:"password"`           // Deprecated: ignored
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ProgressResponse is the response passed to progress functions like
// [PullProgressFunc] and [PushProgressFunc].
type ProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// PushRequest is the request passed to [Client.Push].
type PushRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ListResponse is the response from [Client.List].
type ListResponse struct {
	Models []ListModelResponse `json:"models"`
}

// ProcessResponse is the response from [Client.Process].
type ProcessResponse struct {
	Models []ProcessModelResponse `json:"models"`
}

// ListModelResponse is a single model description in [ListResponse].
type ListModelResponse struct {
	Name        string       `json:"name"`
	Model       string       `json:"model"`
	RemoteModel string       `json:"remote_model,omitempty"`
	RemoteHost  string       `json:"remote_host,omitempty"`
	ModifiedAt  time.Time    `json:"modified_at"`
	Size        int64        `json:"size"`
	Digest      string       `json:"digest"`
	Details     ModelDetails `json:"details,omitempty"`
}

// ProcessModelResponse is a single model description in [ProcessResponse].
type ProcessModelResponse struct {
	Name          string       `json:"name"`
	Model         string       `json:"model"`
	Size          int64        `json:"size"`
	Digest        string       `json:"digest"`
	Details       ModelDetails `json:"details,omitempty"`
	ExpiresAt     time.Time    `json:"expires_at"`
	SizeVRAM      int64        `json:"size_vram"`
	ContextLength int          `json:"context_length"`
}

type TokenResponse struct {
	Token string `json:"token"`
}

// GenerateResponse is the response passed into [GenerateResponseFunc].
type GenerateResponse struct {
	// Model is the model name that generated the response.
	Model string `json:"model"`

	// RemoteModel is the name of the upstream model that generated the response.
	RemoteModel string `json:"remote_model,omitempty"`

	// RemoteHost is the URL of the upstream Ollama host that generated the response.
	RemoteHost string `json:"remote_host,omitempty"`

	// CreatedAt is the timestamp of the response.
	CreatedAt time.Time `json:"created_at"`

	// Response is the textual response itself.
	Response string `json:"response"`

	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking string `json:"thinking,omitempty"`

	// Done specifies if the response is complete.
	Done bool `json:"done"`

	// DoneReason is the reason the model stopped generating text.
	DoneReason string `json:"done_reason,omitempty"`

	// Context is an encoding of the conversation used in this response; this
	// can be sent in the next request to keep a conversational memory.
	Context []int `json:"context,omitempty"`

	Metrics

	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	DebugInfo *DebugInfo `json:"_debug_info,omitempty"`

	// Logprobs contains log probability information for the generated tokens,
	// if requested via the Logprobs parameter.
	Logprobs []Logprob `json:"logprobs,omitempty"`

	// Experimental: Image generation fields (may change or be removed)

	// Image contains a base64-encoded generated image.
	// Only present for image generation models.
	Image string `json:"image,omitempty"`

	// Completed is the number of completed steps in image generation.
	// Only present for image generation models during streaming.
	Completed int64 `json:"completed,omitempty"`

	// Total is the total number of steps for image generation.
	// Only present for image generation models during streaming.
	Total int64 `json:"total,omitempty"`
}

// ModelDetails provides details about a model.
type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// UserResponse provides information about a user.
type UserResponse struct {
	ID        uuid.UUID `json:"id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	Bio       string    `json:"bio,omitempty"`
	AvatarURL string    `json:"avatarurl,omitempty"`
	FirstName string    `json:"firstname,omitempty"`
	LastName  string    `json:"lastname,omitempty"`
	Plan      string    `json:"plan,omitempty"`
}

// Tensor describes the metadata for a given tensor.
type Tensor struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Shape []uint64 `json:"shape"`
}

func (m *Metrics) Summary() {
	if m.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", m.TotalDuration)
	}

	if m.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", m.LoadDuration)
	}

	if m.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", m.PromptEvalCount)
	}

	if m.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", m.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(m.PromptEvalCount)/m.PromptEvalDuration.Seconds())
	}

	if m.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", m.EvalCount)
	}

	if m.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", m.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(m.EvalCount)/m.EvalDuration.Seconds())
	}
}

func (opts *Options) FromMap(m map[string]any) error {
	valueOpts := reflect.ValueOf(opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts).Elem()   // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	for key, val := range m {
		opt, ok := jsonOpts[key]
		if !ok {
			slog.Warn("invalid option provided", "option", key)
			continue
		}

		field := valueOpts.FieldByName(opt.Name)
		if field.IsValid() && field.CanSet() {
			if val == nil {
				continue
			}

			switch field.Kind() {
			case reflect.Int:
				switch t := val.(type) {
				case int64:
					field.SetInt(t)
				case float64:
					// when JSON unmarshals numbers, it uses float64, not int
					field.SetInt(int64(t))
				default:
					return fmt.Errorf("option %q must be of type integer", key)
				}
			case reflect.Bool:
				val, ok := val.(bool)
				if !ok {
					return fmt.Errorf("option %q must be of type boolean", key)
				}
				field.SetBool(val)
			case reflect.Float32:
				// JSON unmarshals to float64
				val, ok := val.(float64)
				if !ok {
					return fmt.Errorf("option %q must be of type float32", key)
				}
				field.SetFloat(val)
			case reflect.String:
				val, ok := val.(string)
				if !ok {
					return fmt.Errorf("option %q must be of type string", key)
				}
				field.SetString(val)
			case reflect.Slice:
				// JSON unmarshals to []any, not []string
				val, ok := val.([]any)
				if !ok {
					return fmt.Errorf("option %q must be of type array", key)
				}
				// convert []any to []string
				slice := make([]string, len(val))
				for i, item := range val {
					str, ok := item.(string)
					if !ok {
						return fmt.Errorf("option %q must be of an array of strings", key)
					}
					slice[i] = str
				}
				field.Set(reflect.ValueOf(slice))
			case reflect.Pointer:
				var b bool
				if field.Type() == reflect.TypeOf(&b) {
					val, ok := val.(bool)
					if !ok {
						return fmt.Errorf("option %q must be of type boolean", key)
					}
					field.Set(reflect.ValueOf(&val))
				} else {
					return fmt.Errorf("unknown type loading config params: %v %v", field.Kind(), field.Type())
				}
			default:
				return fmt.Errorf("unknown type loading config params: %v", field.Kind())
			}
		}
	}

	return nil
}

// DefaultOptions is the default set of options for [GenerateRequest]; these
// values are used unless the user specifies other values explicitly.
func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict: -1,

		// set a minimal num_keep to avoid issues on context shifts
		NumKeep:          4,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TypicalP:         1.0,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Seed:             -1,

		Runner: Runner{
			// options set when the model is loaded
			NumCtx:    int(envconfig.ContextLength()),
			NumBatch:  512,
			NumGPU:    -1, // -1 here indicates that NumGPU should be set dynamically
			NumThread: 0,  // let the runtime decide
			UseMMap:   nil,
		},
	}
}

// ThinkValue represents a value that can be a boolean or a string ("high", "medium", "low")
type ThinkValue struct {
	// Value can be a bool or string
	Value interface{}
}

// IsValid checks if the ThinkValue is valid
func (t *ThinkValue) IsValid() bool {
	if t == nil || t.Value == nil {
		return true // nil is valid (means not set)
	}

	switch v := t.Value.(type) {
	case bool:
		return true
	case string:
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// IsBool returns true if the value is a boolean
func (t *ThinkValue) IsBool() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(bool)
	return ok
}

// IsString returns true if the value is a string
func (t *ThinkValue) IsString() bool {
	if t == nil || t.Value == nil {
		return false
	}
	_, ok := t.Value.(string)
	return ok
}

// Bool returns the value as a bool (true if enabled in any way)
func (t *ThinkValue) Bool() bool {
	if t == nil || t.Value == nil {
		return false
	}

	switch v := t.Value.(type) {
	case bool:
		return v
	case string:
		// Any string value ("high", "medium", "low") means thinking is enabled
		return v == "high" || v == "medium" || v == "low"
	default:
		return false
	}
}

// String returns the value as a string
func (t *ThinkValue) String() string {
	if t == nil || t.Value == nil {
		return ""
	}

	switch v := t.Value.(type) {
	case string:
		return v
	case bool:
		if v {
			return "medium" // Default level when just true
		}
		return ""
	default:
		return ""
	}
}

// UnmarshalJSON implements json.Unmarshaler
func (t *ThinkValue) UnmarshalJSON(data []byte) error {
	// Try to unmarshal as bool first
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.Value = b
		return nil
	}

	// Try to unmarshal as string
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		// Validate string values
		if s != "high" && s != "medium" && s != "low" {
			return fmt.Errorf("invalid think value: %q (must be \"high\", \"medium\", \"low\", true, or false)", s)
		}
		t.Value = s
		return nil
	}

	return fmt.Errorf("think must be a boolean or string (\"high\", \"medium\", \"low\", true, or false)")
}

// MarshalJSON implements json.Marshaler
func (t *ThinkValue) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}

type Duration struct {
	time.Duration
}

func (d Duration) MarshalJSON() ([]byte, error) {
	if d.Duration < 0 {
		return []byte("-1"), nil
	}
	return []byte("\"" + d.Duration.String() + "\""), nil
}

func (d *Duration) UnmarshalJSON(b []byte) (err error) {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	d.Duration = 5 * time.Minute

	switch t := v.(type) {
	case float64:
		if t < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		} else {
			d.Duration = time.Duration(t * float64(time.Second))
		}
	case string:
		d.Duration, err = time.ParseDuration(t)
		if err != nil {
			return err
		}
		if d.Duration < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		}
	default:
		return fmt.Errorf("Unsupported type: '%s'", reflect.TypeOf(v))
	}

	return nil
}

// FormatParams converts specified parameter options to their correct types
func FormatParams(params map[string][]string) (map[string]any, error) {
	opts := Options{}
	valueOpts := reflect.ValueOf(&opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts)           // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	out := make(map[string]any)
	// iterate params and set values based on json struct tags
	for key, vals := range params {
		if opt, ok := jsonOpts[key]; !ok {
			return nil, fmt.Errorf("unknown parameter '%s'", key)
		} else {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				switch field.Kind() {
				case reflect.Float32:
					floatVal, err := strconv.ParseFloat(vals[0], 32)
					if err != nil {
						return nil, fmt.Errorf("invalid float value %s", vals)
					}

					out[key] = float32(floatVal)
				case reflect.Int:
					intVal, err := strconv.ParseInt(vals[0], 10, 64)
					if err != nil {
						return nil, fmt.Errorf("invalid int value %s", vals)
					}

					out[key] = intVal
				case reflect.Bool:
					boolVal, err := strconv.ParseBool(vals[0])
					if err != nil {
						return nil, fmt.Errorf("invalid bool value %s", vals)
					}

					out[key] = boolVal
				case reflect.String:
					out[key] = vals[0]
				case reflect.Slice:
					// TODO: only string slices are supported right now
					out[key] = vals
				case reflect.Pointer:
					var b bool
					if field.Type() == reflect.TypeOf(&b) {
						boolVal, err := strconv.ParseBool(vals[0])
						if err != nil {
							return nil, fmt.Errorf("invalid bool value %s", vals)
						}
						out[key] = &boolVal
					} else {
						return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
					}
				default:
					return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
				}
			}
		}
	}

	return out, nil
}
