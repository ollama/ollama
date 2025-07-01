package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// StatusError is an error with an HTTP status code and message.
// @Description StatusError represents an HTTP error response with a status code and message
type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error" example:"model not found"`
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

// ImageData represents the raw binary data of an image file.
// @Description ImageData contains the raw binary data of an image for multimodal models
type ImageData []byte

// GenerateRequest describes a request sent by [Client.Generate]. While you
// have to specify the Model and Prompt fields, all the other fields have
// reasonable defaults for basic uses.
// @Description GenerateRequest represents a text generation request with various options
type GenerateRequest struct {
	// Model is the model name; it should be a name familiar to Ollama from
	// the library at https://ollama.com/library
	Model string `json:"model" example:"llama3.2" binding:"required"`

	// Prompt is the textual prompt to send to the model.
	Prompt string `json:"prompt" example:"Why is the sky blue?"`

	// Suffix is the text that comes after the inserted text.
	Suffix string `json:"suffix" example:""`

	// System overrides the model's default system message/prompt.
	System string `json:"system" example:"You are a helpful assistant."`

	// Template overrides the model's default prompt template.
	Template string `json:"template" example:""`

	// Context is the context parameter returned from a previous call to
	// [Client.Generate]. It can be used to keep a short conversational memory.
	Context []int `json:"context,omitempty" example:[1, 2, 3, 4]`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty" example:"true"`

	// Raw set to true means that no formatting will be applied to the prompt.
	Raw bool `json:"raw,omitempty" example:"false"`

	// Format specifies the format to return a response in.
	Format json.RawMessage `json:"format,omitempty" swaggertype:"string" example:"json"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty" swaggertype:"string" example:"5m"`

	// Images is an optional list of raw image bytes accompanying this
	// request, for multimodal models.
	Images []ImageData `json:"images,omitempty"`

	// Options lists model-specific options. For example, temperature can be
	// set through this field, if the model supports it.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding. Needs to be a pointer so we can distinguish between false
	// (request that thinking _not_ be used) and unset (use the old behavior
	// before this option was introduced)
	Think *bool `json:"think,omitempty" example:"false"`
}

// ChatRequest describes a request sent by [Client.Chat].
// @Description ChatRequest represents a chat completion request with message history
type ChatRequest struct {
	// Model is the model name, as in [GenerateRequest].
	Model string `json:"model" example:"llama3.2" binding:"required"`

	// Messages is the messages of the chat - can be used to keep a chat memory.
	Messages []Message `json:"messages" binding:"required"`

	// Stream enables streaming of returned responses; true by default.
	Stream *bool `json:"stream,omitempty" example:"true"`

	// Format is the format to return the response in (e.g. "json").
	Format json.RawMessage `json:"format,omitempty" swaggertype:"string" example:"json"`

	// KeepAlive controls how long the model will stay loaded into memory
	// following the request.
	KeepAlive *Duration `json:"keep_alive,omitempty" swaggertype:"string" example:"5m"`

	// Tools is an optional list of tools the model has access to.
	Tools `json:"tools,omitempty"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`

	// Think controls whether thinking/reasoning models will think before
	// responding
	Think *bool `json:"think,omitempty" example:"false"`
}

// Tools represents a collection of function tools available to the model
// @Description Tools is a list of function tools that the model can use during conversation
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
// @Description Message represents a single message in a chat sequence
type Message struct {
	Role    string `json:"role" example:"user"`
	Content string `json:"content" example:"Hello, how are you?"`
	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking  string      `json:"thinking,omitempty" example:""`
	Images    []ImageData `json:"images,omitempty"`
	ToolCalls []ToolCall  `json:"tool_calls,omitempty"`
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

// ToolCall represents a function call requested by the model
// @Description ToolCall contains information about a function call made by the model
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the details of a function call
// @Description ToolCallFunction contains the name and arguments for a function call
type ToolCallFunction struct {
	Index     int                       `json:"index,omitempty" example:"0"`
	Name      string                    `json:"name" example:"get_weather"`
	Arguments ToolCallFunctionArguments `json:"arguments"`
}

// ToolCallFunctionArguments represents the arguments passed to a function call
// @Description ToolCallFunctionArguments contains the key-value pairs of function arguments
type ToolCallFunctionArguments map[string]any

func (t *ToolCallFunctionArguments) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// Tool represents a function tool that can be called by the model
// @Description Tool defines a function that the model can call during conversation
type Tool struct {
	Type     string       `json:"type" example:"function"`
	Items    any          `json:"items,omitempty"`
	Function ToolFunction `json:"function"`
}

// PropertyType can be either a string or an array of strings
// @Description PropertyType defines the type(s) allowed for a tool function parameter
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

// ToolFunction represents a function that can be called by the model
// @Description ToolFunction defines the signature and parameters of a callable function
type ToolFunction struct {
	Name        string `json:"name" example:"get_weather"`
	Description string `json:"description" example:"Get current weather information"`
	Parameters  struct {
		Type       string   `json:"type" example:"object"`
		Defs       any      `json:"$defs,omitempty"`
		Items      any      `json:"items,omitempty"`
		Required   []string `json:"required" example:"[\"location\"]"`
		Properties map[string]struct {
			Type        PropertyType `json:"type"`
			Items       any          `json:"items,omitempty"`
			Description string       `json:"description" example:"The location to get weather for"`
			Enum        []any        `json:"enum,omitempty"`
		} `json:"properties"`
	} `json:"parameters"`
}

func (t *ToolFunction) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// ChatResponse is the response returned by [Client.Chat]. Its fields are
// similar to [GenerateResponse].
// @Description ChatResponse represents the response from a chat completion request
type ChatResponse struct {
	Model      string    `json:"model" example:"llama3.2"`
	CreatedAt  time.Time `json:"created_at" example:"2023-01-01T00:00:00Z"`
	Message    Message   `json:"message"`
	DoneReason string    `json:"done_reason,omitempty" example:"stop"`

	Done bool `json:"done" example:"true"`

	Metrics
}

// Metrics contains performance and timing information for model operations
// @Description Metrics provides detailed timing and token count information for requests
type Metrics struct {
	TotalDuration      time.Duration `json:"total_duration,omitempty" example:"5183083" swaggertype:"primitive,integer"`
	LoadDuration       time.Duration `json:"load_duration,omitempty" example:"5183083" swaggertype:"primitive,integer"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty" example:"26"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty" example:"240700000" swaggertype:"primitive,integer"`
	EvalCount          int           `json:"eval_count,omitempty" example:"298"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty" example:"4799921000" swaggertype:"primitive,integer"`
}

// Options specified in [GenerateRequest].  If you add a new option here, also
// add it to the API docs.
// @Description Options represents model-specific options for text generation
type Options struct {
	Runner

	// Predict options used at runtime
	NumKeep          int      `json:"num_keep,omitempty" example:"4"`
	Seed             int      `json:"seed,omitempty" example:"-1"`
	NumPredict       int      `json:"num_predict,omitempty" example:"-1"`
	TopK             int      `json:"top_k,omitempty" example:"40"`
	TopP             float32  `json:"top_p,omitempty" example:"0.9"`
	MinP             float32  `json:"min_p,omitempty" example:"0.0"`
	TypicalP         float32  `json:"typical_p,omitempty" example:"1.0"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty" example:"64"`
	Temperature      float32  `json:"temperature,omitempty" example:"0.8"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty" example:"1.1"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty" example:"0.0"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty" example:"0.0"`
	Stop             []string `json:"stop,omitempty" example:"[\"\\n\"]"`
}

// Runner options which must be set when the model is loaded into memory
// @Description Runner represents options for model execution
type Runner struct {
	NumCtx    int   `json:"num_ctx,omitempty" example:"2048"`
	NumBatch  int   `json:"num_batch,omitempty" example:"512"`
	NumGPU    int   `json:"num_gpu,omitempty" example:"-1"`
	MainGPU   int   `json:"main_gpu,omitempty" example:"0"`
	UseMMap   *bool `json:"use_mmap,omitempty" example:"true"`
	NumThread int   `json:"num_thread,omitempty" example:"0"`
}

// EmbedRequest is the request passed to [Client.Embed].
// @Description EmbedRequest represents a request for embedding input data
type EmbedRequest struct {
	// Model is the model name.
	Model string `json:"model" example:"llama3.2"`

	// Input is the input to embed.
	Input any `json:"input"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty" swaggertype:"string" example:"5m"`

	Truncate *bool `json:"truncate,omitempty" example:"false"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbedResponse is the response from [Client.Embed].
// @Description EmbedResponse represents the response from an embedding request
type EmbedResponse struct {
	Model      string      `json:"model" example:"llama3.2"`
	Embeddings [][]float32 `json:"embeddings"`

	TotalDuration   time.Duration `json:"total_duration,omitempty" swaggertype:"primitive,integer"`
	LoadDuration    time.Duration `json:"load_duration,omitempty" swaggertype:"primitive,integer"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
}

// EmbeddingRequest is the request passed to [Client.Embeddings].
// @Description EmbeddingRequest represents a request for embedding a textual prompt
type EmbeddingRequest struct {
	// Model is the model name.
	Model string `json:"model" example:"llama3.2"`

	// Prompt is the textual prompt to embed.
	Prompt string `json:"prompt" example:"What is the meaning of life?"`

	// KeepAlive controls how long the model will stay loaded in memory following
	// this request.
	KeepAlive *Duration `json:"keep_alive,omitempty" swaggertype:"string" example:"5m"`

	// Options lists model-specific options.
	Options map[string]any `json:"options"`
}

// EmbeddingResponse is the response from [Client.Embeddings].
// @Description EmbeddingResponse represents the response from embedding a textual prompt
type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// CreateRequest is the request passed to [Client.Create].
// @Description CreateRequest represents a request for creating a new model
type CreateRequest struct {
	Model    string `json:"model" example:"new-model"`
	Stream   *bool  `json:"stream,omitempty" example:"true"`
	Quantize string `json:"quantize,omitempty" example:"q4_0"`

	From       string            `json:"from,omitempty" example:"base-model"`
	Files      map[string]string `json:"files,omitempty"`
	Adapters   map[string]string `json:"adapters,omitempty"`
	Template   string            `json:"template,omitempty" example:"default-template"`
	License    any               `json:"license,omitempty"`
	System     string            `json:"system,omitempty" example:"You are a helpful assistant."`
	Parameters map[string]any    `json:"parameters,omitempty"`
	Messages   []Message         `json:"messages,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
	// Deprecated: use Quantize instead
	Quantization string `json:"quantization,omitempty"`
}

// DeleteRequest is the request passed to [Client.Delete].
// @Description DeleteRequest represents a request for deleting a model
type DeleteRequest struct {
	Model string `json:"model" example:"model-to-delete"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowRequest is the request passed to [Client.Show].
// @Description ShowRequest represents a request for showing model details
type ShowRequest struct {
	Model  string `json:"model" example:"model-to-show"`
	System string `json:"system" example:"You are a helpful assistant."`

	// Template is deprecated
	Template string `json:"template" example:"default-template"`
	Verbose  bool   `json:"verbose" example:"true"`

	Options map[string]any `json:"options"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowResponse is the response returned from [Client.Show].
// @Description ShowResponse represents the response from showing model details
type ShowResponse struct {
	License       string             `json:"license,omitempty"`
	Modelfile     string             `json:"modelfile,omitempty"`
	Parameters    string             `json:"parameters,omitempty"`
	Template      string             `json:"template,omitempty"`
	System        string             `json:"system,omitempty"`
	Details       ModelDetails       `json:"details,omitempty"`
	Messages      []Message          `json:"messages,omitempty"`
	ModelInfo     map[string]any     `json:"model_info,omitempty"`
	ProjectorInfo map[string]any     `json:"projector_info,omitempty"`
	Tensors       []Tensor           `json:"tensors,omitempty"`
	Capabilities  []model.Capability `json:"capabilities,omitempty"`
	ModifiedAt    time.Time          `json:"modified_at,omitempty"`
}

// CopyRequest is the request passed to [Client.Copy].
// @Description CopyRequest represents a request for copying a model
type CopyRequest struct {
	Source      string `json:"source" example:"source-model"`
	Destination string `json:"destination" example:"destination-model"`
}

// PullRequest is the request passed to [Client.Pull].
// @Description PullRequest represents a request for pulling a model
type PullRequest struct {
	Model    string `json:"model" example:"model-to-pull"`
	Insecure bool   `json:"insecure,omitempty"` // Deprecated: ignored
	Username string `json:"username"`           // Deprecated: ignored
	Password string `json:"password"`           // Deprecated: ignored
	Stream   *bool  `json:"stream,omitempty" example:"true"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ProgressResponse is the response passed to progress functions like
// [PullProgressFunc] and [PushProgressFunc].
// @Description ProgressResponse represents the progress of a model operation
type ProgressResponse struct {
	Status    string `json:"status" example:"in-progress"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty" example:"100"`
	Completed int64  `json:"completed,omitempty" example:"50"`
}

// PushRequest is the request passed to [Client.Push].
// @Description PushRequest represents a request for pushing a model
type PushRequest struct {
	Model    string `json:"model" example:"model-to-push"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
	Stream   *bool  `json:"stream,omitempty" example:"true"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ListResponse is the response from [Client.List].
// @Description ListResponse represents the response from listing models
type ListResponse struct {
	Models []ListModelResponse `json:"models"`
}

// ProcessResponse is the response from [Client.Process].
// @Description ProcessResponse represents the response from processing models
type ProcessResponse struct {
	Models []ProcessModelResponse `json:"models"`
}

// ListModelResponse is a single model description in [ListResponse].
// @Description ListModelResponse represents a single model in the list response
type ListModelResponse struct {
	Name       string       `json:"name" example:"model-name"`
	Model      string       `json:"model" example:"model-name"`
	ModifiedAt time.Time    `json:"modified_at" example:"2023-01-01T00:00:00Z"`
	Size       int64        `json:"size" example:"1024"`
	Digest     string       `json:"digest" example:"sha256:abcd1234"`
	Details    ModelDetails `json:"details,omitempty"`
}

// ProcessModelResponse is a single model description in [ProcessResponse].
// @Description ProcessModelResponse represents a single model in the process response
type ProcessModelResponse struct {
	Name      string       `json:"name" example:"model-name"`
	Model     string       `json:"model" example:"model-name"`
	Size      int64        `json:"size" example:"1024"`
	Digest    string       `json:"digest" example:"sha256:abcd1234"`
	Details   ModelDetails `json:"details,omitempty"`
	ExpiresAt time.Time    `json:"expires_at" example:"2023-01-01T00:00:00Z"`
	SizeVRAM  int64        `json:"size_vram" example:"512"`
}

// TokenResponse represents an authentication token response
// @Description TokenResponse contains an authentication token for API access
type TokenResponse struct {
	Token string `json:"token" example:"abcd1234"`
}

// GenerateResponse is the response passed into [GenerateResponseFunc].
// @Description GenerateResponse represents the response from a text generation request
type GenerateResponse struct {
	// Model is the model name that generated the response.
	Model string `json:"model" example:"llama3.2"`

	// CreatedAt is the timestamp of the response.
	CreatedAt time.Time `json:"created_at" example:"2023-01-01T00:00:00Z"`

	// Response is the textual response itself.
	Response string `json:"response" example:"The sky is blue because of Rayleigh scattering."`

	// Thinking contains the text that was inside thinking tags in the
	// original model output when ChatRequest.Think is enabled.
	Thinking string `json:"thinking,omitempty" example:""`

	// Done specifies if the response is complete.
	Done bool `json:"done" example:"true"`

	// DoneReason is the reason the model stopped generating text.
	DoneReason string `json:"done_reason,omitempty" example:"stop"`

	// Context is an encoding of the conversation used in this response; this
	// can be sent in the next request to keep a conversational memory.
	Context []int `json:"context,omitempty" example:[1, 2, 3, 4]`

	Metrics
}

// ModelDetails provides details about a model.
// @Description ModelDetails represents detailed information about a model
type ModelDetails struct {
	ParentModel       string   `json:"parent_model" example:"base-model"`
	Format            string   `json:"format" example:"binary"`
	Family            string   `json:"family" example:"llama"`
	Families          []string `json:"families" example:"[\"llama\", \"alpaca\"]"`
	ParameterSize     string   `json:"parameter_size" example:"7B"`
	QuantizationLevel string   `json:"quantization_level" example:"q4_0"`
}

// Tensor describes the metadata for a given tensor.
// @Description Tensor represents metadata for a tensor
type Tensor struct {
	Name  string   `json:"name" example:"embedding_tensor"`
	Type  string   `json:"type" example:"float32"`
	Shape []uint64 `json:"shape" example:[1024, 768]`
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

// Duration represents a time duration with custom JSON marshaling
// @Description Duration wraps time.Duration with custom JSON serialization support
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
			d.Duration = time.Duration(int(t) * int(time.Second))
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
