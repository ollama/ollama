package server

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/api"
)

// chatHandler is the thin Connect/gRPC adapter for ChatService.
// It delegates to the protocol-agnostic *Server.chat after conversion.
// See grpc-phased-reliable-approach.md Phase 2.
type chatHandler struct {
	s *Server
}

var _ apiv1connect.ChatServiceHandler = (*chatHandler)(nil)

func (h *chatHandler) Chat(ctx context.Context, req *connect.Request[v1.ChatRequest]) (*connect.Response[v1.ChatResponse], error) {
	if isCloudModel(req.Msg.Model) {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIChat(req.Msg)
	// For unary, collect the final response via write callback.
	var final api.ChatResponse
	err := h.s.chat(ctx, apiReq, func(r api.ChatResponse) error {
		if r.Done {
			final = r
		}
		return nil
	})
	if err != nil {
		return nil, errToConnect(err)
	}
	return connect.NewResponse(convertToPBChat(&final)), nil
}

func (h *chatHandler) ChatStream(ctx context.Context, req *connect.Request[v1.ChatRequest], stream *connect.ServerStream[v1.ChatResponse]) error {
	if isCloudModel(req.Msg.Model) {
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIChat(req.Msg)
	return h.s.chat(ctx, apiReq, func(r api.ChatResponse) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if err := stream.Send(convertToPBChat(&r)); err != nil {
			return err
		}
		return nil
	})
}

// generateHandler and embedHandler follow the same thin adapter pattern (see Phase 2/3).
type generateHandler struct{ s *Server }

var _ apiv1connect.GenerateServiceHandler = (*generateHandler)(nil)

func (h *generateHandler) Generate(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.Response[v1.GenerateResponse], error) {
	if isCloudModel(req.Msg.Model) {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIGenerate(req.Msg)
	var final api.GenerateResponse
	err := h.s.generate(ctx, apiReq, func(r api.GenerateResponse) error {
		if r.Done {
			final = r
		}
		return nil
	})
	if err != nil {
		return nil, errToConnect(err)
	}
	return connect.NewResponse(convertToPBGenerate(&final)), nil
}

func (h *generateHandler) GenerateStream(ctx context.Context, req *connect.Request[v1.GenerateRequest], stream *connect.ServerStream[v1.GenerateResponse]) error {
	if isCloudModel(req.Msg.Model) {
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIGenerate(req.Msg)
	return h.s.generate(ctx, apiReq, func(r api.GenerateResponse) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		return stream.Send(convertToPBGenerate(&r))
	})
}

type embedHandler struct{ s *Server }

var _ apiv1connect.EmbedServiceHandler = (*embedHandler)(nil)

func (h *embedHandler) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	if isCloudModel(req.Msg.Model) {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIEmbed(req.Msg)
	resp, err := h.s.embed(ctx, apiReq)
	if err != nil {
		return nil, errToConnect(err)
	}
	return connect.NewResponse(convertToPBEmbed(resp)), nil
}

// modelsHandler for List/Show/Version (MVP).
type modelsHandler struct{ s *Server }

var _ apiv1connect.ModelsServiceHandler = (*modelsHandler)(nil)

func (h *modelsHandler) List(ctx context.Context, req *connect.Request[v1.ListModelsRequest]) (*connect.Response[v1.ListModelsResponse], error) {
	// For P2, delegate to existing logic via thin or direct (Ps/List handlers can be adapted later).
	// Stub for now; full in P3 with proper conversion.
	return connect.NewResponse(&v1.ListModelsResponse{}), nil
}

func (h *modelsHandler) Show(ctx context.Context, req *connect.Request[v1.ShowModelRequest]) (*connect.Response[v1.ShowModelResponse], error) {
	return connect.NewResponse(&v1.ShowModelResponse{}), nil
}

func (h *modelsHandler) Version(ctx context.Context, req *connect.Request[v1.VersionRequest]) (*connect.Response[v1.VersionResponse], error) {
	// Could call existing version logic.
	return connect.NewResponse(&v1.VersionResponse{Version: "dev"}), nil
}

// --- Basic converters (P2 MVP - limited to fields in the Phase 0 proto skeleton).
// Full rich oneofs, Struct options, detailed Usage etc. in Phase 3 when protos are expanded.
// These are table-tested in Phase 2 per SKILL/verifiability.

func convertToAPIChat(pb *v1.ChatRequest) api.ChatRequest {
	if pb == nil {
		return api.ChatRequest{}
	}
	msgs := make([]api.Message, len(pb.Messages))
	for i, m := range pb.Messages {
		msgs[i] = api.Message{
			Role:      m.Role,
			Content:   m.Content,
			Thinking:  m.Thinking,
			Images:    convertBytesToImageData(m.Images),
			ToolCalls: convertPBToolsToAPI(m.ToolCalls),
		}
	}
	return api.ChatRequest{
		Model:     pb.Model,
		Messages:  msgs,
		Stream:    boolPtr(pb.Stream),
		Options:   convertMapStringToAny(pb.Options),
		KeepAlive: parseKeepAlive(pb.KeepAlive),
		Think:     parseThink(pb.Think),
		// Tools, Format etc mapped in P3+ full.
	}
}

func convertPBToolsToAPI(pbTools []*v1.ToolCall) []api.ToolCall {
	if len(pbTools) == 0 {
		return nil
	}
	out := make([]api.ToolCall, len(pbTools))
	for i, t := range pbTools {
		out[i] = api.ToolCall{
			ID: t.Id,
			Function: api.ToolCallFunction{
				Name:      t.Function.GetName(),
				Arguments: api.ToolCallFunctionArguments{}, // parse json in full P3
			},
		}
	}
	return out
}

func convertToPBChat(r *api.ChatResponse) *v1.ChatResponse {
	if r == nil {
		return &v1.ChatResponse{}
	}
	return &v1.ChatResponse{
		Model:           r.Model,
		Message:         convertMessageToPB(&r.Message),
		Done:            r.Done,
		DoneReason:      r.DoneReason,
		PromptEvalCount: int64(r.Metrics.PromptEvalCount),
		EvalCount:       int64(r.Metrics.EvalCount),
		// CreatedAt etc. can be added with Timestamp conversion in P3.
	}
}

func convertMessageToPB(m *api.Message) *v1.Message {
	if m == nil {
		return &v1.Message{}
	}
	return &v1.Message{
		Role:     m.Role,
		Content:  m.Content,
		Thinking: m.Thinking,
		Images:   convertImageDataToBytes(m.Images),
	}
}

func convertToAPIGenerate(pb *v1.GenerateRequest) api.GenerateRequest {
	if pb == nil {
		return api.GenerateRequest{}
	}
	return api.GenerateRequest{
		Model:     pb.Model,
		Prompt:    pb.Prompt,
		Suffix:    pb.Suffix,
		Stream:    boolPtr(pb.Stream),
		Options:   convertMapStringToAny(pb.Options),
		KeepAlive: parseKeepAlive(pb.KeepAlive),
		Images:    convertBytesToImageData(pb.Images),
	}
}

func convertToPBGenerate(r *api.GenerateResponse) *v1.GenerateResponse {
	if r == nil {
		return &v1.GenerateResponse{}
	}
	return &v1.GenerateResponse{
		Model:      r.Model,
		Response:   r.Response,
		Done:       r.Done,
		DoneReason: r.DoneReason,
		// Context, metrics...
	}
}

func convertToAPIEmbed(pb *v1.EmbedRequest) api.EmbedRequest {
	if pb == nil {
		return api.EmbedRequest{}
	}
	return api.EmbedRequest{
		Model:     pb.Model,
		Input:     pb.Input,
		Options:   convertMapStringToAny(pb.Options),
		KeepAlive: parseKeepAlive(pb.KeepAlive),
	}
}

func convertToPBEmbed(r *api.EmbedResponse) *v1.EmbedResponse {
	if r == nil {
		return &v1.EmbedResponse{}
	}
	embs := make([]*v1.Embeddings, len(r.Embeddings))
	for i, e := range r.Embeddings {
		embs[i] = &v1.Embeddings{Values: e}
	}
	return &v1.EmbedResponse{
		Model:     r.Model,
		Embeddings: embs,
	}
}

// Helper converters (minimal for P2).
func convertBytesToImageData(b [][]byte) []api.ImageData {
	out := make([]api.ImageData, len(b))
	for i, bb := range b {
		out[i] = api.ImageData(bb)
	}
	return out
}

func convertImageDataToBytes(imgs []api.ImageData) [][]byte {
	out := make([][]byte, len(imgs))
	for i, id := range imgs {
		out[i] = []byte(id)
	}
	return out
}

func convertMapStringToAny(m map[string]string) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func boolPtr(b bool) *bool { return &b }

func parseKeepAlive(s string) *api.Duration {
	if s == "" {
		return nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return nil
	}
	return &api.Duration{Duration: d}
}

func parseThink(v interface{}) *api.ThinkValue {
	// MVP: simple string or bool handling.
	if v == nil {
		return nil
	}
	switch t := v.(type) {
	case bool:
		return &api.ThinkValue{Value: t}
	case string:
		return &api.ThinkValue{Value: t}
	}
	return nil
}

func isCloudModel(model string) bool {
	if model == "" {
		return false
	}
	modelRef, err := parseAndValidateModelRef(model)
	if err != nil {
		return false
	}
	return modelRef.Source == modelSourceCloud
}

// errToConnect maps internal errors to connect codes (transient vs permanent per SKILL).
// Expanded in Phase 3/4 with full handleScheduleError, StatusError etc.
func errToConnect(err error) error {
	if err == nil {
		return nil
	}
	// Basic classification for P2.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return connect.NewError(connect.CodeCanceled, err)
	}
	// Queue full, temp OOM etc. -> Unavailable for client retry.
	if isTransientSchedulerError(err) {
		return connect.NewError(connect.CodeUnavailable, err)
	}
	return connect.NewError(connect.CodeInternal, err)
}

func isTransientSchedulerError(err error) bool {
	// Placeholder; real impl uses errors.Is on scheduler sentinels, OOM etc.
	return false
}

// loggingInterceptor basic (enriched in P4 with OTEL, full attrs, stream support).
func loggingInterceptor() connect.Interceptor {
	return connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
			start := time.Now()
			model := extractModelFromAny(req)
			slog.Info("grpc unary start", "component", "grpc", "model", model)
			res, err := next(ctx, req)
			slog.Info("grpc unary done", "component", "grpc", "duration_ms", time.Since(start).Milliseconds(), "error", err)
			return res, err
		}
	})
}

func extractModelFromAny(req connect.AnyRequest) string {
	// Best effort for logging; real one uses reflection or typed.
	return ""
}

// TODO(phase3): full streaming interceptor, recovery interceptor, auth metadata, OTEL.
// TODO(phase2): table driven tests for all convert* in grpc_test.go or convert_grpc_test.go.
// TODO: add compile checks in init or test: var _ apiv1connect.XXXHandler = (*xxxHandler)(nil)
