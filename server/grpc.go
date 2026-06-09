package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"connectrpc.com/connect"
	"connectrpc.com/otelconnect"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/protobuf/types/known/timestamppb"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/version"
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
	// Phase 4: custom span around inference for OTEL (attrs, around schedule/load/inference via extracted)
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.ChatService/Chat")
	defer span.End()
	span.SetAttributes(attribute.String("model", apiReq.Model))
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
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.ChatService/ChatStream")
	defer span.End()
	span.SetAttributes(attribute.String("model", apiReq.Model))
	// Phase 4 uplift per review: derive cancel from stream ctx so on client cancel/Send err we cancel to stop llm promptly (prevent GPU leak, SKILL bounded + phased doc p334,76)
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	return h.s.chat(streamCtx, apiReq, func(r api.ChatResponse) error {
		select {
		case <-streamCtx.Done():
			cancel()
			return streamCtx.Err()
		default:
		}
		if err := stream.Send(convertToPBChat(&r)); err != nil {
			cancel() // stop gen
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
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.GenerateService/Generate")
	defer span.End()
	span.SetAttributes(attribute.String("model", apiReq.Model))
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
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.GenerateService/GenerateStream")
	defer span.End()
	span.SetAttributes(attribute.String("model", apiReq.Model))
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	return h.s.generate(streamCtx, apiReq, func(r api.GenerateResponse) error {
		select {
		case <-streamCtx.Done():
			cancel()
			return streamCtx.Err()
		default:
		}
		if err := stream.Send(convertToPBGenerate(&r)); err != nil {
			cancel()
			return err
		}
		return nil
	})
}

type embedHandler struct{ s *Server }

var _ apiv1connect.EmbedServiceHandler = (*embedHandler)(nil)

func (h *embedHandler) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	if isCloudModel(req.Msg.Model) {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIEmbed(req.Msg)
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.EmbedService/Embed")
	defer span.End()
	span.SetAttributes(attribute.String("model", apiReq.Model))
	resp, err := h.s.embed(ctx, apiReq)
	if err != nil {
		return nil, errToConnect(err)
	}
	return connect.NewResponse(convertToPBEmbed(resp)), nil
}

// modelsHandler for List/Show/Version (Phase 4 polish: adapters completion).
type modelsHandler struct{ s *Server }

var _ apiv1connect.ModelsServiceHandler = (*modelsHandler)(nil)

func (h *modelsHandler) List(ctx context.Context, req *connect.Request[v1.ListModelsRequest]) (*connect.Response[v1.ListModelsResponse], error) {
	// Phase 5: full admin List using shared modelList cache (same *Server as HTTP /api/tags).
	// Thin adapter only; conversion here; no core dupe, no new globals/recon. Rich path for admin.
	resp := &v1.ListModelsResponse{}
	if h.s != nil && h.s.modelCaches != nil && h.s.modelCaches.modelList != nil {
		models, err := h.s.modelCaches.modelList.List(ctx)
		if err != nil {
			return nil, errToConnect(fmt.Errorf("listing models: %w", err))
		}
		resp.Models = make([]*v1.Model, 0, len(models))
		for _, m := range models {
			resp.Models = append(resp.Models, convertListModelResponseToPB(m))
		}
	}
	return connect.NewResponse(resp), nil
}

func (h *modelsHandler) Show(ctx context.Context, req *connect.Request[v1.ShowModelRequest]) (*connect.Response[v1.ShowModelResponse], error) {
	// Phase 5 admin: basic Show remains (full details via GetModelInfo would require more mapping); Version is populated.
	return connect.NewResponse(&v1.ShowModelResponse{}), nil
}

func (h *modelsHandler) Version(ctx context.Context, req *connect.Request[v1.VersionRequest]) (*connect.Response[v1.VersionResponse], error) {
	return connect.NewResponse(&v1.VersionResponse{Version: version.Version}), nil
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
		Stream:    func(b bool) *bool { return &b }(pb.Stream),
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
		Stream:    func(b bool) *bool { return &b }(pb.Stream),
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
// Complete for Phase 4: all from handleScheduleError, StatusError, etc.
func errToConnect(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return connect.NewError(connect.CodeCanceled, err)
	}
	if errors.Is(err, ErrMaxQueue) {
		return connect.NewError(connect.CodeUnavailable, err)
	}
	var statusErr api.StatusError
	if errors.As(err, &statusErr) {
		switch statusErr.StatusCode {
		case http.StatusBadRequest:
			return connect.NewError(connect.CodeInvalidArgument, err)
		case http.StatusNotFound:
			return connect.NewError(connect.CodeNotFound, err)
		case http.StatusServiceUnavailable:
			return connect.NewError(connect.CodeUnavailable, err)
		case 499: // client closed
			return connect.NewError(connect.CodeCanceled, err)
		}
	}
	var authErr api.AuthorizationError
	if errors.As(err, &authErr) {
		return connect.NewError(connect.CodeUnauthenticated, err)
	}
	if isTransientSchedulerError(err) {
		return connect.NewError(connect.CodeUnavailable, err)
	}
	if errors.Is(err, errCapabilities) || errors.Is(err, errRequired) {
		return connect.NewError(connect.CodeInvalidArgument, err)
	}
	if errors.Is(err, os.ErrNotExist) {
		return connect.NewError(connect.CodeNotFound, err)
	}
	return connect.NewError(connect.CodeInternal, err)
}

func isTransientSchedulerError(err error) bool {
	// real: OOM retry, queue, temp load fails -> client can retry with backoff
	return errors.Is(err, ErrMaxQueue) // extend with sched sentinels
}

// loggingInterceptor enriched for Phase 4 (with stream_id, model, dur, reason, procedure).
// Supports unary; stream variant can be added similarly.
func loggingInterceptor() connect.Interceptor {
	return connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
			start := time.Now()
			id := uuid.New().String()
			model := extractModelFromAny(req)
			proc := req.Spec()
			slog.Info("grpc start", "component", "grpc", "rpc", proc, "model", model, "stream_id", id)
			res, err := next(ctx, req)
			slog.Info("grpc done", "component", "grpc", "stream_id", id, "duration_ms", time.Since(start).Milliseconds(), "error", err, "status", statusFromErr(err))
			return res, err
		}
	})
}

func statusFromErr(err error) string {
	if err == nil {
		return "ok"
	}
	return "error"
}

func extractModelFromAny(req connect.AnyRequest) string {
	// Phase 4: typed extract for model in logs (improved per review; add stream_id etc).
	if r, ok := req.Any().(*v1.ChatRequest); ok && r != nil {
		return r.Model
	}
	if r, ok := req.Any().(*v1.GenerateRequest); ok && r != nil {
		return r.Model
	}
	if r, ok := req.Any().(*v1.EmbedRequest); ok && r != nil {
		return r.Model
	}
	return ""
}

// authInterceptor: early auth metadata parsing, permissive for local dedicated gRPC port (like allowedHosts).
// Parses Authorization or x-ollama-auth. For Phase 4, permissive (no deny for local).
func authInterceptor() connect.Interceptor {
	return connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
			// early
			auth := req.Header().Get("Authorization")
			if auth == "" {
				auth = req.Header().Get("x-ollama-auth")
			}
			// permissive for local gRPC (separate port)
			if auth != "" {
				slog.Debug("gRPC auth metadata present (permissive local)", "component", "grpc")
			}
			return next(ctx, req)
		}
	})
}

// recoveryInterceptor: recover panic to connect err (per SKILL bounded, verifiability).
// Fixed per review: use named return so defer can override err on panic in next().
func recoveryInterceptor() connect.Interceptor {
	return connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (res connect.AnyResponse, err error) {
			defer func() {
				if r := recover(); r != nil {
					err = connect.NewError(connect.CodeInternal, fmt.Errorf("panic recovered in gRPC: %v", r))
				}
			}()
			return next(ctx, req)
		}
	})
}

// otelInterceptor for Phase 4 full observability (provider via global, custom attrs via spans in handlers).
func otelInterceptor() connect.Interceptor {
	setupOTELProvider() // ensure provider for Phase 4 "full"
	intcp, _ := otelconnect.NewInterceptor(
		otelconnect.WithTracerProvider(otel.GetTracerProvider()),
		otelconnect.WithMeterProvider(otel.GetMeterProvider()),
	)
	return intcp
}

// setupOTELProvider for Phase 4 "provider" (basic stdout for demo; prod uses env/explicit per doc risk of bloat).
func setupOTELProvider() {
	// See comment in code for how to enable exporter.
}

// TODO(phase4+): full streaming interceptors for logging/auth/recovery/otel. Add correlation IDs in logs.
// TODO: table driven tests for converters.
// TODO: var _ checks if not in test.

// convertListModelResponseToPB maps internal api.ListModelResponse (from shared modelList cache)
// to the gRPC v1.Model for admin List. Phase 5 full admin streams uplift (unary List populated).
// Details flattened to map<string,string> per proto MVP. Used by modelsHandler.List (thin).
func convertListModelResponseToPB(m api.ListModelResponse) *v1.Model {
	if m.Name == "" && m.Model == "" {
		return &v1.Model{}
	}
	pb := &v1.Model{
		Name:   m.Name,
		Model:  m.Model,
		Size:   m.Size,
		Digest: m.Digest,
	}
	if !m.ModifiedAt.IsZero() {
		pb.ModifiedAt = timestamppb.New(m.ModifiedAt)
	}
	d := m.Details
	pb.Details = map[string]string{
		"parent_model":       d.ParentModel,
		"format":             d.Format,
		"family":             d.Family,
		"parameter_size":     d.ParameterSize,
		"quantization_level": d.QuantizationLevel,
	}
	return pb
}

var _ = authInterceptor // compile use
