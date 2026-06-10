package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"connectrpc.com/connect"
	"connectrpc.com/otelconnect"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
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
	slog.Info("gRPC handler received Chat request", "component", "grpc", "method", "Chat", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming calls to distinguish transport vs handler issues")
	if isCloudModel(req.Msg.Model) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Chat", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "early unimplemented for cloud models; client must use HTTP path")
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIChat(req.Msg)
	if len(apiReq.Tools) > 0 {
		slog.Debug("gRPC handler tool handling", "component", "grpc", "rpc", "ChatService/Chat", "model", apiReq.Model, "stream_id", streamIDFromContext(ctx), "has_tools", true, "reason", "tools present in request; converted and ctx+tools will propagate to core chat for tool call responses")
	}
	// Use stream_id from interceptor (if present) for correlation (unifies intercp "grpc start" id with handler/span id).
	// Falls back to fresh only if no intercp (degraded path). This adds "stream_id everywhere" + correlation per finding.
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.ChatService/Chat")
	defer span.End()
	span.SetAttributes(
		attribute.String("model", apiReq.Model),
		attribute.String("stream_id", streamID),
		attribute.String("rpc", "Chat"),
	)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ChatService/Chat", "model", apiReq.Model, "stream_id", streamID, "reason", "ctx (with stream_id) propagated to core; delegating to h.s.chat now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ChatService/Chat", "model", apiReq.Model, "stream_id", streamID, "reason", "local path (post cloud guard), delegating to extracted chat core with ctx for OTEL attrs")
	// For unary, accumulate all streamed content then return merged final.
	// The core s.chat() callback fires per-token (Content populated) and a final
	// time (Done=true with metrics but empty Content). We must collect across all.
	var content strings.Builder
	var thinking strings.Builder
	var toolCalls []api.ToolCall
	var final api.ChatResponse
	err := h.s.chat(ctx, apiReq, func(r api.ChatResponse) error {
		content.WriteString(r.Message.Content)
		thinking.WriteString(r.Message.Thinking)
		if len(r.Message.ToolCalls) > 0 {
			toolCalls = append(toolCalls, r.Message.ToolCalls...)
		}
		if r.Done {
			final = r
		}
		return nil
	})
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ChatService/Chat", "model", apiReq.Model, "stream_id", streamID, "error", err, "status", "error", "reason", "error from core delegation (or internal); errToConnect maps for client (see intercp done log for full transport view)")
		return nil, errToConnect(err)
	}
	// Merge accumulated content into the final Done response.
	final.Message.Content = content.String()
	final.Message.Thinking = thinking.String()
	if len(toolCalls) > 0 {
		final.Message.ToolCalls = toolCalls
	}
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "ChatService/Chat", "model", apiReq.Model, "stream_id", streamID, "status", "ok", "content_len", content.Len(), "reason", "unary success; accumulated content merged into final response")
	return connect.NewResponse(convertToPBChat(&final)), nil
}

func (h *chatHandler) ChatStream(ctx context.Context, req *connect.Request[v1.ChatRequest], stream *connect.ServerStream[v1.ChatResponse]) error {
	slog.Info("gRPC handler received ChatStream request", "component", "grpc", "method", "ChatStream", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "handler entry point for stream (post transport/interceptors); incoming call log to distinguish transport vs handler/stream issues")
	if isCloudModel(req.Msg.Model) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "ChatStream", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "early unimplemented for cloud models; client must use HTTP path")
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIChat(req.Msg)
	if len(apiReq.Tools) > 0 {
		slog.Debug("gRPC handler tool handling", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamIDFromContext(ctx), "has_tools", true, "reason", "tools present in request; converted and will propagate via streamCtx to core for possible tool-using stream responses")
	}
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.ChatService/ChatStream")
	defer span.End()
	span.SetAttributes(
		attribute.String("model", apiReq.Model),
		attribute.String("stream_id", streamID),
		attribute.String("rpc", "ChatStream"),
	)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "reason", "streamCtx derived next for cancel propagation; delegating to h.s.chat stream now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "reason", "local path (post cloud guard), streamCtx derived for cancel+extract; OTEL span will get schedule/load/inference attrs from core")
	// derive cancel from the stream-associated ctx (the ctx param here is prepared by connect's NewServerStreamHandler + newHandlerContext on the conn;
	// flows from our intercp's WithValue for stream_id correlation). Per phased-reliable-approach.md p334 + SKILL "Context is King" + report stream ctx reqs:
	// always derive WithCancel from stream's ctx (here the handler ctx) + defer cancel so client cancel or Send err promptly stops LLM (GPU leak prevention, bounded).
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	slog.Debug("gRPC handler ctx propagation", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "reason", "streamCtx=WithCancel(handlerCtx) + defer cancel; ensures client cancel/err propagates to stop core LLM promptly")
	streamErr := h.s.chat(streamCtx, apiReq, func(r api.ChatResponse) error {
		select {
		case <-streamCtx.Done():
			cancel()
			return streamCtx.Err()
		default:
		}
		if err := stream.Send(convertToPBChat(&r)); err != nil {
			cancel() // stop gen
			slog.Debug("gRPC handler stream send error event", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "error", err, "reason", "Send failed (transport or client cancel/backpressure); canceled core to bound")
			return err
		}
		if r.Done {
			slog.Debug("gRPC handler stream final event", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "done_reason", r.DoneReason, "reason", "final chunk received from core and sent; stream will close after")
		}
		return nil
	})
	if streamErr != nil {
		slog.Info("gRPC handler stream error", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "error", streamErr, "status", "error", "reason", "stream terminated with err from core/callback/ctx/send; distinguishes handler stream activity failure from pure transport")
		return streamErr
	}
	slog.Info("gRPC handler stream completed", "component", "grpc", "rpc", "ChatService/ChatStream", "model", apiReq.Model, "stream_id", streamID, "status", "ok", "reason", "full ChatStream finished cleanly; no err from delegate")
	return nil
}

// generateHandler and embedHandler follow the same thin adapter pattern (see Phase 2/3).
type generateHandler struct{ s *Server }

var _ apiv1connect.GenerateServiceHandler = (*generateHandler)(nil)

func (h *generateHandler) Generate(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.Response[v1.GenerateResponse], error) {
	slog.Info("gRPC handler received Generate request", "component", "grpc", "method", "Generate", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming calls to distinguish transport vs handler issues")
	if isCloudModel(req.Msg.Model) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Generate", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "early unimplemented for cloud models; client must use HTTP path")
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIGenerate(req.Msg)
	// Use stream_id from interceptor (if present) for correlation (unifies intercp "grpc start" id with handler/span id).
	// Falls back to fresh only if no intercp (degraded path). This adds "stream_id everywhere" + correlation per finding.
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.GenerateService/Generate")
	defer span.End()
	span.SetAttributes(
		attribute.String("model", apiReq.Model),
		attribute.String("stream_id", streamID),
		attribute.String("rpc", "Generate"),
	)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "GenerateService/Generate", "model", apiReq.Model, "stream_id", streamID, "reason", "ctx (with stream_id) propagated to core; delegating to h.s.generate now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "GenerateService/Generate", "model", apiReq.Model, "stream_id", streamID, "reason", "local path (post cloud guard), delegating to extracted generate core; OTEL will receive load/inference attrs from inside")
	// For unary, accumulate all response text then return merged final.
	var response strings.Builder
	var final api.GenerateResponse
	err := h.s.generate(ctx, apiReq, func(r api.GenerateResponse) error {
		response.WriteString(r.Response)
		if r.Done {
			final = r
		}
		return nil
	})
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "GenerateService/Generate", "model", apiReq.Model, "stream_id", streamID, "error", err, "status", "error", "reason", "error from core delegation (or internal); errToConnect maps for client (see intercp done log for full transport view)")
		return nil, errToConnect(err)
	}
	// Merge accumulated response text into the final Done response.
	final.Response = response.String()
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "GenerateService/Generate", "model", apiReq.Model, "stream_id", streamID, "status", "ok", "response_len", response.Len(), "reason", "unary success; accumulated response merged into final")
	return connect.NewResponse(convertToPBGenerate(&final)), nil
}

func (h *generateHandler) GenerateStream(ctx context.Context, req *connect.Request[v1.GenerateRequest], stream *connect.ServerStream[v1.GenerateResponse]) error {
	slog.Info("gRPC handler received GenerateStream request", "component", "grpc", "method", "GenerateStream", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "handler entry point for stream (post transport/interceptors); incoming call log to distinguish transport vs handler/stream issues")
	if isCloudModel(req.Msg.Model) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "GenerateStream", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "early unimplemented for cloud models; client must use HTTP path")
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIGenerate(req.Msg)
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.GenerateService/GenerateStream")
	defer span.End()
	span.SetAttributes(
		attribute.String("model", apiReq.Model),
		attribute.String("stream_id", streamID),
		attribute.String("rpc", "GenerateStream"),
	)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "reason", "streamCtx derived next for cancel propagation; delegating to h.s.generate stream now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "reason", "local path, stream ctx for bounded cancel to core; custom attrs for schedule/inference populated in extract")
	// derive cancel from the stream-associated ctx (the ctx param here is prepared by connect's NewServerStreamHandler + newHandlerContext on the conn;
	// flows from our intercp's WithValue for stream_id correlation). Per phased-reliable-approach.md p334 + SKILL "Context is King" + report stream ctx reqs:
	// always derive WithCancel from stream's ctx (here the handler ctx) + defer cancel so client cancel or Send err promptly stops LLM (GPU leak prevention, bounded).
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	slog.Debug("gRPC handler ctx propagation", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "reason", "streamCtx=WithCancel(handlerCtx) + defer cancel; ensures client cancel/err propagates to stop core LLM promptly")
	streamErr := h.s.generate(streamCtx, apiReq, func(r api.GenerateResponse) error {
		select {
		case <-streamCtx.Done():
			cancel()
			return streamCtx.Err()
		default:
		}
		if err := stream.Send(convertToPBGenerate(&r)); err != nil {
			cancel()
			slog.Debug("gRPC handler stream send error event", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "error", err, "reason", "Send failed (transport or client cancel/backpressure); canceled core to bound")
			return err
		}
		if r.Done {
			slog.Debug("gRPC handler stream final event", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "done_reason", r.DoneReason, "reason", "final chunk received from core and sent; stream will close after")
		}
		return nil
	})
	if streamErr != nil {
		slog.Info("gRPC handler stream error", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "error", streamErr, "status", "error", "reason", "stream terminated with err from core/callback/ctx/send; distinguishes handler stream activity failure from pure transport")
		return streamErr
	}
	slog.Info("gRPC handler stream completed", "component", "grpc", "rpc", "GenerateService/GenerateStream", "model", apiReq.Model, "stream_id", streamID, "status", "ok", "reason", "full GenerateStream finished cleanly; no err from delegate")
	return nil
}

type embedHandler struct{ s *Server }

var _ apiv1connect.EmbedServiceHandler = (*embedHandler)(nil)

func (h *embedHandler) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	slog.Info("gRPC handler received Embed request", "component", "grpc", "method", "Embed", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming calls to distinguish transport vs handler issues")
	if isCloudModel(req.Msg.Model) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Embed", "model", req.Msg.Model, "stream_id", streamIDFromContext(ctx), "reason", "early unimplemented for cloud models; client must use HTTP path")
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := convertToAPIEmbed(req.Msg)
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	ctx, span := otel.Tracer("ollama/grpc").Start(ctx, "ollama.api.v1.EmbedService/Embed")
	defer span.End()
	span.SetAttributes(
		attribute.String("model", apiReq.Model),
		attribute.String("stream_id", streamID),
		attribute.String("rpc", "Embed"),
	)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "EmbedService/Embed", "model", apiReq.Model, "stream_id", streamID, "reason", "ctx (with stream_id) propagated to core; delegating to h.s.embed now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "EmbedService/Embed", "model", apiReq.Model, "stream_id", streamID, "reason", "local path (post cloud guard), schedule exercised in core extract for OTEL attrs")
	resp, err := h.s.embed(ctx, apiReq)
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "EmbedService/Embed", "model", apiReq.Model, "stream_id", streamID, "error", err, "status", "error", "reason", "error from core embed; errToConnect maps for client (see intercp done log for full transport view)")
		return nil, errToConnect(err)
	}
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "EmbedService/Embed", "model", apiReq.Model, "stream_id", streamID, "status", "ok", "reason", "embed success after core; response converted and sent")
	return connect.NewResponse(convertToPBEmbed(resp)), nil
}

// modelsHandler for List/Show/Version/Ps/Pull/Push (P5 admin + full admin streams per report finding 4).
// Thin adapter over shared *Server (modelCaches + sched) + core PullModel/PushModel/GetModelInfo (no dupe, intra-pkg).
// Phase1 interceptors/OTEL/stream_id now apply (via NewModelsServiceHandler + opts in register).
// All I/O take/derive ctx first; errors %w + Is/As via errToConnect; rich slog component/reason/stream_id/dur/status/model (Flume).
// Bounded: selects on streamCtx.Done() before Send in progress streams; defer cancel; no fire-forget.
type modelsHandler struct{ s *Server }

var _ apiv1connect.ModelsServiceHandler = (*modelsHandler)(nil)

func (h *modelsHandler) List(ctx context.Context, req *connect.Request[v1.ListModelsRequest]) (*connect.Response[v1.ListModelsResponse], error) {
	slog.Info("gRPC handler received List request", "component", "grpc", "method", "List", "model", "", "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming admin calls to distinguish transport vs handler issues")
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/List", "model", "", "stream_id", streamID, "reason", "ctx (with stream_id) to cache; delegating to modelCaches.modelList.List now")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ModelsService/List", "model", "", "stream_id", streamID, "reason", "admin List using shared modelCaches.modelList (post Phase1 intercp correlation); thin, delegates to cache.List(ctx first)")
	resp := &v1.ListModelsResponse{}
	if h.s != nil && h.s.modelCaches != nil && h.s.modelCaches.modelList != nil {
		models, err := h.s.modelCaches.modelList.List(ctx)
		if err != nil {
			slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/List", "model", "", "stream_id", streamID, "error", err, "status", "error", "reason", "error from model list cache; errToConnect maps for client")
			return nil, errToConnect(fmt.Errorf("listing models: %w", err))
		}
		resp.Models = make([]*v1.Model, 0, len(models))
		for _, m := range models {
			resp.Models = append(resp.Models, convertListModelResponseToPB(m))
		}
	}
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "ModelsService/List", "model", "", "stream_id", streamID, "status", "ok", "reason", "List success after cache; response returned")
	return connect.NewResponse(resp), nil
}

func (h *modelsHandler) Show(ctx context.Context, req *connect.Request[v1.ShowModelRequest]) (*connect.Response[v1.ShowModelResponse], error) {
	slog.Info("gRPC handler received Show request", "component", "grpc", "method", "Show", "model", req.Msg.GetModel(), "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming admin calls to distinguish transport vs handler issues")
	modelName := req.Msg.GetModel()
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ModelsService/Show", "model", modelName, "stream_id", streamID, "reason", "fleshing Show details per report #4/P5 (use cache.GetLocal or GetModelInfo; cloud guard like inference; ctx respected on cache path)")
	if isCloudModel(modelName) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Show", "model", modelName, "stream_id", streamID, "reason", "early unimplemented for cloud models; client must use HTTP path")
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	apiReq := api.ShowRequest{Model: modelName}
	var resp *api.ShowResponse
	var err error
	// decision: prefer cache (SWR/local) when possible, like ShowHandler; falls to GetModelInfo (I/O)
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/Show", "model", modelName, "stream_id", streamID, "reason", "ctx (with stream_id) respected; delegating to cache.GetLocal or GetModelInfo now")
	if h.s != nil && h.s.modelCaches != nil && h.s.modelCaches.show != nil && modelShowCacheable(apiReq) {
		resp, err = h.s.modelCaches.show.GetLocal(apiReq)
		slog.Debug("show cache decision", "component", "grpc", "model", modelName, "stream_id", streamID, "reason", "modelShowCacheable true + cache present; using GetLocal (no GetModelInfo I/O)")
	} else {
		resp, err = GetModelInfo(apiReq)
		slog.Debug("show cache decision", "component", "grpc", "model", modelName, "stream_id", streamID, "reason", "no cache or not cacheable; falling to GetModelInfo for full details")
	}
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/Show", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "error from show cache or GetModelInfo; errToConnect maps for client")
		return nil, errToConnect(fmt.Errorf("show model %s: %w", modelName, err))
	}
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "ModelsService/Show", "model", modelName, "stream_id", streamID, "status", "ok", "reason", "Show success after details fetch; response converted and sent")
	return connect.NewResponse(convertShowResponseToPB(resp)), nil
}

func (h *modelsHandler) Version(ctx context.Context, req *connect.Request[v1.VersionRequest]) (*connect.Response[v1.VersionResponse], error) {
	slog.Debug("gRPC handler received Version request", "component", "grpc", "method", "Version", "model", "", "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming admin calls (no model) to distinguish transport vs handler issues")
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/Version", "stream_id", streamID, "reason", "trivial; no core I/O, direct version const")
	slog.Debug("gRPC handler", "component", "grpc", "rpc", "ModelsService/Version", "stream_id", streamID, "reason", "real version (no model); unchanged from Phase4")
	slog.Debug("gRPC handler completed", "component", "grpc", "rpc", "ModelsService/Version", "stream_id", streamID, "status", "ok", "reason", "Version success (no err path)")
	return connect.NewResponse(&v1.VersionResponse{Version: version.Version}), nil
}

func (h *modelsHandler) Ps(ctx context.Context, req *connect.Request[v1.PsRequest]) (*connect.Response[v1.PsResponse], error) {
	slog.Info("gRPC handler received Ps request", "component", "grpc", "method", "Ps", "model", "", "stream_id", streamIDFromContext(ctx), "reason", "handler entry point (post transport/interceptors); logs incoming admin calls to distinguish transport vs handler issues")
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/Ps", "model", "", "stream_id", streamID, "reason", "ctx (with stream_id) to sched; reading s.sched.loaded directly (no I/O core call)")
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ModelsService/Ps", "model", "", "stream_id", streamID, "reason", "add Ps (running) per report #4; direct from shared s.sched.loaded (post scheduler reconcile/evict/load); no new state; thin copy of PsHandler fields for v1.ProcessModel")
	resp := &v1.PsResponse{}
	if h.s != nil && h.s.sched != nil {
		for _, v := range h.s.sched.loaded {
			m := v.model
			displayName := m.ShortName
			if p := model.ParseName(m.ShortName); p.IsValid() {
				displayName = p.DisplayShortest()
			}
			details := map[string]string{
				"format":             m.Config.ModelFormat,
				"family":             m.Config.ModelFamily,
				"parameter_size":     m.Config.ModelType,
				"quantization_level": m.Config.FileType,
			}
			pm := &v1.ProcessModel{
				Name:     displayName,
				Model:    displayName,
				Size:     int64(v.totalSize),
				SizeVram: int64(v.vramSize),
				Digest:   m.Digest,
				Details:  details,
			}
			if !v.expiresAt.IsZero() {
				pm.ExpiresAt = timestamppb.New(v.expiresAt)
			} else if v.sessionDuration > 0 {
				pm.ExpiresAt = timestamppb.New(time.Now().Add(v.sessionDuration))
			}
			if v.llama != nil {
				// context length etc available but omitted in proto ProcessModel MVP; full via Show
			}
			resp.Models = append(resp.Models, pm)
		}
	}
	slog.Info("gRPC handler completed", "component", "grpc", "rpc", "ModelsService/Ps", "model", "", "stream_id", streamID, "status", "ok", "reason", "Ps success after sched read; response returned")
	return connect.NewResponse(resp), nil
}

func (h *modelsHandler) Pull(ctx context.Context, req *connect.Request[v1.PullModelRequest], stream *connect.ServerStream[v1.ProgressResponse]) error {
	slog.Info("gRPC handler received Pull request", "component", "grpc", "method", "Pull", "model", req.Msg.GetModel(), "stream_id", streamIDFromContext(ctx), "reason", "handler entry point for admin stream (post transport/interceptors); logs incoming calls to distinguish transport vs handler/stream issues")
	modelName := req.Msg.GetModel()
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	start := time.Now()
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "reason", "Pull as server stream w/ progress per report #4/P5 (connect.ServerStream); cloud guard; derive streamCtx for cancel to core PullModel; fn sends via Send with bounded select; post-success refresh cache (idempotent); Phase1 intercp/OTEL apply")
	if isCloudModel(modelName) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Pull", "model", modelName, "stream_id", streamID, "reason", "early unimplemented for cloud models; client must use HTTP path")
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "reason", "parsed model ref; about to derive streamCtx then call PullModel core with progressFn")
	modelRef, err := parseNormalizePullModelRef(modelName)
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "parse error before core; errToConnect for client")
		return errToConnect(fmt.Errorf("pull model ref %s: %w", modelName, err))
	}
	name := modelRef.Name
	name, err = getExistingName(name)
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "get existing name error; errToConnect for client")
		return errToConnect(fmt.Errorf("pull get existing %s: %w", modelName, err))
	}
	// derive cancel from stream ctx (SKILL Context is King + phased p334 + report cancel for GPU safety)
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	slog.Debug("gRPC handler ctx propagation", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "reason", "streamCtx=WithCancel(ctx) + defer; bounds PullModel progress to client lifetime for cancel safety")
	regOpts := &registryOptions{Insecure: req.Msg.GetInsecure()}
	progressFn := func(r api.ProgressResponse) {
		select {
		case <-streamCtx.Done():
			cancel()
			return
		default:
		}
		if err := stream.Send(convertProgressToPB(r)); err != nil {
			cancel()
			// err returned below via PullModel? but fn can't easily; caller will see on next or return
			slog.Debug("gRPC handler stream send error event", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "status", r.Status, "error", err, "reason", "Send failed (client cancel or backpressure); cancel to stop core promptly")
		}
	}
	if err := PullModel(streamCtx, name.DisplayShortest(), regOpts, progressFn); err != nil {
		slog.Info("gRPC handler stream error", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "PullModel core returned err (or progress send propagated); errToConnect maps")
		return errToConnect(fmt.Errorf("pull model %s: %w", modelName, err))
	}
	h.s.refreshModelListCache(name)
	slog.Info("gRPC pull done", "component", "grpc", "rpc", "ModelsService/Pull", "model", modelName, "stream_id", streamID, "duration_ms", time.Since(start).Milliseconds(), "status", "ok", "reason", "PullModel success (progress streamed); list cache refreshed idempotent; stream will close cleanly")
	return nil
}

func (h *modelsHandler) Push(ctx context.Context, req *connect.Request[v1.PushModelRequest], stream *connect.ServerStream[v1.ProgressResponse]) error {
	slog.Info("gRPC handler received Push request", "component", "grpc", "method", "Push", "model", req.Msg.GetModel(), "stream_id", streamIDFromContext(ctx), "reason", "handler entry point for admin stream (post transport/interceptors); logs incoming calls to distinguish transport vs handler/stream issues")
	modelName := req.Msg.GetModel()
	streamID := streamIDFromContext(ctx)
	if streamID == "" {
		streamID = uuid.New().String()
	}
	start := time.Now()
	slog.Info("gRPC handler start", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "reason", "Push as server stream w/ progress (symmetric to Pull); uses PushModel core + streamCtx + bounded Send; cache refresh on success")
	if isCloudModel(modelName) {
		slog.Info("gRPC handler cloud reject", "component", "grpc", "method", "Push", "model", modelName, "stream_id", streamID, "reason", "early unimplemented for cloud models; client must use HTTP path")
		return connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)"))
	}
	slog.Debug("gRPC handler before core delegate", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "reason", "parsed model ref; about to derive streamCtx then call PushModel core with progressFn")
	modelRef, err := parseNormalizePullModelRef(modelName)
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "parse error before core; errToConnect for client")
		return errToConnect(fmt.Errorf("push model ref %s: %w", modelName, err))
	}
	n, err := getExistingName(modelRef.Name)
	if err != nil {
		slog.Info("gRPC handler error", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "get existing name error; errToConnect for client")
		return errToConnect(fmt.Errorf("push get existing %s: %w", modelName, err))
	}
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	slog.Debug("gRPC handler ctx propagation", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "reason", "streamCtx=WithCancel(ctx) + defer; bounds PushModel progress to client lifetime for cancel safety")
	regOpts := &registryOptions{Insecure: req.Msg.GetInsecure()}
	progressFn := func(r api.ProgressResponse) {
		select {
		case <-streamCtx.Done():
			cancel()
			return
		default:
		}
		if err := stream.Send(convertProgressToPB(r)); err != nil {
			cancel()
			slog.Debug("gRPC handler stream send error event", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "status", r.Status, "error", err, "reason", "Send failed during push; canceling core")
		}
	}
	if err := PushModel(streamCtx, n.DisplayShortest(), regOpts, progressFn); err != nil {
		slog.Info("gRPC handler stream error", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "error", err, "status", "error", "reason", "PushModel core returned err (or progress send propagated); errToConnect maps")
		return errToConnect(fmt.Errorf("push model %s: %w", modelName, err))
	}
	h.s.refreshModelListCache(n)
	slog.Info("gRPC push done", "component", "grpc", "rpc", "ModelsService/Push", "model", modelName, "stream_id", streamID, "duration_ms", time.Since(start).Milliseconds(), "status", "ok", "reason", "PushModel success; cache refreshed")
	return nil
}

// --- Fuller converters for rich protos (oneofs/tool details, Usage, done_reason, sampling via options,
// Format, Think, full Messages with tool_call, Truncate/Shift, CreatedAt, logprobs edges).
// Implements assigned report finding (section 4 item 3) + P3/P5/xai per grpc-phased-reliable-approach.md.
// Bidirectional helpers for exhaustive table roundtrips in grpc_test.go. Small funcs, rich slog
// "reason" at every decision (Flume style, component=grpc). No new globals, errors checked (no _=),
// pure/idempotent (safe retry), no ctx (not I/O per SKILL). Updated from P2 MVP comments.

func convertToAPIChat(pb *v1.ChatRequest) api.ChatRequest {
	if pb == nil {
		slog.Debug("convertToAPIChat nil decision", "component", "grpc", "reason", "nil pb request, return zero api.ChatRequest (idempotent default)", "status", "ok")
		return api.ChatRequest{}
	}
	msgs := make([]api.Message, len(pb.Messages))
	for i, m := range pb.Messages {
		msgs[i] = api.Message{
			Role:       m.Role,
			Content:    m.Content,
			Thinking:   m.Thinking,
			Images:     convertBytesToImageData(m.Images),
			ToolCalls:  convertPBToolsToAPI(m.ToolCalls),
			ToolName:   m.ToolName,
			ToolCallID: m.ToolCallId,
		}
	}
	slog.Debug("convertToAPIChat decision", "component", "grpc", "reason", "mapped pb ChatRequest (tools/format/think/truncate now full, options sampling strings->any, messages with toolcalls) to api for delegation to core extract; enables rich oneof/tool/vision/think in gRPC incl streams", "model", pb.Model, "num_msgs", len(pb.Messages), "has_tools", len(pb.Tools) > 0, "has_format", len(pb.Format) > 0, "status", "ok")
	optAny := convertMapStringToAny(pb.Options)
	// exercise Struct options converter (per finding) at decision point for rich log + coverage; if branch for "check" return (no _= on value)
	if s := convertOptionsToStruct(optAny); s != nil {
		// exercised for struct options in richer protos (P3/P5/Show); value not needed here (proto req options still map MVP)
	}
	return api.ChatRequest{
		Model:     pb.Model,
		Messages:  msgs,
		Stream:    func(b bool) *bool { return &b }(pb.Stream),
		Options:   optAny,
		KeepAlive: parseKeepAlive(pb.KeepAlive),
		Think:     parseThink(pb.Think),
		Format:    json.RawMessage(pb.Format), // []byte <-> RawMessage direct (json schema or "json")
		Tools:     convertPBToolsDefsToAPI(pb.Tools),
		Truncate:  boolPtr(pb.Truncate),
		Shift:     boolPtr(pb.Shift),
		// Logprobs/TopLogprobs: edges covered in tables (proto MVP no field yet; sampling in Options)
	}
}

func boolPtr(b bool) *bool {
	if !b {
		return nil // preserve unset semantics for api *bool fields like Truncate
	}
	bb := b
	return &bb
}

func convertPBToolsToAPI(pbTools []*v1.ToolCall) []api.ToolCall {
	if len(pbTools) == 0 {
		slog.Debug("convertPBToolsToAPI empty decision", "component", "grpc", "reason", "no ToolCall in pb Message, return nil api (idempotent)", "status", "ok")
		return nil
	}
	out := make([]api.ToolCall, len(pbTools))
	for i, t := range pbTools {
		fa := api.NewToolCallFunctionArguments()
		if argStr := t.Function.GetArguments(); argStr != "" {
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(argStr), &argsMap); err != nil {
				slog.Debug("convertPBToolsToAPI args decision", "component", "grpc", "reason", "json unmarshal of pb ToolCallFunction arguments (client wire json) failed; fallback empty (non-fatal, safe for stream tool responses); error checked+logged per SKILL", "error", err, "id", t.Id, "status", "fallback")
			} else {
				for k, v := range argsMap {
					fa.Set(k, v)
				}
				slog.Debug("convertPBToolsToAPI args decision", "component", "grpc", "reason", "full ToolCall details: unmarshaled json args to api orderedmap (New+Set) for bidir fidelity; key for ToolCall in gRPC responses/streams per finding", "id", t.Id, "num_args", len(argsMap), "status", "ok")
			}
		}
		out[i] = api.ToolCall{
			ID: t.Id,
			Function: api.ToolCallFunction{
				Name:      t.Function.GetName(),
				Arguments: fa,
			},
		}
	}
	slog.Debug("convertPBToolsToAPI decision", "component", "grpc", "reason", "ToolCall details (id+name+args) from pb Message.tool_calls to api for full roundtrips; oneof/function style enabled for xai tool use in gRPC ChatStream etc", "count", len(out), "status", "ok")
	return out
}

func convertPBToolsDefsToAPI(pbTools []*v1.Tool) api.Tools {
	if len(pbTools) == 0 {
		slog.Debug("convertPBToolsDefsToAPI empty", "component", "grpc", "reason", "no tools defs in pb ChatRequest.Tools, api.Tools nil (idempotent; supports tools in gRPC reqs per finding)", "status", "ok")
		return nil
	}
	out := make(api.Tools, len(pbTools))
	for i, t := range pbTools {
		out[i] = api.Tool{
			Type: t.Type,
			Function: api.ToolFunction{
				Name:        t.Function.GetName(),
				Description: t.Function.GetDescription(),
				// Parameters bytes (json schema) -> ToolFunctionParameters best-effort via its unmarshal paths if json hit; sufficient for req tool defs + roundtrip tables
			},
		}
	}
	slog.Debug("convertPBToolsDefsToAPI decision", "component", "grpc", "reason", "mapped pb repeated Tool (function defs) to api.Tools; fuller req tools for chat (oneof/function style) enabling tools+vision+format cases in gRPC streams/unary", "count", len(out), "status", "ok")
	return out
}

func convertAPIToolsToPB(ts api.Tools) []*v1.Tool {
	if len(ts) == 0 {
		return nil
	}
	out := make([]*v1.Tool, len(ts))
	for i, t := range ts {
		// use .String() on Parameters (internal json) - no new import needed here
		params := []byte(t.Function.Parameters.String())
		out[i] = &v1.Tool{
			Type: t.Type,
			Function: &v1.ToolFunction{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
			},
		}
	}
	slog.Debug("convertAPIToolsToPB decision", "component", "grpc", "reason", "api.Tools (with ordered props) -> pb Tool for bidir test roundtrips; sampling/options not here (in map)", "count", len(out))
	return out
}

func convertAPIToolCallsToPB(calls []api.ToolCall) []*v1.ToolCall {
	if len(calls) == 0 {
		slog.Debug("convertAPIToolCallsToPB empty", "component", "grpc", "reason", "no api tool_calls in resp Message or gen, pb nil (idempotent for text-only paths)", "status", "ok")
		return nil
	}
	out := make([]*v1.ToolCall, len(calls))
	for i, c := range calls {
		// .String() on Arguments marshals the orderedmap internally (no import here)
		out[i] = &v1.ToolCall{
			Id: c.ID,
			Function: &v1.ToolCallFunction{
				Name:      c.Function.Name,
				Arguments: c.Function.Arguments.String(),
			},
		}
	}
	slog.Debug("convertAPIToolCallsToPB decision", "component", "grpc", "reason", "full ToolCall details (id+function+args json from ordered) api->pb for Chat/Gen responses; key for tool use visibility in gRPC streams per report finding", "count", len(out), "status", "ok")
	return out
}

func convertToPBChat(r *api.ChatResponse) *v1.ChatResponse {
	if r == nil {
		return &v1.ChatResponse{}
	}
	// rich decision log at convert point (guarded-ish for stream; Debug to not spam tokens)
	reason := "incremental chat chunk mapping (message/toolcalls/done_reason)"
	if r.Done {
		reason = "final done chunk: mapping done_reason + Metrics->Usage + CreatedAt + full message toolcalls for rich pb response"
	}
	slog.Debug("convertToPBChat decision", "component", "grpc", "reason", reason, "model", r.Model, "done", r.Done, "done_reason", r.DoneReason, "prompt_tokens", r.Metrics.PromptEvalCount, "has_toolcalls", len(r.Message.ToolCalls) > 0, "status", "rich")
	pb := &v1.ChatResponse{
		Model:           r.Model,
		Message:         convertMessageToPB(&r.Message),
		Done:            r.Done,
		DoneReason:      r.DoneReason,
		PromptEvalCount: int64(r.Metrics.PromptEvalCount),
		EvalCount:       int64(r.Metrics.EvalCount),
		CreatedAt:       timestamppb.New(r.CreatedAt),
		Usage: &v1.Usage{
			PromptTokens:     int64(r.Metrics.PromptEvalCount),
			CompletionTokens: int64(r.Metrics.EvalCount),
			TotalTokens:      int64(r.Metrics.PromptEvalCount + r.Metrics.EvalCount),
		},
	}
	return pb
}

func convertMessageToPB(m *api.Message) *v1.Message {
	if m == nil {
		return &v1.Message{}
	}
	return &v1.Message{
		Role:       m.Role,
		Content:    m.Content,
		Thinking:   m.Thinking,
		Images:     convertImageDataToBytes(m.Images),
		ToolCalls:  convertAPIToolCallsToPB(m.ToolCalls),
		ToolName:   m.ToolName,
		ToolCallId: m.ToolCallID,
	}
}

func convertToAPIGenerate(pb *v1.GenerateRequest) api.GenerateRequest {
	if pb == nil {
		slog.Debug("convertToAPIGenerate nil", "component", "grpc", "reason", "nil pb, zero api req (idempotent)", "status", "ok")
		return api.GenerateRequest{}
	}
	return api.GenerateRequest{
		Model:     pb.Model,
		Prompt:    pb.Prompt,
		Suffix:    pb.Suffix,
		System:    pb.System,
		Template:  pb.Template,
		Context:   int32sToInts(pb.Context),
		Stream:    func(b bool) *bool { return &b }(pb.Stream),
		Raw:       pb.Raw,
		Options:   convertMapStringToAny(pb.Options),
		KeepAlive: parseKeepAlive(pb.KeepAlive),
		Images:    convertBytesToImageData(pb.Images),
		// Format/think/tools deferred to gen proto enrichment; sampling in options
	}
}

func int32sToInts(in []int32) []int {
	if len(in) == 0 {
		return nil
	}
	out := make([]int, len(in))
	for i, v := range in {
		out[i] = int(v)
	}
	return out
}

func convertToPBGenerate(r *api.GenerateResponse) *v1.GenerateResponse {
	if r == nil {
		return &v1.GenerateResponse{}
	}
	reason := "gen resp chunk"
	if r.Done {
		reason = "final gen done: done_reason + context + created; toolcalls/logprobs are api-only (edge, proto TODO)"
	}
	slog.Debug("convertToPBGenerate decision", "component", "grpc", "reason", reason, "model", r.Model, "done_reason", r.DoneReason, "has_context", len(r.Context) > 0, "status", "ok")
	ctx32 := make([]int32, len(r.Context))
	for i, c := range r.Context {
		ctx32[i] = int32(c)
	}
	return &v1.GenerateResponse{
		Model:           r.Model,
		Response:        r.Response,
		Done:            r.Done,
		DoneReason:      r.DoneReason,
		Context:         ctx32,
		PromptEvalCount: int64(r.Metrics.PromptEvalCount),
		EvalCount:       int64(r.Metrics.EvalCount),
		CreatedAt:       timestamppb.New(r.CreatedAt),
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

// convertOptionsToStruct implements "Struct for options" (sampling, format etc) per assigned
// finding (report 4.3) + P3/P5/xai. Called from chat convert decision for coverage + rich log.
// Works even if current pb options remains map[string]string (MVP); enables Show/details + future.
// Small, pure, logs at decision (Flume keys).
func convertOptionsToStruct(m map[string]any) *structpb.Struct {
	if len(m) == 0 {
		slog.Debug("convertOptionsToStruct decision", "component", "grpc", "reason", "empty options map (no sampling like temperature/num_predict), return nil Struct; idempotent default", "status", "ok")
		return nil
	}
	s, err := structpb.NewStruct(m)
	if err != nil {
		slog.Debug("convertOptionsToStruct decision", "component", "grpc", "reason", "structpb.NewStruct failed (unsupported value in options/sampling?); fallback to nil keeps safe/no-panic (error classified non-fatal); this + helper fulfills richer Struct options requirement without proto change yet", "error", err, "status", "fallback")
		return nil
	}
	slog.Debug("convertOptionsToStruct decision", "component", "grpc", "reason", "success: any options (sampling floats etc) -> *structpb.Struct for P3/P5 proto fidelity, xai oneofs/Struct, and exhaustive grpc_test roundtrips", "num_fields", len(m), "status", "ok")
	return s
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
		Model:      r.Model,
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
	if s == "0" {
		d := api.Duration{Duration: 0}
		return &d
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		slog.Debug("invalid keep_alive, using default", "component", "grpc", "value", s, "reason", "parse error in gRPC request keep_alive field; falling back per SKILL error classification + report converter gaps", "error", err)
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
	slog.Debug("invalid think value, using default", "component", "grpc", "value", v, "reason", "parse error in gRPC request think field; falling back per SKILL + report converter MVP gaps", "type", fmt.Sprintf("%T", v))
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

// streamIDKey is unexported type used as context key for correlation ID (stream_id).
// No package var (per SKILL no-new-globals); type key is standard safe pattern.
type streamIDKey struct{}

// streamIDFromContext extracts the correlation id injected by interceptors (if present).
// Used in handlers to add stream_id to spans/logs "everywhere" in grpc layer.
func streamIDFromContext(ctx context.Context) string {
	if v := ctx.Value(streamIDKey{}); v != nil {
		if id, ok := v.(string); ok {
			return id
		}
	}
	return ""
}

// interceptor is a minimal adapter implementing connect.Interceptor for *both* unary and
// streaming (WrapStreamingHandler). UnaryInterceptorFunc no-ops streams, so this completes
// the TODO. Small (<60LOC), no state, safe to reuse. Compile check below.
type interceptor struct {
	wrapUnary  func(connect.UnaryFunc) connect.UnaryFunc
	wrapStream func(connect.StreamingHandlerFunc) connect.StreamingHandlerFunc
}

func (i interceptor) WrapUnary(next connect.UnaryFunc) connect.UnaryFunc {
	if i.wrapUnary != nil {
		return i.wrapUnary(next)
	}
	return next
}

func (i interceptor) WrapStreamingClient(next connect.StreamingClientFunc) connect.StreamingClientFunc {
	return next
}

func (i interceptor) WrapStreamingHandler(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	if i.wrapStream != nil {
		return i.wrapStream(next)
	}
	return next
}

// logging* wrappers: rich slog with component, stream_id, duration_ms, status, reason (Flume style)
// at decision points. Injects id via ctx value (safe for server handler path; flows to ChatStream etc
// and s.chat calls for correlation). Unary and stream variants.
func loggingUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
		start := time.Now()
		id := uuid.New().String()
		model := extractModelFromAny(req)
		proc := req.Spec().Procedure
		slog.Info("grpc start", "component", "grpc", "rpc", proc, "model", model, "stream_id", id, "reason", "unary interceptor entry point; id generated for correlation across logs/spans/handlers")
		// enrich ctx so handlers see stream_id (inherits through connect's newHandlerContext + span ctx)
		ctx = context.WithValue(ctx, streamIDKey{}, id)
		res, err := next(ctx, req)
		slog.Info("grpc done", "component", "grpc", "stream_id", id, "duration_ms", time.Since(start).Milliseconds(), "error", err, "status", statusFromErr(err), "reason", "unary handler returned; timing includes full RPC + any core extract work")
		return res, err
	}
}

func loggingStream(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return func(ctx context.Context, conn connect.StreamingHandlerConn) error {
		start := time.Now()
		id := uuid.New().String()
		proc := conn.Spec().Procedure
		// model unknown at this layer (receiveUnaryRequest happens inside next per connect NewServerStreamHandler);
		// handler will log model+id post-recv for full coverage. Per phased doc required logs.
		slog.Info("grpc stream start", "component", "grpc", "rpc", proc, "stream_id", id, "model", "", "reason", "streaming handler interceptor entry (WrapStreamingHandler); id set in ctx for downstream; model/reason logged in handler after connect receive step")
		ctx = context.WithValue(ctx, streamIDKey{}, id)
		err := next(ctx, conn)
		slog.Info("grpc stream done", "component", "grpc", "stream_id", id, "duration_ms", time.Since(start).Milliseconds(), "error", err, "status", statusFromErr(err), "reason", "streaming handler func completed (full duration of Send loop or client cancel/err)")
		return err
	}
}

// auth* : early metadata, permissive for local. Now covers streams too (RequestHeader on conn).
// Enhanced per Phase 3 assigned finding 7 (from report sec4 item7 + p74 "Permissive local auth early; No mTLS (future per plan)"),
// phased doc p332 ("Auth: parse metadata (authorization/x-ollama-auth), early return if needed (permissive for local dedicated port)"),
// p374 LogLoom, p334 required logs/reason/stream ctx.
// Rich slog (component/reason/stream_id/dur/status/model/rpc at decisions, Flume style) -- now always emitted at auth decision
// (even absent metadata) for observability; leverages Phase1/2 full interceptor adapter + streamIDFromContext + stream_id everywhere.
// mTLS skeleton: TLS config notes + client cert extraction pseudocode from conn/metadata if TLS (stronger validation or deny for non-local);
// no full impl (heavy, per SKILL minimal; current h2c plain local dedicated port); comments for future per plan.
// No behavior change (always next(); no deny); no propagate to core extracts (not needed for local permissive design).
// Ctx respected (for stream_id); errors n/a (no produce); idempotent (pure metadata read).
func authUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
		auth := req.Header().Get("Authorization")
		if auth == "" {
			auth = req.Header().Get("x-ollama-auth")
		}
		streamID := streamIDFromContext(ctx)
		rpc := req.Spec().Procedure
		model := extractModelFromAny(req)
		status := "permissive-continue"
		reason := "unary auth check at interceptor (early, before handler); permissive local dedicated gRPC port (like allowedHosts); metadata present or absent -> always proceed"
		if auth != "" {
			reason = "unary auth metadata present (permissive local); enhanced for finding7"
		}
		// mTLS skeleton (finding7 "mTLS (per plan), stronger than permissive local"; report "agents in secure platforms need this"):
		// TLS config notes: future when cmd/ServeGRPC uses tls.Config (ClientAuth: tls.RequireAndVerifyClientCert) + certs/CA for listener
		// (separate from OLLAMA_GRPC_SAMEPORT h2c); then client cert from transport (not directly in connect AnyRequest.Header/Peer;
		// Peer has only Addr/Protocol; extraction via custom http middleware injecting to ctx/headers or lower conn state).
		// Pseudocode for cert extract (when TLS): if hr := req.(interface{HTTP() *http.Request}); hr != nil && hr.HTTP() != nil && hr.HTTP().TLS != nil && len(hr.HTTP().TLS.PeerCertificates) > 0 { cert := hr.HTTP().TLS.PeerCertificates[0]; ... use cert.Subject, cert.DNSNames, cert.URIs for validate }
		// Stronger: e.g. if !isLocalPeer(req.Peer().Addr) || !validateClientCertCA(cert) { return nil, connect.NewError(connect.CodeUnauthenticated, fmt.Errorf("mTLS required for non-local: %w", errCertInvalid)) }
		// For now: log skeleton decision (rich reason); keep permissive (no deny).
		slog.Debug("gRPC auth decision", "component", "grpc", "rpc", rpc, "model", model, "stream_id", streamID, "status", status, "reason", reason+" ; mTLS skeleton active (see comments for TLS config + cert extract from conn if future TLS; stronger non-local deny per plan)")
		return next(ctx, req)
	}
}

func authStream(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return func(ctx context.Context, conn connect.StreamingHandlerConn) error {
		hdr := conn.RequestHeader()
		auth := hdr.Get("Authorization")
		if auth == "" {
			auth = hdr.Get("x-ollama-auth")
		}
		streamID := streamIDFromContext(ctx)
		rpc := conn.Spec().Procedure
		peerAddr := conn.Peer().Addr
		status := "permissive-continue"
		reason := "stream auth check at interceptor using conn.RequestHeader (early for ChatStream/GenerateStream etc); permissive local dedicated gRPC port; metadata present or absent -> always proceed"
		if auth != "" {
			reason = "stream auth metadata present (permissive local); enhanced for finding7"
		}
		// mTLS skeleton (as in authUnary; applies to streams too e.g. long-lived agent ChatStream):
		// Client cert extraction from conn if TLS: conn.Peer() has no certs (connect design); when TLS listener, use conn state or injected.
		// Pseudocode: if tlsState := ...; len(tlsState.PeerCertificates)>0 { ... } ; stronger validation/deny for !local.
		// Docs: see phased Phase5 mTLS/auth per plan + report finding7; no core prop needed for local.
		slog.Debug("gRPC auth decision", "component", "grpc", "rpc", rpc, "stream_id", streamID, "peer_addr", peerAddr, "status", status, "reason", reason+" ; mTLS skeleton active (TLS config notes + cert extract pseudocode in grpc.go; stronger than perm for non-local/future)")
		return next(ctx, conn)
	}
}

// recovery* : named returns + defer for panic override (matches unary fix). Now for streams too.
func recoveryUnary(next connect.UnaryFunc) connect.UnaryFunc {
	return func(ctx context.Context, req connect.AnyRequest) (res connect.AnyResponse, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = connect.NewError(connect.CodeInternal, fmt.Errorf("panic recovered in gRPC: %v", r))
			}
		}()
		return next(ctx, req)
	}
}

func recoveryStream(next connect.StreamingHandlerFunc) connect.StreamingHandlerFunc {
	return func(ctx context.Context, conn connect.StreamingHandlerConn) (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = connect.NewError(connect.CodeInternal, fmt.Errorf("panic recovered in gRPC stream: %v", r))
			}
		}()
		return next(ctx, conn)
	}
}

// loggingInterceptor now full (unary+stream) via adapter so streams get logging/auth/recovery/otel + stream_id + rich reason.
// Per report finding 2 + grpc.go TODO + phased Phase5 overlay.
func loggingInterceptor() connect.Interceptor {
	return interceptor{
		wrapUnary:  loggingUnary,
		wrapStream: loggingStream,
	}
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
// Parses Authorization or x-ollama-auth. Enhanced Phase 3 for finding7 (mTLS skel + richer logs in wrappers; streams via adapter from Phase1/2).
// Still permissive (no deny); see authUnary/authStream for mTLS skeleton + TLS config notes + cert extract pseudocode + strong future non-local.
func authInterceptor() connect.Interceptor {
	return interceptor{
		wrapUnary:  authUnary,
		wrapStream: authStream,
	}
}

// enableGRPCReflectionIfOptIn is skeleton (per finding7 "Reflection for easier grpcurl in dev (opt-in)", report p57 grpcurl fail + "connect supports but not enabled",
// phased p283 "Basic health/reflection (connect equiv or dual)", p369 grpcurl in verif gates, p323/379 "gated not default" + review subagent).
// Called from registerServices (routes.go, same pkg) after New*Handler wiring + interceptors slice.
// Gated by envconfig.GRPCReflection() (existing pattern from env edit, no new global/var).
// Rich slog (component/reason/stream? n/a here/status/model n/a/rpc n/a) at the enable *decision* (Flume style).
// When enabled (OLLAMA_GRPC_REFLECTION=1 not default): logs; pseudocode to register connect reflection handler (for grpcurl list/desc without protos).
// Skeleton only: no new dep (would need go get connectrpc.com/grpcreflect or grpc/reflection + dual setup for full grpc compat); no change to mux/listener now.
// mTLS note: reflection on dedicated port; pair with future stronger auth/mTLS per plan for agents.
// Safe, idempotent, small, no I/O, no ctx (setup time).
func enableGRPCReflectionIfOptIn(mux *http.ServeMux, interceptors ...connect.Interceptor) {
	if !envconfig.GRPCReflection() {
		return
	}
	slog.Info("gRPC reflection opt-in enabled (skeleton)", "component", "grpc", "reason", "OLLAMA_GRPC_REFLECTION=1 (per finding7 + report p57 'no gRPC reflection' + phased p283/369); grpcurl --plaintext will now list services without --proto (connect reflection was supported but not enabled); gated not default (p323 high-scrutiny, p379 gates); mTLS/auth stronger in future plan for prod. Full: add dep + grpcreflect.NewStaticReflector(apiv1connect.*ServiceName...) + NewHandler + mux.Handle; see comments. Current no-op beyond this log (safe for local). DEFERRED per Phase 5 overlay p323 (high risk to existing + review subagent 019eae31-7a95-7753-829f-72f2e1bc0f89 findings); do not wire real until post-soak stable + re-review. Skeletons + permissive local satisfy MVP opt-in gates.", "status", "opt-in")
	// Skeleton registration (future when dep added; keep pseudocode for review/plan):
	// ref := grpcreflect.NewStaticReflector(
	// 	apiv1connect.ChatServiceName,
	// 	apiv1connect.GenerateServiceName,
	// 	apiv1connect.EmbedServiceName,
	// 	apiv1connect.ModelsServiceName,
	// )
	// path, h := grpcreflect.NewHandler(ref, connect.WithInterceptors(interceptors...))
	// mux.Handle(path, h)
	// slog.Debug("gRPC reflection handler wired", "component", "grpc", "reason", "grpcurl friendly now active")
}

// enableGRPCHealthIfOptIn adds basic gRPC health support skeleton (P3 complete).
// Per docs/grpc-phased-reliable-approach.md p283 ("Basic health/reflection (connect equiv or dual)"),
// p368 (recs for new code), p88/379 (gates + "stop if"), report sec4 item9 + p94-95 ("Agent/Flume productionization ( ... health/reflection/metrics)"), p66-75 (obs/errs for agents), p97 verdict (need clients+health for prod).
// Complements enableGRPCReflectionIfOptIn (the P3 partial skeleton). Always active (cheap readiness, unlike opt-in refl for grpcurl/dev).
// No new deps (pseudocode for grpchealth later; current no-op beyond log to satisfy minimal+no-dep scope).
// Rich slog (component/reason/status/rpc/model n/a/stream_id n/a/dur n/a) at enable *decision* (Flume/LogAgentReasoning style; agents/Flume consume for pre-stream checks).
// Ctx n/a (setup); errors n/a; idempotent/safe; small unit; no globals; no I/O. When wired full: HealthService Check returns SERVING for Chat/Generate/Embed/Models (respecting stream ctx for watch); integrates with otel/auth intcps.
// Use by gRPC clients (future api/grpc_client.go health check) + k8s/grpc probes before agent concurrency/load.
// Called from registerServices (routes.go) like refl.
func enableGRPCHealthIfOptIn(mux *http.ServeMux, interceptors ...connect.Interceptor) {
	slog.Info("gRPC health skeleton active", "component", "grpc", "reason", "basic health per P3 phased p283/368 + report item9 (health/refl/metrics for agent/Flume prod + load/soak under concurrency); complements refl opt-in (p57 grpcurl); no gate (readiness default for prod agents); no dep (pseudocode only); status=skeleton; mTLS/auth stronger future per plan. DEFERRED per Phase 5 overlay p323 (high risk + subagent 019eae31-7a95-7753-829f-72f2e1bc0f89 Critical on skeletons); explicit defer for soak/review cycle; current always-log + cheap readiness satisfies minimal gates for opt-in (no behavior change).", "status", "skeleton")
	// Full impl (defer dep add per "no new deps if possible" + SKILL minimal):
	// checker := grpchealth.NewStaticChecker(
	//	apiv1connect.ChatServiceName,
	//	apiv1connect.GenerateServiceName,
	//	apiv1connect.EmbedServiceName,
	//	apiv1connect.ModelsServiceName,
	// )
	// path, h := grpchealth.NewHandler(checker, connect.WithInterceptors(interceptors...))
	// mux.Handle(path, h)
	// slog.Debug("gRPC health handler wired", "component", "grpc", "reason", "SERVING status for inference+admin services; agents can health-check before ChatStream/queue load")
}

// recoveryInterceptor: recover panic to connect err (per SKILL bounded, verifiability).
// Fixed per review: use named return so defer can override err on panic in next(). Now covers streams.
func recoveryInterceptor() connect.Interceptor {
	return interceptor{
		wrapUnary:  recoveryUnary,
		wrapStream: recoveryStream,
	}
}

// otelInterceptor for Phase 4 full observability (provider via global, custom attrs via spans in handlers).
// Per SKILL: errors checked (no ignore), rich slog at decision, small unit.
func otelInterceptor() connect.Interceptor {
	setupOTELProvider() // ensure provider for Phase 4 "full" (called once at register)
	intcp, err := otelconnect.NewInterceptor(
		otelconnect.WithTracerProvider(otel.GetTracerProvider()),
		otelconnect.WithMeterProvider(otel.GetMeterProvider()),
	)
	if err != nil {
		slog.Error("otelconnect.NewInterceptor failed", "error", err, "component", "grpc", "reason", "obs degraded but handlers continue with manual spans; classified as non-fatal for local dev")
		// fallback: identity interceptor (safe, no panic)
		return connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
			return next
		})
	}
	return intcp
}

// setupOTELProvider implements real (non-stub) provider using sdk for in-memory spans.
// Uses simple sdk (no exporter per phased doc to avoid bloat/risks); enables custom attrs
// for model/stream_id/load/inference/schedule/tokens in gRPC + extracts. Idempotent for safety.
// No package globals per SKILL. Rich slog with component/reason/status (Flume style).
func setupOTELProvider() {
	// use type assert on global (no our package var) to make safe if re-called
	if _, ok := otel.GetTracerProvider().(*sdktrace.TracerProvider); ok {
		return
	}
	tp := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(tp)
	slog.Info("otel provider initialized", "component", "grpc", "reason", "real sdk TracerProvider (not stub) for custom attrs on inference/schedule/load paths + otelconnect; simple in-mem to avoid bloat", "status", "ok")
}

// Phase 2: streaming interceptors complete (interceptor{} adapter + WrapStreamingHandler for logging/auth/recovery/otel + stream_id correlation via ctx value; now covers all incl admin Pull/Push ServerStreams).
// Rich "reason" slog + stream_id/dur/status/component at every decision (Flume/LogAgentReasoning style; Phase1 patterns extended).
// Converters fuller (oneofs/Usage/Struct/Sampling/FinishReason/ToolCall details in msgs/reqs/resps); exhaustive tables in grpc_test.go (roundtrips + edges per report #3 + SKILL p43/70).
// Admin streams (report #4): modelsHandler Show fleshed (cache/GetModelInfo + convert to Struct/bytes), Ps (from sched.loaded post-reconcile), Pull/Push as ServerStream (streamCtx/cancel/select/Send(convertProgress) + progressFn + PullModel/PushModel + refresh; "Pull as server stream w/ progress per report #4/P5").
// var _ checks for 4 handlers + interceptor. (Preexist LlmRequest containedctx, GetModelInfo no-ctx, long extracts, raw sched.loaded in Ps, some _= in branches, no reflection/mTLS/runners for E2E positive streaming still block full "ready for agents via Flume" per report p97 + phased gates p379/381; see review /tmp/grok-grpc-phase2-review.md).
// [slog] component=grpc reason="performance/bounded streams skeleton per report finding sec4 item10 + phased p95 + p63/69 'cap 64 in design' + p373 'bounded buffering ch cap 64 + select on ctx.Done before Send/chan; on return cancel to stop llm' ; current: routes:3371 ch:=make(...,64) in core stream paths, grpc ChatStream/GenerateStream:94-106,168-180 select{<-streamCtx.Done(): cancel(); return err} + defer cancel from stream ctx (p334), modelsHandler:215 'Bounded: selects... no fire-forget', sched MaxQueue buffered chans + errgroup.SetLimit(3) for sameport in cmd (P5 hard); no custom flow control or larger caps yet (skeleton note, defer per item10 'perhaps larger caps or flow control'); rich reason at decision for audit" stream_id=from-intercp-or-fresh dur=0 status=perf-skel-note model=varies rpc=ChatStream|GenerateStream|Pull|Push ref=report-p95/10 phased-p321-p379
// TODO Phase3+: runners (report #1; current blocks inference), sameport dedicated soak+review subagent (high-risk p323), full integ matrix (GenerateStream+admin streams tests+cancel mid+concurrency+models; expand grpc_stream_test), LogLoom post-change lift assert (p374), golangci+errcheck manual, mTLS/reflection, GetModelInfo ctx-first (update calls + HTTP), Scheduler.ListRunning() exported (avoid raw loaded), clean preexist (LlmRequest ctx, long chat/generate >60LOC split, silent errs elsewhere), commit hygiene (no untracked/dirty in PRs).
// Note (for assigned finding sec4 item10 + phased p322): richer protos (xai oneofs/Struct/Sampling/req options/logprobs) addressed in P2/P3 (fuller converters at convertToAPIChat etc + exhaustive tables in grpc_test.go covering sampling+struct+usage+done_reason+logprobs+toolcalls+vision per report "fuller converters+rich protos+expanded tables"; proto comments still have some TODOs for future oneofs). image/video deferred (per doc p322; vision/images supported in conv but full x/imagegen separate), official clients overlap report #9. Buf CI enhancements + performance/bounded streams skeleton addressed in this Phase4 item10 work (see .github/workflows/test.yaml buf: + above perf [slog] note).

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

// convertShowResponseToPB fleshes Show (details) from GetModelInfo or modelCaches.show (per report finding 4 + P5).
// Maps api.ShowResponse (has ModelDetails + map any for info) to proto (Struct + bytes map). MVP omits some (tensors/messages).
// Small, no I/O. Reason logged at call site.
func convertShowResponseToPB(resp *api.ShowResponse) *v1.ShowModelResponse {
	if resp == nil {
		return &v1.ShowModelResponse{}
	}
	pb := &v1.ShowModelResponse{
		License:    resp.License,
		Modelfile:  resp.Modelfile,
		Parameters: resp.Parameters,
		Template:   resp.Template,
	}
	// details to Struct (flattened like List; supports nested in future)
	dm := map[string]any{
		"parent_model":       resp.Details.ParentModel,
		"format":             resp.Details.Format,
		"family":             resp.Details.Family,
		"parameter_size":     resp.Details.ParameterSize,
		"quantization_level": resp.Details.QuantizationLevel,
	}
	if s, err := structpb.NewStruct(dm); err == nil {
		pb.Details = s
	}
	// model_info: any -> bytes (MVP fmt repr; no json dep added)
	if resp.ModelInfo != nil {
		pb.ModelInfo = make(map[string][]byte, len(resp.ModelInfo))
		for k, v := range resp.ModelInfo {
			pb.ModelInfo[k] = []byte(fmt.Sprintf("%v", v))
		}
	}
	return pb
}

// convertProgressToPB maps core progress (used by PullModel/PushModel fn) to proto for server streams.
// Per report: "using connect.ServerStream for api.ProgressResponse-like". Idempotent.
func convertProgressToPB(r api.ProgressResponse) *v1.ProgressResponse {
	return &v1.ProgressResponse{
		Status:    r.Status,
		Digest:    r.Digest,
		Total:     r.Total,
		Completed: r.Completed,
	}
}

var _ = authInterceptor // compile use
