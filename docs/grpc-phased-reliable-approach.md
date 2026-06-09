# Reliable-Go Systems Overlay: Logical Phased Approach for Adding gRPC Support to ollamas (feature/grpc-initial)

**Date:** 2026-06-03  
**Author:** Principal Go Engineer & Reliability Specialist (Flume/xAI patterns; Grok Build subagent)  
**Inputs (MUST-READ, read via tools before edits):**  
- `/Users/jonathandoughty/.grok/skills/reliable-go-systems/SKILL.md` (full mandatory checklist + principles; **quoted extensively below**).  
- `docs/grpc-architecture-plan.md` (master synthesized plan; critiqued/annotated here as "the Plan").  
- `/tmp/grpc-research-report-1.md` (xai-proto, industry, pitfalls incl. ctx/GPU [grpc-research:138]).  
- `/tmp/ollamas-grpc-codebase-review.md` (entrypoints, gotchas incl. Gin, cloud, scheduler, low logs [ollamas-review:192-205]).  
- `logloom-graph.json` (AST; 7.67% coverage = 398/5191 instrumented funcs; 50 uninstrumented in server/routes.go incl. scheduleRunner:205/ChatHandler:2422/GenerateRoutes:1794/etc.; 52 in server/sched.go incl. getRunner/InitScheduler; 27 logged nodes in routes.go, 70 in sched.go but clustered).  
- Key sources (verified via read_file/grep/run_terminal): `server/routes.go:100` (Server struct), `:1794` (GenerateRoutes), `:1917` (Serve), `:205` (scheduleRunner), `:2422` (ChatHandler), `cmd/cmd.go:1996` (RunServer), `envconfig/config.go`, `server/sched.go:91` (InitScheduler), `llm/server.go:59` (LlamaServer iface), `api/types.go`, `logutil/logutil.go`, `server/model_resolver.go`, `integration/utils_test.go:430` (serverMutex), `.golangci.yaml`, `docs/development.md`.  

**Status:** This is the **authoritative "reliable overlay"**. Implementers **MUST** follow it *on top of* the arch plan. It enforces/prioritizes/details reliable-go-systems patterns (Kubernetes reconciliation, NASA Power of 10, Google/AWS/Uber prod Go, xAI agent orchestration, Flume Claimer/Sweeper/Runner + LogAgentReasoning). All claims cite exact paths/lines.  

**Core Directive (from SKILL):** "This skill encodes the **strong commonalities** across high-reliability, large-scale Go codebases... It is not a style guide — it is a set of **enforced patterns and review checklists** that agents must follow (and verify) unless the user explicitly says 'ignore reliable-go for this task'." "When to Deviate: Only with explicit user instruction..."

**How to Use:** Planning (this doc + protos + arch plan) is *separate* from execution (gRPC server code changes). For any impl work >3 steps: **use todo_write** (idempotent updates; mark completed ONLY when fully verified per gates). Follow "implement → review (reliable checklist + this doc) → fix" loop per phase (like the implement skill). "stop if" criteria are non-negotiable.

---

## 1. Executive Critique of the Plan from Reliable-Go Lens

The Plan (`docs/grpc-architecture-plan.md`) is a strong, grounded synthesis (cites exact lines from reviews, LogLoom, sources; prioritizes "extract once, serve many protocols" + shared `*Server` + separate port for safety; incorporates reliable-go in inputs p12 + general rules p221-229 + streaming p195-204 + obs p176-181 + risks p562-583). It correctly identifies the single best entrypoint (Serve:1917 + *Server + scheduleRunner:205 + sched), thin adapters, extraction to avoid Gin coupling, ctx/deadlines critical, shared sched to prevent contention, LogLoom for gaps, errgroup hints, compile checks, -race/table tests, buf/connect primary.

**However, from the reliable-go-systems SKILL lens, it is incomplete as an *implementation* guide and requires this overlay to enforce patterns. Specific shortfalls (with quotes + citations):**

- **Context is King (SKILL p13):** " `ctx context.Context` is **always** the first parameter for public functions that do I/O... Never store ctx in structs. Use `context.WithTimeout` / deadlines liberally. Respect cancellation..."  
  Plan partially addresses (general rules p222: "Every I/O/long func: ctx first"; streaming p196-201: "Always `ctx, cancel := context.WithCancel(stream.Context())`"; scheduleRunner already takes it at routes:205 and passes to sched.getRunner:165). **Critique:** Insufficiently *specific* on *which* new public funcs (e.g. `RegisterGRPC`, `ServeGRPC`, extracted `chat(ctx, req, write func) error`, `generate(...)`, converters?, interceptors, grpc handlers). Does not explicitly call out/require avoiding the existing violation in `server/sched.go:31` (`LlmRequest { ctx context.Context //nolint:containedctx }` -- containedctx linter is *enabled* in `.golangci.yaml`). Propagation from gRPC `stream.Context()` (critical gotcha per [grpc-research:138], [ollamas-review:192]) to scheduleRunner/scheduler/llm calls (r.Chat etc. at routes:2972 etc. often use c.Request.Context() directly) is mentioned but not gated. Call sites in handlers (e.g. 2616: `s.scheduleRunner(c.Request.Context(), ...)`) must be updated in extracts to use passed ctx. **Augment needed:** See checklist + phases.

- **Errors Are Values (SKILL p15):** "Never ignore errors (`_ = foo()` is a bug). Always check. Use `fmt.Errorf("...: %w", err)`. Use `errors.Is` / `errors.As`... Define package-level sentinel errors... Error strings are lowercase, no trailing punctuation. 'Indent error flow' — happy path has minimal nesting."  
  Plan addresses (p223: "Errors: check all, `%w` wrap, `errors.Is/As`..."; error mapping p189-193 + p448 in grpc.go sketch; sentinels like errRequired at routes:124). Good use in scheduleRunner (207: `fmt.Errorf("model %w", errRequired)`). **Critique:** Does not mandate *classification* of transient (e.g. GPU OOM as retryable via scheduler's oomRetryAttempted:40 in sched.go, rate limits, context.DeadlineExceeded) vs permanent for gRPC codes + client retries (Idempotency section). No explicit "no _ = " or errcheck in new paths (note: `.golangci.yaml` *disables* errcheck linter, but skill forbids ignores). Gin.H error sentinels (streamResponse:2050 etc.) must be mapped cleanly without duplication. **Augment:** Classify in errToConnect; wrap all in extracts.

- **Reconciliation over Event-Driven Edges (SKILL p17, "the single biggest reliability win from Kubernetes"):** "For any queue/worker/state machine... Event or timer → enqueue a **key**... Worker dequeues → **reconcile current desired state** (level-driven, not edge-driven)... Idempotent handler. On transient error → `queue.AddRateLimited(key)` + backoff... Use rate-limiting queues... Bounded workers + `wait.UntilWithContext` or `errgroup` + ctx... maps directly to Flume's Claimer/Sweeper/Runner model."  
  **Plan strength + gap:** Plan notes (p203: "Scheduler semaphores... respected (shared)"; p4 in recs: "if touching, ensure rate-limited, idempotent reconcile on key (model? req id)"). *Scheduler* (sched.go:61) *already* implements a custom version: buffered chans (pendingReqCh etc sized to MaxQueue:94), LlmRequest keyed by schedulerModelKey:193, processPending:221 (select loop on ctx.Done() + events, level-driven reconcile of loaded map:75, evict logic, findRunnerToUnload, useLoadedRunner idempotent + refcount, oomRetry, rate via queue + backoffs in practice). getRunner:165 enqueues. Run:210 spawns workers. This is *reconciliation*, not pure pubsub edges. **Critique:** Plan does not *explicitly* praise/mandate "preserve and extend this pattern; gRPC handlers enqueue via scheduleRunner (never bypass or add edge-driven workers)". If any new state (e.g. stream lifecycle), use key+reconcile not fire events. No workqueue from k8s, but custom is fine if follows the rules (idempotent, rate, backoff on transient). LlmRequest stores ctx (anti-pattern per #1). **Augment:** See phases for "if touching sched".

- **Bounded Everything + Resource Discipline (SKILL p25, NASA Power of 10 + K8s + AWS):** "No unbounded loops or recursion in hot paths. Bounded concurrency (semaphores, `golang.org/x/sync/semaphore` or errgroup with limit, worker pools). Timeouts + deadlines on **all** I/O... `defer` for every resource... No fire-and-forget goroutines. Every goroutine has an owner and a clear exit condition (usually ctx.Done()). Assertions / runtime checks in critical paths (NASA Rule 5)."  
  Plan addresses well (p225: "Bounded: chans, workers, timeouts. `defer` resources. `errgroup` for lifecycles."; p201: "No fire-and-forget..."; streaming p196-202: "Bounded internal chans (cap 16-64...); select on ctx.Done()..."; sched has semaphores/NumParallel/MaxQueue:493 etc.; shutdown signals + unload:1995). go.mod has golang.org/x/sync. **Critique:** RunServer:1996 (cmd/cmd.go) is *not* using errgroup yet (simple sequential Serve); dual listeners must use `g, ctx := errgroup.WithContext(...)`; `g.SetLimit(2)` or equiv; signals integrated. Existing go funcs (routes:2728 callback loop with defer close(ch) but ctx from c.Request; 2969; sched:415 retry sleeper, 483 useLoaded go on ctx.Done()) have owners but not always errgroup-tracked. No explicit bounds on new gRPC streams without backpressure. LoadTimeout etc from env good, but enforce per-RPC. **Augment needed in Phase 1/2.**

- **Observability is Not Optional (SKILL p34):** "Every significant action, state transition, error, and decision must produce **structured** output: Use `log/slog` ... consistent keys: `component`, `task_id`, `role`, `status`, `error`, `duration_ms`, trace/span... For agentic systems (Grok/xAI style): rich "reasoning" / audit logs that feed popouts, graphs, and post-mortems (see Flume's LogAgentReasoning). Health/readiness... Request/correlation IDs propagated via ctx."  
  Plan has dedicated section (p176-181): "slog (existing: logutil...); ... rich attrs on *all* paths (model, grpc_method... stream_id or req uuid, tokens... `slog.Info("grpc chat started"...`; LogLoom gaps: "Only ~28 instrumented points in `server/routes.go`... sched... clustered... **Plan:** New gRPC paths + extracted methods get comprehensive coverage + "reasoning" style logs (e.g. `slog.Debug("decision: local inference path", "reason", ...)` ... OTEL via otelconnect; health/reflection; post-impl re-run logloom (p215). **CRITICAL GAP (MUST ENFORCE):** Does not cite *exact* numbers from logloom-graph.json (queried 2026-06-03): **7.67% coverage (398/5191 functions)**; **50 uninstrumented in server/routes.go** (incl. `Server.scheduleRunner`, `Server.ChatHandler`, `Server.GenerateRoutes`, `Server.EmbedHandler`, `handleScheduleError`, `allowedHostsMiddleware`, `filterThinkTags`, `getExistingName`, many handlers + helpers); **52 uninstrumented in server/sched.go** (incl. `InitScheduler`, `Scheduler.getRunner`, `Scheduler.GetRunner`, `processPending` paths, load/evict, `LlmRequest.*`); only 27 logged nodes in routes.go (grep confirmed ~28 in plan). Current logs (routes:28 total; sched:70 but mostly debug in process loops, e.g. "new model fits...", no consistent "component", "reason", "stream_id", "tokens", "duration_ms" on decision paths). No "reasoning" examples at evict/load/choice points (one "reason" for mmap at sched:1144). Plan calls for addition but not *mandated sites* or "re-run logloom build + report after *every* phase touching routes/sched to measure lift". No requirement for slog on *token emit paths* (via llm callbacks), error classify, schedule decision. **Augment:** See checklist + phases (MUST add to cover uninstr funcs touched; use "reason" key for Flume-style at branches like "chose to evict X for new gRPC req Y because VRAM Z < available"). Bridge slog<->OTEL. golangci + -race mandated only generally.

- **Idempotency + Safe Retries with Jitter (SKILL p41):** "All mutating operations and handlers must be safe to retry. Use exponential backoff + jitter... Distinguish retryable vs. permanent errors. Circuit-breaker / bulkhead..."  
  Plan touches (p225 general; scheduler queue as bulkhead; error map for codes like Canceled/Invalid). Scheduler already rate-limited + retry paths (oom, evict). **Critique:** No explicit "make gRPC handlers/enqueues idempotent (e.g. on model key or future req ID; safe to re-call scheduleRunner)"; no jitter if adding backoff in adapters; transient classification for gRPC status (UNAVAILABLE for queue full/GPU temp vs INVALID for bad model). Cloud passthrough errors must be permanent for gRPC. **Augment in error paths + sched note.**

- **Simplicity, Small Units, Verifiability (SKILL p43, NASA + Google):** "Functions short enough to fit on one screen (~50-60 lines max for complex logic). Simple control flow. No deep nesting. Liberal use of table-driven tests + fuzzing + `-race`. Zero warnings under `golangci-lint` (staticcheck, errcheck, govet...) + `go test -race`. Compile-time interface checks: `var _ SomeInterface = (*Impl)(nil)`."  
  Plan has (p227: "Long funcs: keep extracted <60 LOC where possible; helper funcs."; "Tests: `-race`, table-driven. `golangci-lint` clean."; "Add compile checks: `var _ chatv1...` p452; "PR: ... tests"). Good call for thin extract. **Critique:** Does not *enforce per-phase* "golangci-lint run + go test -race before advance" (only in verif steps, not gates); .golangci disables errcheck (conflicts with Errors Are Values -- must still manually ensure in reviews); no table tests mandated for converters (plan says "Unit tests for converters (table...)" good but buried); new files must be small. Existing long funcs (ChatHandler:2422+ is beast with nested structured outputs dance ~2720-3000) must be split in extract. No fuzz. **Augment:** Explicit in each phase verif + "stop if".

- **No Globals / Explicit Construction (SKILL p50):** "Configuration at startup only. Functional options pattern for complex constructors. Dependency injection (manual is fine...). Avoid `init()` for anything non-trivial."  
  Plan general (p228: "No globals"). Current code has package vars (routes:96 `var useClient2`, :98 `var mode`, :123 `var (errRequired...)`; sched:87 `var defaultModelsPerGPU`; global slog.SetDefault in Serve:1918; gin init:108). Server constructed explicitly:1951. **Critique:** Plan does not call out avoiding *new* globals for gRPC (e.g. no `var grpcEnabled`; use envconfig.GRPCHost() == nil guard, pass via Server or explicit in cmd). No init() additions. GRPCHost must mirror Host exactly (no magic). **Augment:** Enforce in wiring phases.

- **Agent Orchestration Layer (SKILL p52, xAI/Grok + K8s):** "For complex multi-step or agent-driven work: Separate **planning** (read-only, research, todo breakdown) from **execution**. Use structured task tracking (todo lists) for anything >3 steps. Delegate via isolated sub-agents/workers with clear contracts... Verification loops: implement → review → fix until zero issues (or explicit approval). Rich reasoning capture at every decision point..."  
  **Plan strength + gap:** Plan itself separates (synthesis from research + this is planning); cites "Track in todo_write during impl" (p231); "Per phase: ... verif, risks..."; " (Tracked via 8+ todo_write items; synthesis complete.)" at end. This task *is* the agent orchestration (planning doc). **Critique:** Does not *mandate* "use todo_write (idempotent) for *all* impl phases/work >3 steps; only complete when verified per gates"; no "implement-review-fix loop" explicit in phase structure; no "suggest using check-work skill or review subagent at end of phases" (this task does); rich reasoning in *this* doc but not required in code logs. Phases lack "planning vs exec" note per phase. **This overlay fixes it.**

**Overall Verdict on Plan:** Solid foundation (additive, shared core, ctx hints, LogLoom awareness, risks mapped). But as "master plan for implementers", it under-specifies enforcement of SKILL (no per-phase gates/"stop if", weak log mandates vs actual 7.67% data, no explicit agent loops/todo in impl, incomplete on globals/errclass/no-fireforget/ctx-in-structs). This doc is the production reliability voice: treat as non-negotiable overlay. "Use it."

---

## 2. Mandatory Reliable-Go Checklist Applied to gRPC Addition

**From SKILL "Mandatory Review Checklist (Agent Must Apply Before Proposing Code)" (p61):** Before *any* edit, internally answer/document:

- [ ] Is every I/O or long-running function taking `ctx` as first param and respecting it?  
- [ ] Are all errors checked, wrapped with `%w`, and classified (transient/permanent)?  
- [ ] If this is a worker/queue/state machine: is it using the reconciliation + rate-limited queue pattern (or equivalent)?  
- [ ] Are goroutines bounded? Is there a clear owner and shutdown path for every goroutine?  
- [ ] Is structured logging + key identifiers (task_id, component, role, etc.) present on all significant paths?  
- [ ] Are timeouts/deadlines present on all external calls (ES, LLM, git, HTTP)?  
- [ ] Is the handler idempotent? Safe to retry?  
- [ ] Did I run `golangci-lint` (or equivalent strict config) and `go test -race` mentally or actually?  
- [ ] For Flume-specific: Did I respect the existing Claimer/Sweeper/Runner separation, state machine in `pkg/types`, and LogAgentReasoning requirements?  
- [ ] Have I added or updated rich reasoning logs for any new decision points?

**Applied to gRPC (how Plan addresses + *required augmentations* for this work; must re-apply before every edit):**

1. **Context is King (SKILL p13, checklist p63):**  
   Plan addresses: general rule p222, streaming ctx/cancel p196, scheduleRunner already ctx-first (routes:205), calls to sched/llm noted.  
   **Must augment:** *Every* new public/exported I/O func **MUST** have `ctx context.Context` as *first* param after receiver: `func (s *Server) ServeGRPC(ctx context.Context, ln net.Listener) error`, `func (s *Server) chat(ctx context.Context, req api.ChatRequest, write func(api.ChatResponse) error) error` (and generate/embed equivalents), `func (h *chatHandler) Chat(ctx context.Context, req *connect.Request[...]) ...`, any converter that may I/O (unlikely), interceptors. *Never* `context.Background()`. Propagate *always*: from gRPC stream handler use `stream.Context()` (or the ctx param) -> extracted -> scheduleRunner (update call sites from c.Request.Context()) -> sched.getRunner (already takes) -> LlmRequest (note: existing containedctx at sched:31; avoid in any *new* structs) -> llm.LlamaServer.Chat(ctx,...) etc (iface at llm/server.go:65). In streams: `ctx, cancel := context.WithCancel(stream.Context()); defer cancel(); ... select { case <-ctx.Done(): return ctx.Err() ... }`; on handler return (Send err or client cancel): cancel to stop gen promptly (prevent GPU leak per [grpc-research:138]). Deadlines: respect per-RPC + env LoadTimeout. **Stop if:** any new/edited long func lacks ctx first or passes bg/ignores Done. Use `context.WithTimeout` on loads/calls.

2. **Errors Are Values — Explicit, Wrapped, Typed (SKILL p15, checklist p64):**  
   Plan addresses: %w, Is/As, mapping to connect codes (p189-193, p448), lowercase no punct, sentinels (errRequired etc).  
   **Must augment:** Check *every* err (no _ = in new code; manually despite golangci errcheck disabled). Wrap *all*: `fmt.Errorf("grpc chat %s: %w", model, err)`. Classify: define/use sentinels or typed for transient (e.g. `var ErrTransient = errors.New("transient");` or use existing ErrMaxQueue:89 in sched, context errors, OOM-retryable). In errToConnect: map Canceled->CodeCanceled, Deadline->CodeDeadlineExceeded, transient/queue->CodeUnavailable (retryable), bad model/caps->InvalidArgument, etc. Use `errors.Is/As` for branching (happy path indent-minimal). In extracted: return err; callers (gin thin + grpc) map. Gin.H errors stay internal to HTTP compat. **Stop if:** ignored err or bare return without wrap/classify.

3. **Reconciliation over Event-Driven Edges (SKILL p17, checklist p65):**  
   Plan addresses: notes scheduler queue (p203); "gRPC handlers must enqueue keys not edges" (in recs p4).  
   **Must augment:** *Preserve* existing Scheduler pattern (sched.go:61-82: pending/finished/expired/unloaded chans, loaded map keyed by model, processPending:221 level-driven reconcile loop on select/ctx.Done(), getRunner:165 enqueues LlmRequest by schedulerModelKey:193, idempotent useLoadedRunner:471 + refcounts, rate via MaxQueue buffers + evict/oom logic, backoff in practice). gRPC path *must* go through scheduleRunner:205 -> sched.getRunner (no bypass, no new edge-driven goroutine for streams). If *any* new worker/state for gRPC (e.g. stream lifecycle claimer), enqueue *key* (model or reqID), reconcile desired vs current (loaded/VRAM), rate-limit, transient -> requeue. Never "on stream start, go load". **Stop if:** new code adds pubsub/edge without key+reconcile+idempotent.

4. **Bounded Everything + Resource Discipline (SKILL p25, checklist p66):**  
   Plan addresses: errgroup hints, bounded chans 16-64, semaphores in sched, defer, no fire-forget (p201,225), timeouts.  
   **Must augment:** In Phase 1 cmd/cmd.go:1996 RunServer: use `g, ctx := errgroup.WithContext(context.Background())`; `g.SetLimit(2)`; `g.Go( func() error { return serveHTTP(...) } )`; `if lnGRPC != nil { g.Go(...) }`; return g.Wait(). Signals: cancel ctx, close both, sched.unloadAllRunners(), <-done. In ServeGRPC: http.Server with ctx-aware Shutdown. Streams: *bounded* token chan from core (cap 64); writer: `select { case ch <- r: case <-ctx.Done(): return ctx.Err() }`. Every go func() (e.g. callback loops in extract like routes:2728/2969) *must* have owner (the ctx or errgroup), defer close, select on Done, no fire-forget. Prealloc, asserts in hot paths (NASA). Defer *every* Close/Unlock/Cancel. **Stop if:** unbounded chan/goroutine or missing defer/owner/ctx select in new paths.

5. **Observability is Not Optional (SKILL p34, checklist p67):**  
   Plan addresses: rich slog+OTEL+reasoning callout + LogLoom use (p176-181,215).  
   **Must augment (CRITICAL):** *EVERY* significant path in gRPC + extracted + touched (accept, modelRef parse/cloud branch, GetModel, schedule decision/success/fail, render choice (harmony/tools/think), token emit via callback, error classify (handleScheduleError), load/evict, finish reason, shutdown) **MUST** emit `slog` (preserve logutil custom TRACE/-8 + basename) with **consistent keys**: `component:"grpc"` (or "server" for shared), `method:"Chat"`, `model: req.Model`, `stream_id: uuid.New().String()` (or task_id/req uuid), `status`, `duration_ms: time.Since(start).Milliseconds()`, `error?`, `tokens?` (prompt/completion), `reason?` (for decisions). **Rich "reasoning" logs (Flume LogAgentReasoning style) at decision points:** e.g. `slog.Debug("chose inference path", "reason", "no :cloud suffix, local model", "model", ..., "component", "grpc")`; in sched paths if touched: `"chose to evict X for new gRPC req Y because VRAM Z < available after loaded"`. Add *to cover uninstrumented*: at min, instrument scheduleRunner:205 (currently uninstr per LogLoom), extracted chat/generate (was in ChatHandler:2422 uninstr), error paths, decision branches. Use in interceptors (start/end). IDs propagated via ctx (add if needed). Health/reflection + OTEL (otelconnect interceptor with GenAI semconv). **Post any routes/sched/grpc change:** `~/.local/bin/logloom build` (updates graph.json); `~/.local/bin/logloom report` or `logloom lint`; measure lift (e.g. new nodes for grpc funcs, coverage >7.67%, 0 remaining uninstr in touched paths like scheduleRunner). **Stop if:** missing slog with keys on path, or no reasoning at branches, or post-logloom shows no coverage gain on new code.

6. **Idempotency + Safe Retries with Jitter (SKILL p41, checklist p69):**  
   Plan addresses: scheduler as bulkhead/queue, error codes for client retry.  
   **Must augment:** gRPC handlers/enqueue paths *idempotent* (re-call safe; model key based, or add req ID in proto for dedupe if complex). Scheduler already supports (reconcile on key, refcounts prevent dupe loads). Classify errors for safe retry (transient -> client can retry with backoff+jitter; permanent no). If adding any retry in adapters: exp backoff + jitter (never pure). Circuit/bulkhead via existing MaxQueue/NumParallel. **Stop if:** mutating path not safe for retry or no transient/permanent distinction.

7. **Simplicity, Small Units, Verifiability (SKILL p43, checklist p70):**  
   Plan addresses: <60 LOC, table tests, -race, golangci, compile checks (p227,452).  
   **Must augment:** Enforce *strictly* in phases: extracted funcs split to <60 LOC (ChatHandler beast -> chat() + helpers like renderPrompt, applyStructured etc). *New files small*. Table-driven tests for *all* converters (api <-> pb roundtrips: text, tools, images, think, errors, edge cases). `go test -race -count=1` + `golangci-lint run` (fix; manually ensure no ignored errs despite linter disable) *before* phase complete. `var _ chatv1.ChatServiceHandler = (*chatHandler)(nil)` (and for others; connect generated iface). Fuzz optional for parsers. **Stop if:** func >60 LOC complex logic, no table test for convert, or lint/race fail.

8. **No Globals / Explicit Construction (SKILL p50, checklist p71):**  
   Plan addresses: "No globals" in rules.  
   **Must augment:** *No new package-level vars* for gRPC (no `var grpcServer` or enable flags; GRPCHost() from envconfig returns nil to disable; explicit in cmd). Use Server struct for state (extend minimally at routes:100 if needed, e.g. grpc handlers registered). Config at startup (envconfig explicit, like Host). Manual DI preferred (*Server holds sched). Avoid init() additions (gin one is pre-existing). slog.SetDefault remains in Serve (once). **Stop if:** new global or magic init for grpc.

9. **Agent Orchestration Layer (SKILL p52, checklist p71-72):**  
   Plan addresses: separation in synthesis, todo mentions.  
   **Must augment (THIS DOC ENFORCES):** Planning (arch plan + *this* reliable overlay + protos + research) *fully separate* from execution (code changes in phases). For *any* impl >3 steps (all phases): **use todo_write** (create list at start, update idempotently, merge, mark only when done+verified). Delegate (e.g. sub for converters vs streams). **Verification loops mandatory:** for each phase/task: implement (edits) → review (apply full checklist above + this doc mentally/in notes + run verifs) → fix (until zero issues per gates) → mark todo complete. Rich reasoning *in code logs* (see #5) + in this planning doc. "Suggest using check-work skill or review subagent at end of phases" (e.g. "review the Phase 2 extraction diff through reliable-go lens using SKILL.md + this doc"). **Stop if:** no todo tracking or skipped review-fix loop.

**Flume-specific (checklist p71):** Respect Claimer/Sweeper/Runner (sched is the runner/queue; gRPC reuses via scheduleRunner; no new loaders). State machine in sched (loaded/pending/expired). LogAgentReasoning: rich "reason" logs (see #5).

Apply this checklist *before proposing/committing any code*. Document answers in reasoning/todos.

---

## 3. Logical Phased Approach (Refined + Prioritized for Reliability)

This *refines* the Plan's Phases 0-5 (see grpc-architecture-plan.md:234 for Phase 0, :319 Phase1, :383 Phase2, :488 Phase3, :528 Phase4, :546 Phase5; effort estimates preserved but gates may extend). **Adds:** explicit reliable steps (ctx/errs/logs/bounded per checklist), required log sites/reasoning (targeting LogLoom gaps: instrument uninstr like scheduleRunner + extracted + new grpc paths), test reqs (always -race + grpc+http integ), review checklist application, "stop if" criteria, implement→review→fix loops, todo_write tracking, verif gates between phases.  

**Cross-cutting (all phases):**  
- Follow general rules from Plan p221-229 *plus* full checklist above.  
- Separate planning/execution: this doc + protos = planning artifact. Code changes = execution.  
- **Always:** todo_write at phase start (list sub-tasks), update on progress, only complete when gates pass.  
- Use `component:"grpc"` (new) + "server" (shared extracts) in logs.  
- After *any* change to routes/sched/grpc: re-run logloom + check lift.  
- PRs must reference this doc + show checklist application + before/after logloom.  
- **Implement → review (checklist + this doc) → fix loop:** Do not advance phase until green.

### Refined Phase 0: Foundations (Deps, Proto Skeleton, Buf, Codegen, Minimal Build) — Effort: S (1-2 days); Planning-Heavy
**Goal (Plan p235):** Buildable skeleton, *no runtime change*. (Protos + buf = pure planning; no server code yet.)

**Concrete Tasks + Files (enhanced from Plan p237-293):**  
1-8. (Identical to Plan: go get connect/otelconnect/grpc/otel + -tool gens; buf install + update docs/development.md:5; mkdir proto/ollama/api/v1/; create minimal chat.proto + models.proto + embed.proto (MVP unary+stream Chat/Generate/Embed + List/Version; mirror api/types + xai oneofs later); buf.yaml + buf.gen.yaml at root (go_package github.com/ollama/ollama/gen/proto...); `buf dep update && buf lint && buf generate`; `go mod tidy && go build .`; //go:generate note. Commit generated (like sentencepiece).)  
**New reliable tasks:**  
9. Create initial todo_write for Phase 0 (planning only).  
10. Update docs/development.md with buf + "re-run logloom after changes" + "follow docs/grpc-phased-reliable-approach.md".  
11. Verify no new globals/init/ctx funcs (n/a yet).

**Required Log Sites / Reasoning Points:** None (no logic; note in proto comments: "impl must add slog+reason per reliable overlay #5").

**Test Requirements:** `go build .` clean; `buf build/lint` clean; no tests yet (protos).

**Review Checklist Application (apply before/after each task; document in todo):**  
- Ctx/errs/logs/etc: N/A (no Go logic). Globals: no new. Simplicity: protos small. Agent: this is planning phase; todo used.  
- Run `golangci-lint run` (no change); `go test -race ./...` (skip slow? but full where possible).

**"Stop If" Criteria:** buf lint/generate fails; go build fails (e.g. dep conflict); new global added; docs not updated for reliable process. Do not proceed to code changes.

**Verification Gates (self + gates):**  
- `buf --version && buf dep update && buf lint && buf generate && go mod tidy && go build .`  
- `golangci-lint run` (clean).  
- `go test -race -short ./envconfig ./server` (or equiv; no breakage).  
- Manual: `git status` shows only proto/gen/go.mod/sum/docs; no server/*.go changes.  
- todo_write: list created, all subtasks completed only on green.  
- LogLoom: optional pre (baseline).  
**Gate:** All green + checklist passed + "planning complete" note. Then advance.

**Loop:** (Planning tasks) → review (this section) → fix → mark todo complete.

**Files:** go.mod, proto/ollama/api/v1/*.proto (new), buf.yaml (new), buf.gen.yaml (new), docs/development.md.

**Risks/Mitigations (Plan p314 + reliable):** Buf friction: exact cmds in doc. Proto drift: MVP minimal, 1:1 api first. No reliable violation possible yet.

---

### Refined Phase 1: Listener + Lifecycle Wiring (envconfig, cmd/cmd.go RunServer Dual Listen + Graceful, Serve Extension) — Effort: M (2-3 days)
**Goal (Plan p319):** Dual listeners, shared state, graceful shutdown, *zero HTTP change*. Refactor for extractability. (Wiring = planning + minimal exec; extracts in Phase 2.)

**Concrete Tasks + Files (enhanced from Plan p322-381):**  
1. `envconfig/config.go`: Add `GRPCHost() *url.URL` (exact mirror of Host:22; default "127.0.0.1:11435"; OLLAMA_GRPC_HOST; support disable via empty; update AsMap:311, Values, docs, config_test.go). **Reliable:** explicit construction, no global.  
2. Refactor `server/routes.go:1917 Serve` (critical): Extract `func setupServer(...) (*Server, http.Handler, *Scheduler, ...)` (move prune 1923, s=1951, GenerateRoutes:1968, ctx/sched 1975, modelCaches, GPU 2013, defaultNumCtx 2021, etc. Keep Serve signature/behavior *identical* for compat. Add `func (s *Server) ServeGRPC(ctx context.Context, ln net.Listener) error` (stub: http.Server + h2c per connect, call registerServices; *ctx first*). Add `func (s *Server) registerServices(mux *http.ServeMux)` stub. Add basic logs in stubs with component:"grpc". **Reliable:** ctx first on new ServeGRPC; defer closes; bounded; slog on paths; keep funcs small; update existing scheduleRunner call sites? (defer to 2 for thin).  
3. `cmd/cmd.go:1996 RunServer`: After keypair:2014, compute lnHTTP + (if envconfig.GRPCHost() !=nil ) lnGRPC. `g, ctx := errgroup.WithContext(context.Background())`; `g.SetLimit(2)`. Setup s, h, sched... (use new setup). `g.Go( func() error { ... srvr.Serve(lnHTTP) } )`; if lnGRPC { `g.Go( func() error { return s.ServeGRPC(ctx, lnGRPC) } )` }. Enhance signals (1995): notify, srvr.Close(), grpcSrv.Shutdown(ctx), schedDone(), unloadAllRunners(), done(). `return g.Wait()`. Handle ErrServerClosed. Update env docs. **Reliable (KEY):** errgroup + ctx for dual + shutdown (no fire-forget); ctx prop; no new globals (use envconfig); logs.  
4-5. Minor (p347-348). Update integration/utils_test.go:452 startServer to also set OLLAMA_GRPC_HOST (distinct port or "" to disable for matrix).  

**Required Log Sites / Reasoning Points (target LogLoom gaps):** In new ServeGRPC/register: `slog.Info("starting grpc listener", "component", "grpc", "addr", ln.Addr())`; on shutdown "grpc shutdown complete". In setup (if moved): ensure existing logs (1919 etc) + add "reason" where decisions (e.g. defaultNumCtx choice). Note: this touches Serve (logged) but prep for instrumenting uninstr like GenerateRoutes:1794 (call it).

**Test Requirements:** `go test ./envconfig -run Test -race`; `go build .`; manual dual: `OLLAMA_GRPC_HOST=127.0.0.1:11435 go run . serve` (lsof shows 2 ports; HTTP 11434 unchanged; signals graceful, unload logs); `go test -race ./server -run 'TestRoutes|TestChat|TestGenerate'` (zero regression).

**Review Checklist Application:** Before edits: apply full 9+ above (esp ctx on ServeGRPC, errgroup bounded no fire, no globals, slog keys, todo). After: re-apply + run lint/race. Document in todo: "Ctx first: yes on ServeGRPC. Bounded: errgroup yes. Observ: added 2 logs. No globals: yes (envconfig)."

**"Stop If" Criteria:** Any new func (ServeGRPC) lacks ctx first; no errgroup in RunServer or leaks (test with pprof/ctrl-c); missing slog with component/model on grpc paths; golangci fails or new _=err; race test fails; HTTP regression (curl 11434 vs before); todo not updated. **Do not proceed.**

**Verification Gates:**  
- Env: `OLLAMA_GRPC_HOST=... go run . serve` + signals + dual ports + shared sched (temp debug log "single sched instance").  
- `go test -race ...`; `golangci-lint run`.  
- LogLoom: `~/.local/bin/logloom build && ~/.local/bin/logloom report | grep -E '(grpc|coverage|routes.go|sched.go)'` (note baseline; new logs instrumented).  
- Full checklist answers + todo_write updates (phase tasks completed only on green).  
- Diff: HTTP behavior identical (manual + tests).  
**Gate:** All pass → advance. Use review subagent if complex refactor.

**Files:** envconfig/config.go (+test), cmd/cmd.go:1996, server/routes.go:1917+ (refactor + new ServeGRPC/register), integration/utils_test.go, docs/*.

**Risks (Plan p376 + reliable):** Refactor breaks HTTP: minimal moves + pre/post tests + httptest. Shutdown races: errgroup+ctx explicit. Env edge: copy Host exactly + tests. Violate ctx/bounded: gates catch.

**Loop:** (tasks) → review (checklist/this) → fix (e.g. add missing ctx) → gates → todo complete.

---

### Refined Phase 2: Core Logic Extraction + Thin Adapters — Effort: L (4-5 days, core value; *Execution Starts Here*)
**Goal (Plan p383):** Protocol-agnostic *Server methods; gin handlers thin; gRPC skeleton delegates; converters; first end-to-end.

**Concrete Tasks + Files (enhanced from Plan p386-486):**  
1. **Extraction in `server/routes.go`** (heavy; target Chat/Generate/Embed first; List/Show/Ps easier): Introduce (private):  
   ```go
   // chat handles core chat after bind/cloud validation. write is called for each response.
   // Respects ctx for cancel/backpressure. Returns err (mapped by caller).
   // MUST be <60 LOC or split; ctx first.
   func (s *Server) chat(ctx context.Context, req api.ChatRequest, write func(api.ChatResponse) error) error { ... }
   ```
   Move: name resolution, GetModel (dedupe), scheduleRunner (update to use passed ctx not c.Request if any), render (msgs, harmony, tools, filterThink), ch or direct, llm callback (use write; select on ctx), error paths (return err; let caller map gin.H), prompt/render calls, structured outputs. *Ensure:* all ctx passed, *no gin*. **Add rich slog + reasoning at branches** (see below). Similar for `generate(ctx, api.GenerateRequest, write func...) error`; `embed(ctx, api.EmbedRequest) (*api.EmbedResponse, error)`.  
   Update **gin handlers to thin wrappers** (ChatHandler:2422 etc -> bind/validate/modelRef/cloud guard/GetModel/some, then `if !stream { var final; err := s.chat(c.Request.Context(), req, func(r){final=r}); c.JSON } else { ... bridge ch or direct; use existing streamResponse for compat }`. *Zero behavior change for HTTP.* Test heavily.  
   Handle internal ch/gin.H: in extracted return err; compat bridge for HTTP. Extract helpers stay.  
   **Reliable:** ctx first (chat etc); errors %w + classify; small units; *add logs to instrument uninstr funcs* (scheduleRunner now gets logs inside if not; the extract covers ChatHandler paths).  
2. **Thin gRPC adapters `server/grpc.go` (new file, keep small):**  
   ```go
   package server
   // ...
   type chatHandler struct { s *Server }
   var _ chatv1.ChatServiceHandler = (*chatHandler)(nil)  // compile check
   func (h *chatHandler) Chat(ctx context.Context, req *connect.Request[pb.ChatRequest]) (*connect.Response[pb.ChatResponse], error) { ... }
   func (h *chatHandler) ChatStream(ctx context.Context, req *connect.Request[pb.ChatRequest], stream *connect.ServerStream[pb.ChatResponse]) error {
     apiReq := convertToAPI(req.Msg)
     if isCloud(apiReq.Model) { return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)")) }
     return h.s.chat(ctx, apiReq, func(r api.ChatResponse) error {
       select {
       case <-ctx.Done(): return ctx.Err()
       default:
       }
       if err := stream.Send(convertToPB(r)); err != nil { return err }
       return nil
     })
   }
   func (s *Server) registerServices(mux *http.ServeMux) {
     path, h := chatv1.NewChatServiceHandler(&chatHandler{s}, connect.WithInterceptors(loggingInterceptor(), otelconnect.NewInterceptor(...)))
     mux.Handle(path, h)
     // ...
   }
   ```
   Converters in grpc.go or `server/convert_grpc.go` (table tests): `convertToAPI`, `convertToPB` (handle oneofs, json for format/options, images bytes, orderedmap). Error mapper `errToConnect` (classify + wrap).  
   **Interceptors (basic):** logging (slog + ids: stream_id=uuid, model, dur, component:"grpc", reason?), recovery (panic->err), timeout (from ctx).  
   **Reliable (KEY for streams):** respect stream ctx (use it or ctx param); bounded (select); no gin; thin; compile check; errors wrapped.  
3. Add logs per gaps + reliable (see below).  
4. Unit tests for converters (table-driven roundtrips). Temp ch in streams ok, clean later.  
5. Update routes.go register call.

**Required Log Sites / Reasoning Points (MANDATORY to address 7.67%/50+ uninstr; cite LogLoom):**  
- In extracted `chat`/`generate` (covers former ChatHandler:2422 uninstr paths + scheduleRunner calls): at entry `slog.Info("chat started", "component", "server", "model", req.Model, "stream", req.Stream, "think", req.Think, "stream_id", id)`; after cloud branch: `slog.Debug("cloud branch", "reason", "modelRef.Source==Cloud", "model", ...)` (or error for grpc); post-schedule: `slog.Info("scheduled runner", "component", "server", "model", ..., "load_ms", ..., "reason", "fits in VRAM after X evictions")`; render choice: `slog.Debug("render choice", "reason", "shouldUseHarmony(m) true because family", "model", ...)`; error: `slog.Error("chat prompt error", "error", err, "reason", "template failure")`; on finish: `slog.Info("chat done", "done_reason", ..., "tokens", ..., "duration_ms", ...)`; token emit wrapper: add dur/tokens if possible.  
- In scheduleRunner (currently uninstr per LogLoom; add even if shared): at key decisions (caps fail, numCtxAuto, sched call, runner select) + `slog.Debug("schedule decision", "reason", "evict needed or first load", "model", name, "vram", ...)` .  
- In grpc.go handlers/interceptors: start "grpc unary/stream start", end with dur/status/err/tokens, "reason" for cloud guard or convert issues.  
- In error classify: "classified error as transient" etc.  
Use consistent keys; propagate ids via ctx where possible. This will lift coverage for routes/sched paths + new grpc.

**Test Requirements:** `go test ./server -run 'Test(Chat|Generate|Embed|Route)' -race -count=1` (pre/post diff for HTTP); converters table tests pass; manual: start (grpc port), temp client or buf curl/grpcurl on ChatStream (proto from 0/3); ctx cancel during "stream" stops promptly (unit if possible, check logs/GPU); logs visible with keys/reason; no dupe logic; full matrix (tools, vision, thinking, streaming, errors, cloud error paths for grpc).

**Review Checklist Application:** Before extract: full checklist (ctx on chat(), errs in all moved code, reconciliation via existing sched, bounded in any new go (the callback), *logs+reason on ALL branches*, small funcs, table tests planned, no globals, todo tracking). After each sub-extract: re-apply + run race/lint. "Ctx: all new first-param. Observ: added 10+ sites covering prior uninstr. Reconcile: yes via schedule. Bounded: selects + ctx in writers."

**"Stop If" Criteria:** Extract changes HTTP behavior (tests fail/diff); any func >60LOC; missing ctx first or %w or slog+reason on path (incl scheduleRunner); no table for convert; race/lint fail; cloud guard missing or dupe; ctx cancel doesn't stop (GPU leak risk); todo not tracking. **Do not proceed to Phase 3.**

**Verification Gates:**  
- HTTP identical (full test matrix + manual harmony/gemma4/structured/web_search/remote).  
- gRPC reaches core (temp log "grpc reached extracted chat"); basic unary/stream works (stub or real).  
- Converters roundtrip key structs. Ctx cancel test. Logs with rich attrs/reason.  
- `go test -race`; `golangci-lint run`.  
- LogLoom: `~/.local/bin/logloom build && ~/.local/bin/logloom report` (verify new nodes for chat/grpc funcs; coverage lift; scheduleRunner now instrumented; no uninstr in extracted paths).  
- Checklist + todo_write (phase subtasks complete only on green).  
- Concurrent http+grpc on same model: shared sched, no OOM/contention.  
**Gate:** All + review signoff (self or subagent) → advance.

**Files:** server/routes.go (extracts at ~2422/256/754/205/2616 areas + thin handlers), server/grpc.go (new), server/convert_grpc.go (opt), tests (routes_*_test.go, new grpc unit), api/types (minor).

**Risks (Plan p480 + reliable):** Extraction changes HTTP: copy-paste first then delete; pair/review diffs; *all* tests + manual. Stream semantics: map in bridge. Long funcs: split. Tool/orderedmap quirks: exhaustive table tests. Cloud dupe: centralize. Violate observ/reliable: gates + logloom catch.

**Loop:** (heavy extraction + grpc skeleton + logs) → review (checklist + logs present + race) → fix → gates → todo complete. *This is the high-risk execution phase; use subagent review mid-phase if stuck.*

---

### Refined Phase 3: Proto Surface + Streaming MVP — Effort: M (2-3 days)
**Goal (Plan p488):** Full proto defs + working gRPC Chat stream + others; testable.

**Tasks (enhanced):**  
1. Flesh protos (from Phase 0 skeleton): all fields from api/types (Chat/Generate/Embed/List/Show/Ps/Version/Usage/FinishReason enum/Options as Struct/Message with images/tool_calls/Tools oneof per xai, Think, etc.). Add stream_id support? Enrich. `buf generate`.  
2. Complete adapters in server/grpc.go: impl all MVP (use extracted from 2); converters full; interceptors full (add auth metadata basic). Basic health/reflection (connect equiv or dual).  
3. Test: grpcurl/buf curl on streams/unary/errors/reflection; small Go client example; cancel mid-stream; error codes (bad model=NotFound, caps=Invalid). Update routes register.  
**Reliable adds:** Full logs/reason in all new paths (cover any remaining); ctx respect verified; bounded; err classify; compile checks for all services; table tests expanded; todo tracking.

**Log Sites:** Extend Phase 2: every RPC path, progress if any, health.

**Tests/Verif/StopIf/Gates:** As Plan p498 + Phase2 gates + full logloom re-run (lift confirmed); `go test -race`; manual dual port concurrent; grpcurl works; same sched behavior (ps shows loads from both). Stop if missing logs on new RPCs or ctx not from stream.

**Files:** proto/... (expand), server/grpc.go, gen/proto/... (regen commit), integration/grpc_test.go (new or extend), api/examples/.

**Loop + todo:** Yes.

---

### Refined Phase 4: Polish, Compat, Observability, Tests — Effort: M (2 days)
**Goal (Plan p528):** Full auth/OTEL/error, integ, docs, lint, obs demo, backcompat.

**Tasks (enhanced from Plan p529-542):**  
1-3. Full auth metadata interceptor (permissive local like allowedHosts; early); OTEL full (provider, custom attrs + span around inference/schedule/load, slog bridge); error mapping complete (all from handleScheduleError:3080, StatusError, etc.).  
4. Integration: update utils_test.go:430 startServer for GRPC_HOST (distinct or conditional OLLAMA_TEST_GRPC=0); new grpc tests (connect client, spawn or direct; port mutex safe since serial).  
5. Unit expand (extractors with mock/real sched; converters).  
6. Docs: docs/api.md "## gRPC API" (ports, grpcurl/buf curl ex, protos, env); update envconfig docs.  
7. Linting: `golangci-lint run` (zero); `buf lint`.  
8. Obs demo: logs rich + trace; backcompat HTTP 100% (diff large runs). Version gRPC.  
**Reliable adds:** Re-run logloom (measure final lift in routes/sched/grpc paths; target cover prior 50+ uninstr touched); full checklist re-apply + gates; todo for polish; recommend review subagent.

**Log Sites:** Complete all per #5; add correlation IDs.

**Tests/Verif/StopIf/Gates:** Full suite green; manual dual concurrent no OOM; logs/obs visible; grpcurl clean; `go test -race`; logloom post-build report (coverage, nodes for grpc); checklist documented. Stop if coverage not lifted or logs missing keys/reason on paths. Gate: all + "Phase 4 reliable complete".

**Files:** As Plan + logloom re-run artifacts (not committed).

**Risks:** Port collisions (random/disable); OTEL bloat (optional).

**Loop:** Polish → review (obs + logs + lint + logloom) → fix → gates.

---

### Refined Phase 5: Advanced/Optional — Effort: L (post-MVP, 1-2w+)
**As Plan p546 (cmux opt-in OLLAMA_GRPC_SAMEPORT=1 with soheilhy/cmux + thorough soak; full admin streams; image/video deferred; Buf CI in .github; official clients; richer protos per xai; mTLS auth; load tests; etc.).**  
**Reliable overlay (mandatory if done):** *Higher scrutiny*: full errgroup/bounded for cmux; ctx everywhere; *new* log sites + re-run logloom; table tests + -race + golangci for all; no globals; reconciliation if new workers; review subagent *required* before landing (high risk to existing per Plan); todo_write for the whole; "stop if" any reliable violation or HTTP regression in soak. Defer until stable (Plan agrees). Planning for cmux etc separate from exec.

---

## 4. Specific Recommendations for New Code (Reliable Patterns)

- **gRPC server struct:** `type chatHandler struct { s *Server }`; `var _ chatv1.ChatServiceHandler = (*chatHandler)(nil)` (and equiv for GenerateServiceHandler etc). Thin, delegates to *Server extracted. (See Phase 2 sketch.)

- **Interceptors (register in NewXXXHandler(..., connect.WithInterceptors(...)) ):**  
  Logging: `func loggingInterceptor() connect.Interceptor { return connect.UnaryInterceptorFunc( func(next connect.UnaryFunc) connect.UnaryFunc { return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) { start := time.Now(); id := uuid.New().String(); model := extractModel(req); slog.Info("grpc start", "component", "grpc", "rpc", req.Spec().Procedure(), "model", model, "stream_id", id); res, err := next(ctx, req); slog.Info("grpc done", "component", "grpc", "stream_id", id, "duration_ms", time.Since(start).Milliseconds(), "error", err, "status", statusFromErr(err)); return res, err } }) }` (streaming variant similar; enrich with reasoning on branches). Recovery: wrap next, recover panic -> connect err. Timeout/deadline: enforce from ctx (already in connect/grpc). Auth: parse metadata (authorization/x-ollama-auth), early return if needed (permissive for local dedicated port).

- **Stream handlers:** Always: `ctx := stream.Context()` (or param); `select { case <-ctx.Done(): return ctx.Err(); case <-sendDone: ... }`; bounded buffering `ch := make(chan api.ChatResponse, 64)` from core; in write func: select on ctx before Send/chan; on return (client cancel/Send err): cancel derived ctx to stop llm promptly. No unbounded streams.

- **For scheduler (existing queue/reconcile at sched.go:61+):** If touching (avoid unless necessary): keep rate-limited (buffered chans from MaxQueue), idempotent reconcile on *key* (model key), transient -> requeue equiv. Do *not* store new ctx in structs. gRPC enqueues via existing scheduleRunner path only.

- **Shutdown:** Always `ctx + errgroup` (Phase 1); explicit Close + sched.unloadAllRunners() + <-done; signals already good (enhance for dual). Defer everywhere. Test with kill.

- **New files:** Keep small (< few hundred LOC); funcs <60 LOC; table-driven tests (converters, error maps). E.g. grpc.go for handlers/interceptors, convert_grpc.go for maps (or inline if tiny).

- **LogLoom:** After *any* addition touching routes/sched/grpc (esp Phase 2+): `~/.local/bin/logloom build` (updates graph); `~/.local/bin/logloom lint` or `report`; verify new nodes, reasoning tags, coverage lift for "github.com/ollama/ollama/server/grpc" + prior uninstr now covered (scheduleRunner etc). Plan to re-lint/build in CI later.

- **Other:** Use `context.WithTimeout` on external (llm loads); explicit owners for go; manual DI via *Server; rich reasoning in logs at *every* decision (evict, load choice, render, error class, path branch).

---

## 5. Risk Register + Reliability Mitigations

| Gotcha (from /tmp/*-review*.md + research + LogLoom + sources) | Reliable Pattern Mitigation | SKILL Ref + "Stop If" |
|---------------------------------------------------------------|-----------------------------|-----------------------|
| Gin coupling (bind, c.Stream, gin.H errors, writer hacks; ollamas-review:194, routes:2070/2422) | Extract to ctx+api-only Server.chat/generate/embed (thin gin/grpc wrappers only); no gin in core. | Context is King + Simplicity/Verifiability. Stop if: gin sneaks into extracted or HTTP behavior changes. |
| Cloud intertwining in *every* handler (modelRef.Source==Cloud -> proxyCloudJSONRequest gin-tied; ollamas-review:195, routes:2445/278) | Early guard in gRPC adapter + extracted (clear sentinel err "HTTP-only for cloud"); centralize parseAndValidateModelRef; no gRPC cloud MVP. | Errors as Values + No Globals. Stop if: cloud path dupe or gRPC accepts cloud without error. |
| Scheduler/GPU contention + leaks on cancel (shared *Server mandatory; ctx prop critical; ollamas-review:198, grpc-research:138, sched:31 containedctx, routes:2616) | *Mandatory* single *Server*/sched (via setup); ctx-first everywhere + derived cancel from stream.Context(); select on Done in writers/callbacks; bounded; on return cancel to stop gen + unload; *never* store ctx in *new* structs (note existing LlmRequest). | Context is King + Bounded Everything + Reconciliation (via sched). Stop if: dual sched, missing cancel, or ctx in new struct. |
| Low log coverage (7.67% =398/5191; 50+ uninstr routes incl scheduleRunner:205/ChatHandler:2422/GenerateRoutes:1794 +50 others; 52 sched incl getRunner/Init; clustered; ollamas-review:201, LogLoom query) | *Every* sig path (accept/schedule decision/token emit/error classify/load/evict/finish/cloud/render) + *all* extracted/grpc emit slog with {component, model, stream_id, status, duration_ms, error?, tokens?, reason? for decisions}; rich Flume reasoning at branches; post-change logloom build/report/lint to measure lift + cover gaps. | Observability is Not Optional. Stop if: path without slog+keys/reason, or post-logloom no lift/no nodes for new/touched. |
| Integration test mutex/ports/serial binary (utils_test.go:444 global serverMutex/serverCmd; ollamas-review:201) | Set OLLAMA_GRPC_HOST (distinct port e.g. 11436) or "" (disable) in startServer; new grpc tests opt-in (env/tag); rely on serial; matrix http/grpc. | Verifiability (tests + -race). Stop if: test collision or no grpc coverage in integ. |
| No compile checks / long funcs / globals / init (current .golangci, routes:96 vars, sched:87 var, ChatHandler length) | var _ I = (*T)(nil) for handlers; split extracts <60LOC; no *new* package globals (envconfig + Server fields); no init adds. | Simplicity... + No Globals. Stop if: missing check, long func, new global. |
| Other (high conn/long streams, binary debug, auth unauthed, test mutex, C++ isolation, refactor risk; Plan p562) | Interceptors (timeout/keepalive/auth early + recovery); Connect (JSON compat + reflection + grpcurl + rich status); permissive local + docs for prod; gates + -race; pure Go additive. | Bounded + Idempotency + Observ. Stop if: no backpressure or unauthed prod surface without note. |

All map directly to SKILL; gates in phases close them.

---

## 6. Verification & Gates

**Self-Verify (run before any commit/phase advance; document):**  
1. `go test -race -count=1 ./server -run 'Test.*(Chat|Generate|Embed|Route|.*Handler|schedule)' && go test -race ./envconfig ./cmd` (full where feasible).  
2. `golangci-lint run` (clean; manually confirm no ignored errs in new paths despite disable).  
3. Manual dual: `OLLAMA_GRPC_HOST=127.0.0.1:11435 go run . serve` (in bg); curl http://127.0.0.1:11434/api/tags; grpcurl --plaintext localhost:11435 list; buf curl or small client for ChatStream (tokens incremental, done, usage); concurrent load same model from both ports (ps shows 1 load, no OOM); client cancel mid-gen (check logs "context for request finished", no hung runner/GPU leak via top/ps or llm logs); ctrl-c (graceful, unload logs, exit 0).  
4. Grep/inspect: new slog calls have required keys + "reason" at decisions; ctx first on all I/O funcs; %w on errs; no new globals.  
5. Diff: HTTP responses/behavior identical pre/post (capture test output or integ).  
6. For streams/extract: explicit ctx cancel test (unit or manual).  

**LogLoom (mandatory after routes/sched/grpc changes, esp Phase 2+):**  
`~/.local/bin/logloom build` (updates logloom-graph.json); `~/.local/bin/logloom report > /tmp/post.log` or `logloom lint`; `python3 -c 'import json,d=json.load(open("logloom-graph.json"));c=d["coverage"];print(c); print("grpc nodes:", [n for n in d["nodes"] if "grpc" in str(n)])'`; grep uninstr for "scheduleRunner|chat|grpc". Verify lift (e.g. scheduleRunner now has node, grpc funcs instrumented, overall >7.67% or specific paths covered). Commit? No, but note in PR.

**Apply Checklist:** Before/after every edit: answer the 10 SKILL questions (in thought/todos). "Have I added rich reasoning logs?" etc.

**Gates (non-negotiable; "stop if" in phases are these):** All verif + checklist green + todo subtasks complete (only when verified) + no regression + LogLoom lift (where applicable). If ctx not propagated, missing slog/reason on path, lint/race fail, HTTP change, new global, etc: **do not proceed/merge/advance phase**.

**Cross-Phase/End:** At end of Phase 2 (core extract) and Phase 4 (polish): "recommend using check-work skill or delegating to review subagent with: 'Review the changes/diff through the reliable-go-systems lens (SKILL.md) + docs/grpc-phased-reliable-approach.md. Apply the mandatory checklist. Check ctx/errs/logs/reconcile/bounded/observ/no globals/idemp/simplicity. Verify LogLoom lift and gates. Report issues with file:line quotes.' " Self-review using this doc as overlay.

**Structured Tracking:** todo_write for *all* impl (see initial todos in this session for example; idempotent updates; merge=true for partial). "Use for anything >3 steps."

**Other:** `go test ./...` per dev docs; buf in prereqs; PR explains deps (CONTRIBUTING), links this doc + arch plan, shows reliable checklist + logloom before/after.

---

**End of Reliable Overlay.** This is the production reliability contract for the gRPC work. Follow phases sequentially with loops/gates. After Phase 4: working, observable, safe gRPC (Chat/Embed/Models streams/unary on 11435), zero HTTP impact, shared core/scheduler, *measurable* observability lift, all reliable patterns enforced. Next: Phase 0 PR (proto+buf+docs only), then iterate with todos + reviews.

(Tracked via todo_write throughout this synthesis + prior research; 13+ items managed idempotently. All grounded in primary reads/greps/queries of listed must-reads + sources. This doc + arch plan = complete planning artifacts.)

**References (internal):** SKILL.md quotes p13-80; Plan phases/lines as cited; LogLoom stats from direct python/json query on 2026-06-03 build; reviews [grpc-research:138 etc], [ollamas-review:192-205]; absolute paths as in user_info/workspace.

---

## Summary for Implementers (from Reliability Specialist)
- Read *all* must-reads first (use tools).  
- Planning done (this + arch plan). Execution follows refined phases + loops + gates.  
- Voice of reliability: "Context is King", "Observability is Not Optional", "Reconciliation over Event-Driven", "Bounded Everything", "Errors Are Values", "Idempotency + Safe Retries", "Simplicity... Verifiability", "No Globals", "Agent Orchestration Layer (planning vs execution + todo + review-fix)". Quote and apply SKILL checklist on every step.  
- Output of this task: the doc at `docs/grpc-phased-reliable-approach.md`. Use it as the filter for all gRPC code.

**Path to doc:** `/Users/jonathandoughty/clients/fremenlabs/ollamas/ollamas/docs/grpc-phased-reliable-approach.md` (written via tool after full review/synthesis + todos).
