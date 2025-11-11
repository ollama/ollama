# PHASE 10: AGENT SYSTEM (İki Model Birlikte Çalışma)

## 📋 HEDEFLER
1. ✅ Dual-model agent architecture
2. ✅ Supervisor model (kuralları denetler)
3. ✅ Worker model (işleri yapar)
4. ✅ Todo.md bazlı execution
5. ✅ Phase-by-phase processing
6. ✅ Automatic testing & validation
7. ✅ Progress reporting
8. ✅ Context summarization (phase arası)

## 🏗️ MİMARİ

```
USER
  ↓
Agent Controller
  ↓
┌──────────────────────────────────────┐
│   SUPERVISOR MODEL (Model A)          │
│   - Reads .leah/todo.md              │
│   - Reads .leah/rules.md             │
│   - Plans execution                   │
│   - Validates results                 │
│   - Decides next steps                │
└──────────┬───────────────────────────┘
           │
           ↓ (sends instructions)
┌──────────────────────────────────────┐
│   WORKER MODEL (Model B)              │
│   - Executes tasks                    │
│   - Writes code                       │
│   - Runs tests                        │
│   - Returns results                   │
└──────────┬───────────────────────────┘
           │
           ↓ (results)
┌──────────────────────────────────────┐
│   SUPERVISOR MODEL                    │
│   - Validates results                 │
│   - Tests against criteria            │
│   - Decides: pass/retry/fail          │
└──────────┬───────────────────────────┘
           │
           ↓
     Next Phase OR Retry
```

### Schema
```sql
CREATE TABLE agent_sessions (
  id TEXT PRIMARY KEY,
  workspace_id TEXT,
  supervisor_provider_id TEXT,
  supervisor_model TEXT,
  worker_provider_id TEXT,
  worker_model TEXT,
  status TEXT,  -- running, paused, completed, failed
  current_phase INTEGER,
  started_at TIMESTAMP,
  completed_at TIMESTAMP
);

CREATE TABLE agent_phase_executions (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  phase_index INTEGER,
  phase_name TEXT,
  status TEXT,  -- pending, running, success, failed, retry
  attempts INTEGER DEFAULT 0,
  worker_output TEXT,
  supervisor_feedback TEXT,
  test_results TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES agent_sessions(id)
);

CREATE TABLE agent_messages (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  phase_execution_id INTEGER,
  role TEXT,  -- supervisor, worker, system
  content TEXT,
  timestamp TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES agent_sessions(id),
  FOREIGN KEY (phase_execution_id) REFERENCES agent_phase_executions(id)
);
```

## 📁 DOSYALAR

### 1. Agent Controller
**Dosya:** `/home/user/ollama/agent/controller.go` (YENİ)

```go
package agent

type AgentController struct {
    supervisor providers.Provider
    worker     providers.Provider
    workspace  *workspace.Manager
    todoMgr    *workspace.TodoManager
    rulesMgr   *workspace.RulesManager
}

type AgentSession struct {
    ID            string
    WorkspaceID   string
    Supervisor    ModelConfig
    Worker        ModelConfig
    Status        string
    CurrentPhase  int
}

type ModelConfig struct {
    ProviderID string
    ModelName  string
}

func (ac *AgentController) StartSession(ws *workspace.Workspace, supervisor, worker ModelConfig) (*AgentSession, error) {
    // Load todo list
    todos, err := ac.todoMgr.GetTodos(ws.Path)
    if err != nil {
        return nil, err
    }

    // Load rules
    rules, err := ac.rulesMgr.GetRules(ws.Path)
    if err != nil {
        return nil, err
    }

    session := &AgentSession{
        ID:           uuid.New().String(),
        WorkspaceID:  ws.ID,
        Supervisor:   supervisor,
        Worker:       worker,
        Status:       "running",
        CurrentPhase: 0,
    }

    // Start execution goroutine
    go ac.executeSession(session, todos, rules)

    return session, nil
}

func (ac *AgentController) executeSession(session *AgentSession, todos *workspace.TodoList, rules *workspace.Rules) {
    ctx := context.Background()

    for phaseIdx, phase := range todos.Phases {
        session.CurrentPhase = phaseIdx

        log.Printf("Starting Phase %d: %s", phaseIdx+1, phase.Name)

        // Execute phase with retry logic
        success := false
        maxAttempts := 3

        for attempt := 1; attempt <= maxAttempts; attempt++ {
            log.Printf("Phase %d, Attempt %d/%d", phaseIdx+1, attempt, maxAttempts)

            // Supervisor: Plan the phase
            supervisorPlan, err := ac.supervisorPlanPhase(ctx, phase, rules)
            if err != nil {
                log.Printf("Supervisor planning failed: %v", err)
                continue
            }

            // Worker: Execute the plan
            workerResult, err := ac.workerExecute(ctx, supervisorPlan, rules)
            if err != nil {
                log.Printf("Worker execution failed: %v", err)
                continue
            }

            // Supervisor: Validate results
            validation, err := ac.supervisorValidate(ctx, phase, workerResult, rules)
            if err != nil {
                log.Printf("Supervisor validation failed: %v", err)
                continue
            }

            if validation.Success {
                log.Printf("Phase %d completed successfully", phaseIdx+1)
                success = true

                // Mark phase as complete in todo.md
                ac.todoMgr.MarkPhaseComplete(session.WorkspaceID, phaseIdx)

                break
            } else {
                log.Printf("Phase %d validation failed: %s", phaseIdx+1, validation.Reason)

                // Ask supervisor for correction
                correction, err := ac.supervisorCorrect(ctx, phase, validation, rules)
                if err != nil {
                    log.Printf("Supervisor correction failed: %v", err)
                    continue
                }

                // Worker: Apply correction
                workerResult, err = ac.workerExecute(ctx, correction, rules)
                if err != nil {
                    log.Printf("Worker correction execution failed: %v", err)
                    continue
                }

                // Re-validate
                validation, err = ac.supervisorValidate(ctx, phase, workerResult, rules)
                if err == nil && validation.Success {
                    log.Printf("Phase %d completed after correction", phaseIdx+1)
                    success = true
                    ac.todoMgr.MarkPhaseComplete(session.WorkspaceID, phaseIdx)
                    break
                }
            }
        }

        if !success {
            log.Printf("Phase %d failed after %d attempts", phaseIdx+1, maxAttempts)
            session.Status = "failed"
            ac.generateReport(session, phaseIdx, false)
            return
        }

        // Context summarization before next phase
        if phaseIdx < len(todos.Phases)-1 {
            ac.summarizeContext(session, phaseIdx)
        }
    }

    // All phases completed
    session.Status = "completed"
    ac.generateReport(session, len(todos.Phases)-1, true)
}

func (ac *AgentController) supervisorPlanPhase(ctx context.Context, phase *workspace.Phase, rules *workspace.Rules) (string, error) {
    prompt := fmt.Sprintf(`You are a supervisor AI. Your task is to plan the execution of this phase:

Phase: %s
Status: %s

Tasks:
%s

Test Criteria:
%s

Rules to follow:
%s

Create a detailed execution plan for the worker AI. Be specific and clear.`,
        phase.Name,
        phase.Status,
        formatTasks(phase.Tasks),
        formatTestPlan(phase.TestPlan),
        rules.ToSystemPrompt(),
    )

    req := providers.ChatRequest{
        Model: ac.supervisor.ModelName,
        Messages: []providers.Message{
            {Role: "user", Content: prompt},
        },
    }

    resp, err := ac.supervisor.ChatCompletion(ctx, req)
    if err != nil {
        return "", err
    }

    return resp.Message.Content, nil
}

func (ac *AgentController) workerExecute(ctx context.Context, plan string, rules *workspace.Rules) (string, error) {
    prompt := fmt.Sprintf(`Execute the following plan:

%s

Rules:
%s

Provide the complete implementation with all necessary code and files.`,
        plan,
        rules.ToSystemPrompt(),
    )

    req := providers.ChatRequest{
        Model: ac.worker.ModelName,
        Messages: []providers.Message{
            {Role: "user", Content: prompt},
        },
        Tools: getFileTools(),  // Enable file operations
    }

    resp, err := ac.worker.ChatCompletion(ctx, req)
    if err != nil {
        return "", err
    }

    return resp.Message.Content, nil
}

type ValidationResult struct {
    Success bool
    Reason  string
    Details map[string]interface{}
}

func (ac *AgentController) supervisorValidate(ctx context.Context, phase *workspace.Phase, workerResult string, rules *workspace.Rules) (*ValidationResult, error) {
    prompt := fmt.Sprintf(`Validate the following implementation against the test criteria:

Phase: %s

Test Criteria:
%s

Implementation:
%s

Rules:
%s

Respond with JSON:
{
  "success": true/false,
  "reason": "explanation",
  "details": {}
}`,
        phase.Name,
        formatTestPlan(phase.TestPlan),
        workerResult,
        rules.ToSystemPrompt(),
    )

    req := providers.ChatRequest{
        Model: ac.supervisor.ModelName,
        Messages: []providers.Message{
            {Role: "user", Content: prompt},
        },
    }

    resp, err := ac.supervisor.ChatCompletion(ctx, req)
    if err != nil {
        return nil, err
    }

    // Parse JSON response
    var result ValidationResult
    if err := json.Unmarshal([]byte(resp.Message.Content), &result); err != nil {
        return nil, err
    }

    return &result, nil
}

func (ac *AgentController) generateReport(session *AgentSession, finalPhase int, success bool) {
    report := fmt.Sprintf(`# Agent Execution Report

**Session ID:** %s
**Workspace:** %s
**Status:** %s
**Completed Phases:** %d

## Models Used
- **Supervisor:** %s
- **Worker:** %s

## Summary
%s

**Generated:** %s
`,
        session.ID,
        session.WorkspaceID,
        session.Status,
        finalPhase+1,
        session.Supervisor.ModelName,
        session.Worker.ModelName,
        generateSummary(session, success),
        time.Now().Format("2006-01-02 15:04:05"),
    )

    // Save to workspace
    ws, _ := ac.workspace.GetWorkspace(session.WorkspaceID)
    reportPath := filepath.Join(ws.Path, fmt.Sprintf("report_%s.md", time.Now().Format("20060102_150405")))
    os.WriteFile(reportPath, []byte(report), 0644)
}
```

### 2. Agent UI Component
**Dosya:** `/home/user/ollama/app/ui/app/src/components/AgentRunner.tsx` (YENİ)

```typescript
export function AgentRunner() {
  const { data: workspace } = useActiveWorkspace();
  const { data: providers } = useProviders();
  const startAgent = useStartAgent();

  const [supervisorConfig, setSupervisorConfig] = useState({
    providerId: '',
    model: '',
  });

  const [workerConfig, setWorkerConfig] = useState({
    providerId: '',
    model: '',
  });

  const handleStart = async () => {
    if (!workspace) return;

    await startAgent.mutateAsync({
      workspaceId: workspace.id,
      supervisor: supervisorConfig,
      worker: workerConfig,
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">Agent System</h2>

        <div className="grid grid-cols-2 gap-6">
          {/* Supervisor Config */}
          <div>
            <h3 className="font-semibold mb-2">Supervisor Model</h3>
            <p className="text-sm text-gray-600 mb-3">
              This model will read rules, plan execution, and validate results.
            </p>
            <ModelSelector
              providers={providers}
              selected={supervisorConfig}
              onChange={setSupervisorConfig}
            />
          </div>

          {/* Worker Config */}
          <div>
            <h3 className="font-semibold mb-2">Worker Model</h3>
            <p className="text-sm text-gray-600 mb-3">
              This model will execute tasks and write code.
            </p>
            <ModelSelector
              providers={providers}
              selected={workerConfig}
              onChange={setWorkerConfig}
            />
          </div>
        </div>

        <button
          onClick={handleStart}
          disabled={!supervisorConfig.model || !workerConfig.model}
          className="mt-6 px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
        >
          Start Agent Execution
        </button>
      </div>

      {/* Live Progress */}
      <AgentProgress />
    </div>
  );
}
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ Dual-model system çalışıyor
2. ✅ Todo.md'den phase okuyor
3. ✅ Supervisor doğru plan yapıyor
4. ✅ Worker doğru execute ediyor
5. ✅ Validation çalışıyor
6. ✅ Retry logic çalışıyor
7. ✅ Final report oluşuyor

**SONRAKİ:** Phase 11 - Advanced Features
