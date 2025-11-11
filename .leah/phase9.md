# PHASE 9: WORKSPACE INTEGRATION (Kod Okuma/Yazma/Editleme)

## 📋 HEDEFLER
1. ✅ Workspace dosya ağacı görüntüleme
2. ✅ Dosya okuma izni (kullanıcı onayı)
3. ✅ Dosya yazma/editleme (kullanıcı onayı)
4. ✅ Dosya silme (kullanıcı onayı)
5. ✅ Syntax highlighting
6. ✅ Diff görüntüleme
7. ✅ Tool calling entegrasyonu

## 🏗️ MİMARİ

### File Tools Schema
```sql
CREATE TABLE file_operations (
  id INTEGER PRIMARY KEY,
  chat_id TEXT,
  message_id INTEGER,
  operation TEXT,  -- read, write, delete, edit
  file_path TEXT,
  approved BOOLEAN DEFAULT 0,
  executed BOOLEAN DEFAULT 0,
  diff TEXT,
  timestamp TIMESTAMP
);
```

### Tool Definitions
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "File path relative to workspace"
            }
          },
          "required": ["path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string" },
            "content": { "type": "string" }
          },
          "required": ["path", "content"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit_file",
        "description": "Edit specific lines in a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string" },
            "start_line": { "type": "integer" },
            "end_line": { "type": "integer" },
            "new_content": { "type": "string" }
          },
          "required": ["path", "start_line", "end_line", "new_content"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string" }
          }
        }
      }
    }
  ]
}
```

## 📁 DOSYALAR

### 1. File Tool Handler
**Dosya:** `/home/user/ollama/tools/file_tools.go` (YENİ)

```go
type FileToolHandler struct {
    workspace *workspace.Manager
    approver  *OperationApprover
}

func (fth *FileToolHandler) HandleReadFile(path string) (string, error) {
    // Security check
    if !fth.isPathSafe(path) {
        return "", errors.New("unsafe file path")
    }

    ws, err := fth.workspace.GetActiveWorkspace()
    if err != nil {
        return "", err
    }

    fullPath := filepath.Join(ws.Path, path)

    // Check if approved
    if !fth.approver.IsApproved("read", fullPath) {
        return "", errors.New("operation requires user approval")
    }

    content, err := os.ReadFile(fullPath)
    if err != nil {
        return "", err
    }

    return string(content), nil
}

func (fth *FileToolHandler) HandleWriteFile(path, content string) error {
    if !fth.isPathSafe(path) {
        return errors.New("unsafe file path")
    }

    ws, err := fth.workspace.GetActiveWorkspace()
    if err != nil {
        return err
    }

    fullPath := filepath.Join(ws.Path, path)

    // Request approval
    approved, err := fth.approver.RequestApproval("write", fullPath, content)
    if err != nil || !approved {
        return errors.New("write operation denied")
    }

    // Create directories if needed
    dir := filepath.Dir(fullPath)
    if err := os.MkdirAll(dir, 0755); err != nil {
        return err
    }

    return os.WriteFile(fullPath, []byte(content), 0644)
}

func (fth *FileToolHandler) isPathSafe(path string) bool {
    // Prevent directory traversal
    if strings.Contains(path, "..") {
        return false
    }

    // Prevent absolute paths
    if filepath.IsAbs(path) {
        return false
    }

    return true
}
```

### 2. Approval UI Component
**Dosya:** `/home/user/ollama/app/ui/app/src/components/FileOperationApproval.tsx` (YENİ)

```typescript
export function FileOperationApproval({ operation }: { operation: FileOperation }) {
  const approve = useApproveOperation();
  const reject = useRejectOperation();

  const [showDiff, setShowDiff] = useState(false);

  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <ExclamationTriangleIcon className="h-6 w-6 text-yellow-600" />
        <div className="flex-1">
          <h4 className="font-semibold text-yellow-800 dark:text-yellow-200">
            Permission Required
          </h4>
          <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
            AI wants to <span className="font-mono">{operation.operation}</span> file:
          </p>
          <code className="block mt-2 p-2 bg-yellow-100 dark:bg-yellow-900/40 rounded text-sm">
            {operation.file_path}
          </code>

          {operation.operation === 'write' || operation.operation === 'edit' ? (
            <div className="mt-3">
              <button
                onClick={() => setShowDiff(!showDiff)}
                className="text-sm text-indigo-600 dark:text-indigo-400 underline"
              >
                {showDiff ? 'Hide' : 'Show'} changes
              </button>

              {showDiff && operation.diff && (
                <pre className="mt-2 p-3 bg-gray-900 text-gray-100 rounded text-xs overflow-x-auto">
                  <code>{operation.diff}</code>
                </pre>
              )}
            </div>
          ) : null}

          <div className="flex gap-2 mt-4">
            <button
              onClick={() => approve.mutate(operation.id)}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
            >
              Approve
            </button>
            <button
              onClick={() => reject.mutate(operation.id)}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
            >
              Reject
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

### 3. File Explorer
**Dosya:** `/home/user/ollama/app/ui/app/src/components/FileExplorer.tsx` (YENİ)

```typescript
export function FileExplorer({ workspacePath }: { workspacePath: string }) {
  const { data: files } = useWorkspaceFiles(workspacePath);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());

  const toggleDir = (path: string) => {
    const newExpanded = new Set(expandedDirs);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedDirs(newExpanded);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <h3 className="font-semibold mb-3">Workspace Files</h3>
      <div className="space-y-1">
        {files?.map(file => (
          <FileTreeNode
            key={file.path}
            file={file}
            expanded={expandedDirs.has(file.path)}
            onToggle={() => toggleDir(file.path)}
          />
        ))}
      </div>
    </div>
  );
}
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ File explorer çalışıyor
2. ✅ Tool calling file operations çalışıyor
3. ✅ Approval UI gösteriliyor
4. ✅ Diff view doğru gösteriliyor
5. ✅ Security checks çalışıyor

**SONRAKİ:** Phase 10 - Agent System
