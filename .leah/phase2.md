# PHASE 2: KURALLAR VE TODO SİSTEMİ (.leah klasör yapısı)

## 📋 GENEL BAKIŞ
Bu phase'de kullanıcıların AI modeline kurallar tanımlayabileceği ve todo list yapabileceği bir sistem oluşturuyoruz. Kurallar `.leah/rules.md` dosyasında saklanacak ve her chat request'inde otomatik olarak system prompt'a eklenecek.

## 🎯 HEDEFLER
1. ✅ `.leah` klasör yapısı oluşturma
2. ✅ `rules.md` - Model davranış kuralları
3. ✅ `todo.md` - Görev listesi
4. ✅ GUI ile kural oluşturma/düzenleme
5. ✅ GUI ile todo list yönetimi
6. ✅ Örnek kurallar template'i
7. ✅ Kuralları chat request'lerine otomatik ekleme
8. ✅ Todo takibi ve durum güncellemeleri

---

## 🏗️ MİMARİ TASARIM

### .leah Klasör Yapısı
```
workspace/
└── .leah/
    ├── rules.md              # Model kuralları
    ├── todo.md               # Görev listesi
    ├── templates/            # Kural template'leri
    │   ├── web-dev.md       # Web geliştirme kuralları
    │   ├── backend.md       # Backend kuralları
    │   ├── mobile.md        # Mobile kuralları
    │   └── data-science.md  # Data science kuralları
    ├── history/              # Değişiklik geçmişi
    │   ├── rules-history.json
    │   └── todo-history.json
    └── config.json           # Konfigürasyon
```

### Klasör Yapısı (Kod Tarafı)
```
/home/user/ollama/
├── workspace/
│   ├── manager.go           # YENİ: Workspace yöneticisi
│   ├── rules.go             # YENİ: Rules parser ve manager
│   └── todo.go              # YENİ: Todo parser ve manager
├── app/store/
│   └── schema.sql           # GÜNCELLENECEK: Workspace tablolar
└── app/ui/app/src/
    ├── hooks/
    │   ├── useRules.ts      # YENİ: Rules hook
    │   ├── useTodo.ts       # YENİ: Todo hook
    │   └── useWorkspace.ts  # YENİ: Workspace hook
    ├── components/
    │   ├── RulesEditor.tsx  # YENİ: Kural editörü
    │   ├── TodoManager.tsx  # YENİ: Todo yöneticisi
    │   └── WorkspaceSelector.tsx  # YENİ: Workspace seçici
    └── routes/
        ├── workspace/
        │   ├── index.tsx    # Workspace ana sayfa
        │   ├── rules.tsx    # Rules editor sayfası
        │   └── todo.tsx     # Todo manager sayfası
        └── settings/
            └── workspace.tsx # Workspace ayarları
```

---

## 📁 DETAYLI DOSYA DEĞİŞİKLİKLERİ

### 1. DATABASE SCHEMA GÜNCELLEMESİ

**Dosya:** `/home/user/ollama/app/store/schema.sql`

**Eklenecek Tablolar:**

```sql
-- Workspaces
CREATE TABLE IF NOT EXISTS workspaces (
  id TEXT PRIMARY KEY,                    -- UUID
  name TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,              -- Workspace path
  active BOOLEAN DEFAULT 0,               -- Aktif workspace
  rules_enabled BOOLEAN DEFAULT 1,
  todo_enabled BOOLEAN DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rules (cached from rules.md)
CREATE TABLE IF NOT EXISTS rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT NOT NULL,
  category TEXT NOT NULL,                 -- prohibitions, requirements, code_style, etc.
  rule_text TEXT NOT NULL,
  priority INTEGER DEFAULT 0,             -- 0=normal, 1=high, 2=critical
  enabled BOOLEAN DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Todos (cached from todo.md)
CREATE TABLE IF NOT EXISTS todos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT NOT NULL,
  phase TEXT,                             -- phase1, phase2, etc.
  title TEXT NOT NULL,
  description TEXT,
  status TEXT DEFAULT 'pending',          -- pending, in_progress, completed, failed
  parent_todo_id INTEGER,                 -- For subtasks
  order_index INTEGER DEFAULT 0,          -- Display order
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_todo_id) REFERENCES todos(id) ON DELETE CASCADE
);

-- Todo Checklist Items
CREATE TABLE IF NOT EXISTS todo_checklist (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  todo_id INTEGER NOT NULL,
  text TEXT NOT NULL,
  checked BOOLEAN DEFAULT 0,
  order_index INTEGER DEFAULT 0,
  FOREIGN KEY (todo_id) REFERENCES todos(id) ON DELETE CASCADE
);

-- Rule/Todo History (for version control)
CREATE TABLE IF NOT EXISTS leah_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT NOT NULL,
  file_type TEXT NOT NULL,                -- 'rules' or 'todo'
  content TEXT NOT NULL,                  -- Full file content
  change_summary TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Settings tablosuna eklenecek
ALTER TABLE settings ADD COLUMN active_workspace_id TEXT;

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_rules_workspace ON rules(workspace_id);
CREATE INDEX IF NOT EXISTS idx_todos_workspace ON todos(workspace_id);
CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
CREATE INDEX IF NOT EXISTS idx_leah_history_workspace ON leah_history(workspace_id, file_type);
```

---

### 2. RULES.MD TEMPLATE

**Dosya:** `.leah/rules.md` (Template - kullanıcı workspace'inde oluşturulacak)

```markdown
# AI Model Kuralları

Bu dosya, AI modelinin davranışını ve tercihlerini belirler. Her chat oturumunda bu kurallar otomatik olarak sisteme eklenir.

---

## 🚫 YASAKLAR VE KESINLIKLE YAPILMAMASI GEREKENLER

Bu kurallara kesinlikle uyulmalıdır. Bu kuralları ihlal eden kod veya öneriler kabul edilmez.

- CDN kullanma, tüm bağımlılıklar lokal olmalı
- Harici servislere bağımlılık oluşturma (Google Fonts, analytics, vb.)
- Placeholder veri kullanma, gerçek implementasyon yap
- Test yazmadan kod teslim etme
- Error handling olmadan asenkron kod yazma
- Global değişken kullanma (const/let kullan)
- Console.log'ları production koduna bırakma
- Hardcoded credential kullanma
- SQL injection'a açık kod yazma
- XSS vulnerabilities oluşturma

---

## ✅ ZORUNLU UYULMASI GEREKENLER

Bu kurallar her zaman uygulanmalıdır.

- Her fonksiyon için JSDoc/TypeDoc yorumları yaz
- Değişken isimleri açıklayıcı ve anlamlı olmalı
- Kodları test etmeden teslim etme
- Git commit'lerini anlamlı mesajlarla yap
- Code review öncesi linter'dan geçir
- Type safety kullan (TypeScript, Go types)
- Error handling her zaman implement et
- Logging ekle (development için verbose, production için structured)
- Performance metrics topla (response time, memory usage)
- Security best practices uygula (OWASP Top 10)

---

## 💻 KOD YAZMA KURALLARI

### Genel Kurallar
- **Naming Convention:** camelCase (JS/TS), PascalCase (components, classes), snake_case (Go)
- **Indentation:** 2 spaces (JS/TS), tabs (Go)
- **Line Length:** Max 100 characters
- **File Length:** Max 300 lines (refactor if longer)
- **Function Length:** Max 50 lines (refactor if longer)
- **Complexity:** Max cyclomatic complexity 10

### TypeScript/JavaScript
```typescript
// ✅ İyi örnek
interface User {
  id: string;
  name: string;
  email: string;
}

async function fetchUser(userId: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching user:', error);
    throw error;
  }
}

// ❌ Kötü örnek
function getUser(id) {
  return fetch('/api/users/' + id).then(r => r.json());
}
```

### Go
```go
// ✅ İyi örnek
type User struct {
    ID    string `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func (s *Server) GetUser(ctx context.Context, userID string) (*User, error) {
    if userID == "" {
        return nil, fmt.Errorf("user ID is required")
    }

    user, err := s.db.QueryUser(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to query user: %w", err)
    }

    return user, nil
}

// ❌ Kötü örnek
func GetUser(id string) *User {
    user, _ := db.Query(id)
    return user
}
```

---

## 🗄️ DATABASE KURALLARI

- Migration kullan (manual schema change yapma)
- Foreign keys tanımla
- Index'leri optimize et (N+1 problem'i önle)
- Prepared statements kullan (SQL injection prevention)
- Transaction kullan (ACID compliance)
- Soft delete tercih et (hard delete yerine)

### SQL Örnek
```sql
-- ✅ İyi
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;

-- ❌ Kötü
CREATE TABLE users (
  id TEXT,
  email TEXT
);
```

---

## 🎨 TERCIH EDILEN PROGRAMLAMA DİLİ

**Primary:** TypeScript (Frontend), Go (Backend)
**Secondary:** Python (Scripts, ML), SQL (Database)
**Avoid:** JavaScript (use TypeScript instead), Bash (use Go for scripts)

---

## 🖥️ TERCIH EDILEN ARAYÜZ DİLİ/FRAMEWORK

**Frontend:**
- React 19+ (functional components only)
- TypeScript (strict mode)
- TailwindCSS (utility-first, no custom CSS unless necessary)
- TanStack Query (server state)
- TanStack Router (routing)

**Backend:**
- Go 1.24+ (modern Go)
- Gin (HTTP framework)
- SQLite (embedded database)

**Avoid:**
- jQuery
- Bootstrap (use TailwindCSS)
- Class components (use functional)
- Redux (use TanStack Query + Context API)

---

## 📦 TERCIH EDILEN PAKETLER/KÜTÜPHANELER

### Frontend
- `@tanstack/react-query` - Server state management
- `@tanstack/react-router` - Routing
- `@headlessui/react` - Accessible components
- `framer-motion` - Animations
- `zod` - Schema validation
- `date-fns` - Date manipulation

### Backend
- `gin-gonic/gin` - HTTP framework
- `mattn/go-sqlite3` - SQLite driver
- `google/uuid` - UUID generation
- `golang.org/x/crypto` - Cryptography

---

## 🧪 TEST KURALLARI

- Unit test coverage minimum %80
- Integration test tüm API endpoints için
- E2E test critical user flows için
- Test isimleri açıklayıcı olmalı
- Test data factory kullan (hardcoded data yok)
- Mock kullan (external dependencies için)

### Test Örneği
```typescript
// ✅ İyi
describe('UserService', () => {
  describe('fetchUser', () => {
    it('should return user when ID is valid', async () => {
      const mockUser = createMockUser();
      mockFetch.mockResolvedValueOnce({ ok: true, json: async () => mockUser });

      const result = await fetchUser('123');

      expect(result).toEqual(mockUser);
    });

    it('should throw error when user not found', async () => {
      mockFetch.mockResolvedValueOnce({ ok: false, statusText: 'Not Found' });

      await expect(fetchUser('invalid')).rejects.toThrow('Failed to fetch user');
    });
  });
});
```

---

## 🔐 GÜVENLİK KURALLARI

- API keys environment variables'da sakla
- Passwords hash'le (bcrypt, argon2)
- HTTPS kullan (HTTP redirect)
- CORS doğru configure et
- Rate limiting implement et
- Input validation her zaman yap (client + server)
- Output encoding yap (XSS prevention)
- SQL prepared statements kullan (injection prevention)
- Authentication token'ları güvenli sakla (httpOnly cookies)

---

## 📊 PERFORMANS KURALLARI

- Lazy loading kullan (images, routes, components)
- Code splitting yap (Vite automatic)
- Debounce/throttle kullan (search, scroll)
- Memoization kullan (expensive calculations)
- Virtual scrolling kullan (long lists)
- Image optimization (WebP, lazy load, srcset)
- Bundle size tracking (max 500KB initial)
- Database index'leme (N+1 query prevention)

---

## 📝 DOKÜMANTASYON KURALLARI

- README.md her proje için zorunlu
- API documentation (OpenAPI/Swagger)
- Inline comments sadece "neden" için (kod "ne" yaptığını açıkça göstermeli)
- Change log tut (CHANGELOG.md)
- Architecture decision records (ADRs)

---

## 🔄 GIT KURALLARI

### Commit Messages
```
type(scope): subject

body

footer
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Örnek:**
```
feat(auth): add JWT authentication

Implement JWT-based authentication with refresh tokens.
Add middleware for protected routes.

Closes #123
```

### Branch Naming
- `feature/feature-name`
- `bugfix/bug-description`
- `hotfix/critical-fix`
- `refactor/what-refactoring`

---

## 🚀 DEPLOYMENT KURALLARI

- Environment variables kullan (config dosyaları yok)
- Health check endpoint implement et
- Graceful shutdown implement et
- Logging structured format (JSON)
- Metrics topla (Prometheus format)
- Error tracking (Sentry vb.)
- Database migration otomatik (app startup)

---

## 🎯 CODE REVIEW CHECKLIST

PR açmadan önce kontrol et:
- [ ] Tüm testler geçiyor
- [ ] Linter hata vermiyor
- [ ] Type check başarılı
- [ ] Security scan temiz
- [ ] Performance impact değerlendirildi
- [ ] Documentation güncellendi
- [ ] Breaking change varsa CHANGELOG'a eklendi
- [ ] Database migration varsa test edildi

---

## 📞 ÖZEL DURUMLAR

### Acil Durumlarda
1. Security vulnerability: Önce fix, sonra test
2. Production bug: Hotfix branch + immediate deploy
3. Data loss riski: Immediate backup + rollback plan

### Öğrenme/Deneme Durumlarında
- Experimental feature'lar feature flag ile implement et
- Beta users ile test et
- Rollback planı hazırla

---

**Not:** Bu kurallar "living document"tır. Proje ilerledikçe güncellenebilir.
**Version:** 1.0.0
**Last Updated:** 2025-11-11
```

---

### 3. TODO.MD TEMPLATE

**Dosya:** `.leah/todo.md` (Template)

```markdown
# Proje Todo Listesi

Bu dosya proje görevlerini ve ilerlemesini takip eder.

---

## Phase 1: Proje Kurulumu

**Durum:** ⏳ In Progress

### Görevler
- [x] Repository oluştur
- [x] Initial project setup
- [ ] Environment variables configure et
- [ ] Database schema oluştur
- [ ] CI/CD pipeline kur

**Test Kriterleri:**
- Proje local'de çalışıyor
- CI/CD pipeline başarılı
- Database migration başarılı

**Başarısızlık Durumunda:**
- Environment variables kontrolü yap
- Database connection string kontrol et
- Log'ları incele

---

## Phase 2: Authentication Sistemi

**Durum:** 📋 Pending

### Görevler
- [ ] JWT implementation
- [ ] Login/logout endpoints
- [ ] Password hashing
- [ ] Token refresh mechanism
- [ ] Protected routes middleware

**Test Kriterleri:**
- Login başarılı çalışıyor
- Token validation çalışıyor
- Refresh token working
- Protected routes erişim kontrolü yapıyor

**Başarısızlık Durumunda:**
- Token format kontrolü
- Secret key kontrolü
- Expiry time settings

---

## Phase 3: User Management

**Durum:** 📋 Pending

### Görevler
- [ ] User CRUD endpoints
- [ ] User profile page
- [ ] Avatar upload
- [ ] Email verification
- [ ] Password reset

---

## Phase 4: Core Features

**Durum:** 📋 Pending

### Görevler
- [ ] Feature A implementation
- [ ] Feature B implementation
- [ ] Feature C implementation

---

## Tamamlanan Görevler

### ✅ 2025-11-11
- [x] Initial setup
- [x] Repository created

---

## Notlar

- Her phase tamamlandıktan sonra durum güncellenir
- Test kriterleri mutlaka karşılanmalı
- Başarısızlık durumunda action plan uygulanmalı
```

---

### 4. WORKSPACE MANAGER (GO)

**Dosya:** `/home/user/ollama/workspace/manager.go` (YENİ)

```go
package workspace

import (
    "errors"
    "fmt"
    "os"
    "path/filepath"
)

var (
    ErrWorkspaceNotFound = errors.New("workspace not found")
    ErrInvalidPath       = errors.New("invalid workspace path")
)

// Manager manages workspace operations
type Manager struct {
    store Store
}

// Store interface for persistence
type Store interface {
    GetWorkspace(id string) (*Workspace, error)
    GetActiveWorkspace() (*Workspace, error)
    CreateWorkspace(ws *Workspace) error
    UpdateWorkspace(ws *Workspace) error
    DeleteWorkspace(id string) error
    SetActiveWorkspace(id string) error
}

// Workspace represents a workspace
type Workspace struct {
    ID           string `json:"id"`
    Name         string `json:"name"`
    Path         string `json:"path"`
    Active       bool   `json:"active"`
    RulesEnabled bool   `json:"rules_enabled"`
    TodoEnabled  bool   `json:"todo_enabled"`
}

// NewManager creates a new workspace manager
func NewManager(store Store) *Manager {
    return &Manager{store: store}
}

// Initialize initializes a workspace directory
func (m *Manager) Initialize(workspacePath string) error {
    // Validate path
    absPath, err := filepath.Abs(workspacePath)
    if err != nil {
        return fmt.Errorf("invalid path: %w", err)
    }

    // Check if path exists
    if _, err := os.Stat(absPath); os.IsNotExist(err) {
        return ErrInvalidPath
    }

    // Create .leah directory
    leahPath := filepath.Join(absPath, ".leah")
    if err := os.MkdirAll(leahPath, 0755); err != nil {
        return fmt.Errorf("failed to create .leah directory: %w", err)
    }

    // Create subdirectories
    dirs := []string{"templates", "history"}
    for _, dir := range dirs {
        dirPath := filepath.Join(leahPath, dir)
        if err := os.MkdirAll(dirPath, 0755); err != nil {
            return fmt.Errorf("failed to create %s directory: %w", dir, err)
        }
    }

    // Create rules.md if not exists
    rulesPath := filepath.Join(leahPath, "rules.md")
    if _, err := os.Stat(rulesPath); os.IsNotExist(err) {
        if err := m.createRulesTemplate(rulesPath); err != nil {
            return err
        }
    }

    // Create todo.md if not exists
    todoPath := filepath.Join(leahPath, "todo.md")
    if _, err := os.Stat(todoPath); os.IsNotExist(err) {
        if err := m.createTodoTemplate(todoPath); err != nil {
            return err
        }
    }

    // Create config.json
    configPath := filepath.Join(leahPath, "config.json")
    if _, err := os.Stat(configPath); os.IsNotExist(err) {
        if err := m.createConfigTemplate(configPath); err != nil {
            return err
        }
    }

    return nil
}

// GetLeahPath returns .leah directory path for workspace
func (m *Manager) GetLeahPath(workspacePath string) (string, error) {
    absPath, err := filepath.Abs(workspacePath)
    if err != nil {
        return "", err
    }

    leahPath := filepath.Join(absPath, ".leah")
    if _, err := os.Stat(leahPath); os.IsNotExist(err) {
        return "", fmt.Errorf(".leah directory not found in workspace")
    }

    return leahPath, nil
}

// GetRulesPath returns rules.md path
func (m *Manager) GetRulesPath(workspacePath string) (string, error) {
    leahPath, err := m.GetLeahPath(workspacePath)
    if err != nil {
        return "", err
    }
    return filepath.Join(leahPath, "rules.md"), nil
}

// GetTodoPath returns todo.md path
func (m *Manager) GetTodoPath(workspacePath string) (string, error) {
    leahPath, err := m.GetLeahPath(workspacePath)
    if err != nil {
        return "", err
    }
    return filepath.Join(leahPath, "todo.md"), nil
}

// createRulesTemplate creates default rules.md
func (m *Manager) createRulesTemplate(path string) error {
    template := getRulesTemplate()
    return os.WriteFile(path, []byte(template), 0644)
}

// createTodoTemplate creates default todo.md
func (m *Manager) createTodoTemplate(path string) error {
    template := getTodoTemplate()
    return os.WriteFile(path, []byte(template), 0644)
}

// createConfigTemplate creates default config.json
func (m *Manager) createConfigTemplate(path string) error {
    config := `{
  "version": "1.0.0",
  "rules": {
    "enabled": true,
    "auto_include": true
  },
  "todo": {
    "enabled": true,
    "auto_track": true
  }
}`
    return os.WriteFile(path, []byte(config), 0644)
}

// GetWorkspace returns workspace by ID
func (m *Manager) GetWorkspace(id string) (*Workspace, error) {
    return m.store.GetWorkspace(id)
}

// GetActiveWorkspace returns active workspace
func (m *Manager) GetActiveWorkspace() (*Workspace, error) {
    return m.store.GetActiveWorkspace()
}

// CreateWorkspace creates a new workspace
func (m *Manager) CreateWorkspace(name, path string) (*Workspace, error) {
    // Initialize directory
    if err := m.Initialize(path); err != nil {
        return nil, err
    }

    // Create workspace
    ws := &Workspace{
        ID:           generateID(),
        Name:         name,
        Path:         path,
        Active:       false,
        RulesEnabled: true,
        TodoEnabled:  true,
    }

    if err := m.store.CreateWorkspace(ws); err != nil {
        return nil, err
    }

    return ws, nil
}

// SetActiveWorkspace sets active workspace
func (m *Manager) SetActiveWorkspace(id string) error {
    return m.store.SetActiveWorkspace(id)
}

func generateID() string {
    // Use uuid library
    return uuid.New().String()
}
```

---

### 5. RULES MANAGER

**Dosya:** `/home/user/ollama/workspace/rules.go` (YENİ)

```go
package workspace

import (
    "bufio"
    "os"
    "strings"
)

// RulesManager manages rules.md file
type RulesManager struct {
    workspaceManager *Manager
}

// NewRulesManager creates rules manager
func NewRulesManager(wm *Manager) *RulesManager {
    return &RulesManager{workspaceManager: wm}
}

// GetRules reads and parses rules.md
func (rm *RulesManager) GetRules(workspacePath string) (*Rules, error) {
    rulesPath, err := rm.workspaceManager.GetRulesPath(workspacePath)
    if err != nil {
        return nil, err
    }

    content, err := os.ReadFile(rulesPath)
    if err != nil {
        return nil, err
    }

    return rm.parseRules(string(content)), nil
}

// Rules represents parsed rules
type Rules struct {
    Prohibitions  []string `json:"prohibitions"`
    Requirements  []string `json:"requirements"`
    CodeStyle     []string `json:"code_style"`
    Database      []string `json:"database"`
    Language      string   `json:"language"`
    Framework     string   `json:"framework"`
    Testing       []string `json:"testing"`
    Security      []string `json:"security"`
    Performance   []string `json:"performance"`
    Documentation []string `json:"documentation"`
    Git           []string `json:"git"`
    RawContent    string   `json:"raw_content"`
}

// parseRules parses rules.md content
func (rm *RulesManager) parseRules(content string) *Rules {
    rules := &Rules{
        RawContent: content,
    }

    scanner := bufio.NewScanner(strings.NewReader(content))
    var currentSection string

    for scanner.Scan() {
        line := scanner.Text()
        trimmed := strings.TrimSpace(line)

        // Detect sections
        if strings.HasPrefix(trimmed, "## 🚫 YASAKLAR") {
            currentSection = "prohibitions"
            continue
        } else if strings.HasPrefix(trimmed, "## ✅ ZORUNLU") {
            currentSection = "requirements"
            continue
        } else if strings.HasPrefix(trimmed, "## 💻 KOD YAZMA") {
            currentSection = "code_style"
            continue
        } else if strings.HasPrefix(trimmed, "## 🗄️ DATABASE") {
            currentSection = "database"
            continue
        } else if strings.HasPrefix(trimmed, "## 🧪 TEST") {
            currentSection = "testing"
            continue
        } else if strings.HasPrefix(trimmed, "## 🔐 GÜVENLİK") {
            currentSection = "security"
            continue
        } else if strings.HasPrefix(trimmed, "## 📊 PERFORMANS") {
            currentSection = "performance"
            continue
        } else if strings.HasPrefix(trimmed, "## 📝 DOKÜMANTASYON") {
            currentSection = "documentation"
            continue
        } else if strings.HasPrefix(trimmed, "## 🔄 GIT") {
            currentSection = "git"
            continue
        }

        // Parse bullet points
        if strings.HasPrefix(trimmed, "- ") {
            rule := strings.TrimPrefix(trimmed, "- ")
            switch currentSection {
            case "prohibitions":
                rules.Prohibitions = append(rules.Prohibitions, rule)
            case "requirements":
                rules.Requirements = append(rules.Requirements, rule)
            case "code_style":
                rules.CodeStyle = append(rules.CodeStyle, rule)
            case "database":
                rules.Database = append(rules.Database, rule)
            case "testing":
                rules.Testing = append(rules.Testing, rule)
            case "security":
                rules.Security = append(rules.Security, rule)
            case "performance":
                rules.Performance = append(rules.Performance, rule)
            case "documentation":
                rules.Documentation = append(rules.Documentation, rule)
            case "git":
                rules.Git = append(rules.Git, rule)
            }
        }
    }

    return rules
}

// ToSystemPrompt converts rules to system prompt
func (r *Rules) ToSystemPrompt() string {
    var parts []string

    parts = append(parts, "# WORKSPACE RULES")
    parts = append(parts, "You must strictly follow these rules defined by the user:")
    parts = append(parts, "")

    if len(r.Prohibitions) > 0 {
        parts = append(parts, "## PROHIBITIONS (NEVER DO THESE):")
        for _, rule := range r.Prohibitions {
            parts = append(parts, "- "+rule)
        }
        parts = append(parts, "")
    }

    if len(r.Requirements) > 0 {
        parts = append(parts, "## REQUIREMENTS (ALWAYS DO THESE):")
        for _, rule := range r.Requirements {
            parts = append(parts, "- "+rule)
        }
        parts = append(parts, "")
    }

    if len(r.CodeStyle) > 0 {
        parts = append(parts, "## CODE STYLE:")
        for _, rule := range r.CodeStyle {
            parts = append(parts, "- "+rule)
        }
        parts = append(parts, "")
    }

    if len(r.Security) > 0 {
        parts = append(parts, "## SECURITY:")
        for _, rule := range r.Security {
            parts = append(parts, "- "+rule)
        }
        parts = append(parts, "")
    }

    return strings.Join(parts, "\n")
}

// UpdateRules updates rules.md file
func (rm *RulesManager) UpdateRules(workspacePath string, content string) error {
    rulesPath, err := rm.workspaceManager.GetRulesPath(workspacePath)
    if err != nil {
        return err
    }

    return os.WriteFile(rulesPath, []byte(content), 0644)
}
```

---

### 6. TODO MANAGER

**Dosya:** `/home/user/ollama/workspace/todo.go` (YENİ)

```go
package workspace

import (
    "bufio"
    "fmt"
    "os"
    "strings"
    "time"
)

// TodoManager manages todo.md file
type TodoManager struct {
    workspaceManager *Manager
}

// NewTodoManager creates todo manager
func NewTodoManager(wm *Manager) *TodoManager {
    return &TodoManager{workspaceManager: wm}
}

// TodoList represents parsed todo list
type TodoList struct {
    Phases []*Phase `json:"phases"`
}

// Phase represents a phase in todo list
type Phase struct {
    Name        string       `json:"name"`
    Status      string       `json:"status"` // pending, in_progress, completed, failed
    Tasks       []*Task      `json:"tasks"`
    TestPlan    []string     `json:"test_plan"`
    OnFailure   []string     `json:"on_failure"`
}

// Task represents a task
type Task struct {
    Text      string    `json:"text"`
    Completed bool      `json:"completed"`
    Subtasks  []*Task   `json:"subtasks,omitempty"`
}

// GetTodos parses todo.md
func (tm *TodoManager) GetTodos(workspacePath string) (*TodoList, error) {
    todoPath, err := tm.workspaceManager.GetTodoPath(workspacePath)
    if err != nil {
        return nil, err
    }

    content, err := os.ReadFile(todoPath)
    if err != nil {
        return nil, err
    }

    return tm.parseTodos(string(content)), nil
}

// parseTodos parses todo.md content
func (tm *TodoManager) parseTodos(content string) *TodoList {
    todoList := &TodoList{
        Phases: make([]*Phase, 0),
    }

    scanner := bufio.NewScanner(strings.NewReader(content))
    var currentPhase *Phase
    var parsingSection string

    for scanner.Scan() {
        line := scanner.Text()
        trimmed := strings.TrimSpace(line)

        // Detect phase headers (## Phase X:)
        if strings.HasPrefix(trimmed, "## Phase ") || strings.HasPrefix(trimmed, "## ") {
            if currentPhase != nil {
                todoList.Phases = append(todoList.Phases, currentPhase)
            }

            phaseName := strings.TrimPrefix(trimmed, "## ")
            currentPhase = &Phase{
                Name:   phaseName,
                Status: "pending",
                Tasks:  make([]*Task, 0),
            }
            parsingSection = ""
            continue
        }

        if currentPhase == nil {
            continue
        }

        // Detect status
        if strings.HasPrefix(trimmed, "**Durum:**") {
            status := extractStatus(trimmed)
            currentPhase.Status = status
            continue
        }

        // Detect sections
        if strings.HasPrefix(trimmed, "### Görevler") {
            parsingSection = "tasks"
            continue
        } else if strings.HasPrefix(trimmed, "**Test Kriterleri:**") {
            parsingSection = "test_plan"
            continue
        } else if strings.HasPrefix(trimmed, "**Başarısızlık Durumunda:**") {
            parsingSection = "on_failure"
            continue
        }

        // Parse content based on section
        if strings.HasPrefix(trimmed, "- [") {
            task := parseTask(trimmed)
            if parsingSection == "tasks" && task != nil {
                currentPhase.Tasks = append(currentPhase.Tasks, task)
            } else if parsingSection == "test_plan" {
                currentPhase.TestPlan = append(currentPhase.TestPlan, task.Text)
            } else if parsingSection == "on_failure" {
                currentPhase.OnFailure = append(currentPhase.OnFailure, task.Text)
            }
        }
    }

    // Add last phase
    if currentPhase != nil {
        todoList.Phases = append(todoList.Phases, currentPhase)
    }

    return todoList
}

func extractStatus(line string) string {
    // "**Durum:** ⏳ In Progress" -> "in_progress"
    parts := strings.Split(line, "**Durum:**")
    if len(parts) < 2 {
        return "pending"
    }

    status := strings.TrimSpace(parts[1])
    if strings.Contains(status, "In Progress") || strings.Contains(status, "⏳") {
        return "in_progress"
    } else if strings.Contains(status, "Completed") || strings.Contains(status, "✅") {
        return "completed"
    } else if strings.Contains(status, "Failed") || strings.Contains(status, "❌") {
        return "failed"
    }

    return "pending"
}

func parseTask(line string) *Task {
    // "- [x] Task text" or "- [ ] Task text"
    trimmed := strings.TrimSpace(line)
    if !strings.HasPrefix(trimmed, "- [") {
        return nil
    }

    completed := strings.HasPrefix(trimmed, "- [x]") || strings.HasPrefix(trimmed, "- [X]")
    text := ""

    if completed {
        text = strings.TrimPrefix(trimmed, "- [x] ")
        text = strings.TrimPrefix(text, "- [X] ")
    } else {
        text = strings.TrimPrefix(trimmed, "- [ ] ")
    }

    return &Task{
        Text:      strings.TrimSpace(text),
        Completed: completed,
    }
}

// UpdateTodos updates todo.md file
func (tm *TodoManager) UpdateTodos(workspacePath string, content string) error {
    todoPath, err := tm.workspaceManager.GetTodoPath(workspacePath)
    if err != nil {
        return err
    }

    return os.WriteFile(todoPath, []byte(content), 0644)
}

// MarkTaskComplete marks a task as complete
func (tm *TodoManager) MarkTaskComplete(workspacePath string, phaseIndex, taskIndex int) error {
    todos, err := tm.GetTodos(workspacePath)
    if err != nil {
        return err
    }

    if phaseIndex >= len(todos.Phases) {
        return fmt.Errorf("invalid phase index")
    }

    phase := todos.Phases[phaseIndex]
    if taskIndex >= len(phase.Tasks) {
        return fmt.Errorf("invalid task index")
    }

    phase.Tasks[taskIndex].Completed = true

    // Check if all tasks completed
    allCompleted := true
    for _, task := range phase.Tasks {
        if !task.Completed {
            allCompleted = false
            break
        }
    }

    if allCompleted {
        phase.Status = "completed"
    }

    // Regenerate todo.md
    return tm.saveTodos(workspacePath, todos)
}

// saveTodos saves todo list back to file
func (tm *TodoManager) saveTodos(workspacePath string, todos *TodoList) error {
    var content strings.Builder

    content.WriteString("# Proje Todo Listesi\n\n")
    content.WriteString(fmt.Sprintf("**Son Güncelleme:** %s\n\n", time.Now().Format("2006-01-02 15:04:05")))
    content.WriteString("---\n\n")

    for _, phase := range todos.Phases {
        content.WriteString(fmt.Sprintf("## %s\n\n", phase.Name))

        statusEmoji := "📋"
        statusText := "Pending"
        switch phase.Status {
        case "in_progress":
            statusEmoji = "⏳"
            statusText = "In Progress"
        case "completed":
            statusEmoji = "✅"
            statusText = "Completed"
        case "failed":
            statusEmoji = "❌"
            statusText = "Failed"
        }

        content.WriteString(fmt.Sprintf("**Durum:** %s %s\n\n", statusEmoji, statusText))

        if len(phase.Tasks) > 0 {
            content.WriteString("### Görevler\n")
            for _, task := range phase.Tasks {
                checkbox := "[ ]"
                if task.Completed {
                    checkbox = "[x]"
                }
                content.WriteString(fmt.Sprintf("- %s %s\n", checkbox, task.Text))
            }
            content.WriteString("\n")
        }

        if len(phase.TestPlan) > 0 {
            content.WriteString("**Test Kriterleri:**\n")
            for _, test := range phase.TestPlan {
                content.WriteString(fmt.Sprintf("- %s\n", test))
            }
            content.WriteString("\n")
        }

        if len(phase.OnFailure) > 0 {
            content.WriteString("**Başarısızlık Durumunda:**\n")
            for _, action := range phase.OnFailure {
                content.WriteString(fmt.Sprintf("- %s\n", action))
            }
            content.WriteString("\n")
        }

        content.WriteString("---\n\n")
    }

    return tm.UpdateTodos(workspacePath, content.String())
}
```

---

### 7. SERVER ROUTES (Workspace Endpoints)

**Dosya:** `/home/user/ollama/server/routes.go` (GÜNCELLENECEK - eklenecek kısım)

```go
// Workspace routes
func (s *Server) RegisterWorkspaceRoutes(r *gin.Engine) {
    api := r.Group("/api/workspace")

    // Workspace management
    api.GET("", s.ListWorkspacesHandler)
    api.POST("", s.CreateWorkspaceHandler)
    api.GET("/:id", s.GetWorkspaceHandler)
    api.PUT("/:id", s.UpdateWorkspaceHandler)
    api.DELETE("/:id", s.DeleteWorkspaceHandler)
    api.POST("/:id/activate", s.ActivateWorkspaceHandler)

    // Initialize workspace
    api.POST("/initialize", s.InitializeWorkspaceHandler)

    // Rules management
    api.GET("/:id/rules", s.GetRulesHandler)
    api.PUT("/:id/rules", s.UpdateRulesHandler)

    // Todo management
    api.GET("/:id/todos", s.GetTodosHandler)
    api.PUT("/:id/todos", s.UpdateTodosHandler)
    api.POST("/:id/todos/:phase/:task/complete", s.MarkTaskCompleteHandler)
}

// GetRulesHandler returns workspace rules
func (s *Server) GetRulesHandler(c *gin.Context) {
    workspaceID := c.Param("id")

    workspace, err := s.workspaceManager.GetWorkspace(workspaceID)
    if err != nil {
        c.JSON(404, gin.H{"error": "Workspace not found"})
        return
    }

    rulesManager := workspace.NewRulesManager(s.workspaceManager)
    rules, err := rulesManager.GetRules(workspace.Path)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(200, rules)
}

// UpdateRulesHandler updates workspace rules
func (s *Server) UpdateRulesHandler(c *gin.Context) {
    workspaceID := c.Param("id")

    var req struct {
        Content string `json:"content"`
    }

    if err := c.BindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    workspace, err := s.workspaceManager.GetWorkspace(workspaceID)
    if err != nil {
        c.JSON(404, gin.H{"error": "Workspace not found"})
        return
    }

    rulesManager := workspace.NewRulesManager(s.workspaceManager)
    if err := rulesManager.UpdateRules(workspace.Path, req.Content); err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    // Save to history
    s.saveToHistory(workspaceID, "rules", req.Content, "Manual update")

    c.JSON(200, gin.H{"message": "Rules updated successfully"})
}

// GetTodosHandler returns workspace todos
func (s *Server) GetTodosHandler(c *gin.Context) {
    workspaceID := c.Param("id")

    workspace, err := s.workspaceManager.GetWorkspace(workspaceID)
    if err != nil {
        c.JSON(404, gin.H{"error": "Workspace not found"})
        return
    }

    todoManager := workspace.NewTodoManager(s.workspaceManager)
    todos, err := todoManager.GetTodos(workspace.Path)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(200, todos)
}

// MarkTaskCompleteHandler marks a task as complete
func (s *Server) MarkTaskCompleteHandler(c *gin.Context) {
    workspaceID := c.Param("id")
    phaseIndex := c.Param("phase") // "0", "1", etc.
    taskIndex := c.Param("task")   // "0", "1", etc.

    workspace, err := s.workspaceManager.GetWorkspace(workspaceID)
    if err != nil {
        c.JSON(404, gin.H{"error": "Workspace not found"})
        return
    }

    phase, _ := strconv.Atoi(phaseIndex)
    task, _ := strconv.Atoi(taskIndex)

    todoManager := workspace.NewTodoManager(s.workspaceManager)
    if err := todoManager.MarkTaskComplete(workspace.Path, phase, task); err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(200, gin.H{"message": "Task marked as complete"})
}
```

---

### 8. CHAT HANDLER INTEGRATION (Rules Injection)

**Dosya:** `/home/user/ollama/server/chat_handler.go` (GÜNCELLENECEK)

```go
func (s *Server) ChatHandler(c *gin.Context) {
    var req ChatRequest

    if err := c.BindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // Get active workspace
    workspace, err := s.workspaceManager.GetActiveWorkspace()
    if err == nil && workspace != nil && workspace.RulesEnabled {
        // Inject rules into system prompt
        rulesManager := workspace.NewRulesManager(s.workspaceManager)
        rules, err := rulesManager.GetRules(workspace.Path)
        if err == nil {
            // Prepend rules to messages
            rulesPrompt := rules.ToSystemPrompt()

            // Find system message or create one
            hasSystemMessage := false
            for i, msg := range req.Messages {
                if msg.Role == "system" {
                    // Prepend rules to existing system message
                    req.Messages[i].Content = rulesPrompt + "\n\n" + msg.Content
                    hasSystemMessage = true
                    break
                }
            }

            if !hasSystemMessage {
                // Create new system message with rules
                systemMsg := Message{
                    Role:    "system",
                    Content: rulesPrompt,
                }
                req.Messages = append([]Message{systemMsg}, req.Messages...)
            }
        }
    }

    // Continue with normal chat handling...
    s.handleChat(c, req)
}
```

---

### 9. FRONTEND - RULES HOOK

**Dosya:** `/home/user/ollama/app/ui/app/src/hooks/useRules.ts` (YENİ)

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export interface Rules {
  prohibitions: string[];
  requirements: string[];
  code_style: string[];
  database: string[];
  language: string;
  framework: string;
  testing: string[];
  security: string[];
  performance: string[];
  documentation: string[];
  git: string[];
  raw_content: string;
}

export function useRules(workspaceId: string | undefined) {
  return useQuery({
    queryKey: ['workspace', workspaceId, 'rules'],
    queryFn: async () => {
      if (!workspaceId) return null;
      const response = await fetch(`/api/workspace/${workspaceId}/rules`);
      if (!response.ok) throw new Error('Failed to fetch rules');
      return response.json() as Promise<Rules>;
    },
    enabled: !!workspaceId,
  });
}

export function useUpdateRules() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ workspaceId, content }: { workspaceId: string; content: string }) => {
      const response = await fetch(`/api/workspace/${workspaceId}/rules`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      if (!response.ok) throw new Error('Failed to update rules');
      return response.json();
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['workspace', variables.workspaceId, 'rules'] });
    },
  });
}
```

---

### 10. FRONTEND - RULES EDITOR COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/components/RulesEditor.tsx` (YENİ)

```typescript
import { useState } from 'react';
import { useRules, useUpdateRules } from '../hooks/useRules';
import { PencilIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface RulesEditorProps {
  workspaceId: string;
}

export function RulesEditor({ workspaceId }: RulesEditorProps) {
  const { data: rules, isLoading } = useRules(workspaceId);
  const updateRules = useUpdateRules();
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');

  const handleEdit = () => {
    setEditContent(rules?.raw_content || '');
    setIsEditing(true);
  };

  const handleSave = async () => {
    await updateRules.mutateAsync({ workspaceId, content: editContent });
    setIsEditing(false);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditContent('');
  };

  if (isLoading) {
    return <div className="animate-pulse">Loading rules...</div>;
  }

  if (!rules) {
    return <div>No rules found</div>;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Workspace Rules</h2>
        {!isEditing && (
          <button
            onClick={handleEdit}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          >
            <PencilIcon className="h-5 w-5" />
            Edit Rules
          </button>
        )}
      </div>

      {isEditing ? (
        <div className="space-y-4">
          <textarea
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            className="w-full h-[600px] p-4 font-mono text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
            placeholder="Write your rules in Markdown..."
          />
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={updateRules.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              <CheckIcon className="h-5 w-5" />
              {updateRules.isPending ? 'Saving...' : 'Save'}
            </button>
            <button
              onClick={handleCancel}
              disabled={updateRules.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50"
            >
              <XMarkIcon className="h-5 w-5" />
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Prohibitions */}
          {rules.prohibitions.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-red-600 mb-2">🚫 Prohibitions</h3>
              <ul className="list-disc list-inside space-y-1">
                {rules.prohibitions.map((rule, i) => (
                  <li key={i} className="text-sm">{rule}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Requirements */}
          {rules.requirements.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-green-600 mb-2">✅ Requirements</h3>
              <ul className="list-disc list-inside space-y-1">
                {rules.requirements.map((rule, i) => (
                  <li key={i} className="text-sm">{rule}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Code Style */}
          {rules.code_style.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-blue-600 mb-2">💻 Code Style</h3>
              <ul className="list-disc list-inside space-y-1">
                {rules.code_style.map((rule, i) => (
                  <li key={i} className="text-sm">{rule}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Security */}
          {rules.security.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-yellow-600 mb-2">🔐 Security</h3>
              <ul className="list-disc list-inside space-y-1">
                {rules.security.map((rule, i) => (
                  <li key={i} className="text-sm">{rule}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

---

## 📊 PERFORMANS KRİTERLERİ

- **Rules Parsing:** < 20ms
- **Todo Parsing:** < 20ms
- **Rules Injection:** < 5ms
- **File Read:** < 10ms (cached)
- **UI Update:** < 100ms

---

## ✅ BAŞARI KRİTERLERİ

1. ✅ Kullanıcı workspace oluşturabiliyor
2. ✅ `.leah` klasörü otomatik oluşuyor
3. ✅ `rules.md` ve `todo.md` template'leri oluşuyor
4. ✅ GUI'den kurallar düzenlenebiliyor
5. ✅ Kurallar chat request'lerine otomatik ekleniyor
6. ✅ Todo list UI'dan yönetilebiliyor
7. ✅ Görevler işaretlenebiliyor ve takip ediliyor

---

## 🔄 SONRAKİ PHASE

**Phase 3:** UI/UX İyileştirmeleri ve Yeni Layout Sistemi
