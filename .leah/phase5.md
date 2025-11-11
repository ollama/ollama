# PHASE 5: PROMPT TEMPLATES VE LIBRARY

## 📋 HEDEFLER
1. ✅ Prompt template sistemi
2. ✅ Template kategorileri
3. ✅ Değişken desteği {{var}}
4. ✅ Community templates
5. ✅ Template import/export
6. ✅ One-click apply

## 🏗️ MİMARİ

### Template Schema
```sql
CREATE TABLE prompt_templates (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT,
  description TEXT,
  template_text TEXT NOT NULL,
  variables TEXT,  -- JSON array
  is_public BOOLEAN DEFAULT 0,
  author TEXT,
  rating REAL DEFAULT 0,
  usage_count INTEGER DEFAULT 0,
  created_at TIMESTAMP
);
```

### Template Format
```markdown
---
name: Code Review Expert
category: development
variables: [language, code]
---

You are an expert code reviewer specializing in {{language}}.

Review the following code:
```{{language}}
{{code}}
```

Provide:
1. Security issues
2. Performance problems
3. Best practices violations
4. Refactoring suggestions
```

## 📁 DOSYALAR

### 1. Template Manager
**Dosya:** `/home/user/ollama/templates/manager.go` (YENİ)

```go
type TemplateManager struct {
    store TemplateStore
}

type Template struct {
    ID           string                 `json:"id"`
    Name         string                 `json:"name"`
    Category     string                 `json:"category"`
    Description  string                 `json:"description"`
    Template     string                 `json:"template"`
    Variables    []string               `json:"variables"`
    IsPublic     bool                   `json:"is_public"`
}

func (tm *TemplateManager) Render(template *Template, vars map[string]string) (string, error) {
    result := template.Template

    for key, value := range vars {
        placeholder := "{{" + key + "}}"
        result = strings.ReplaceAll(result, placeholder, value)
    }

    return result, nil
}
```

### 2. Template Library Component
**Dosya:** `/home/user/ollama/app/ui/app/src/components/TemplateLibrary.tsx` (YENİ)

```typescript
export function TemplateLibrary() {
  const { data: templates } = useTemplates();
  const [category, setCategory] = useState<string>('all');

  const categories = [
    'all', 'development', 'writing', 'analysis', 'creative', 'business'
  ];

  const filteredTemplates = templates?.filter(
    t => category === 'all' || t.category === category
  );

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setCategory(cat)}
            className={`px-4 py-2 rounded-md ${
              category === cat
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates?.map(template => (
          <TemplateCard key={template.id} template={template} />
        ))}
      </div>
    </div>
  );
}
```

### Built-in Templates
**Dosya:** `/home/user/ollama/templates/builtin/` (YENİ)

```
builtin/
├── code-review.md
├── bug-fix.md
├── documentation.md
├── refactoring.md
├── test-generation.md
├── api-design.md
├── sql-optimization.md
├── security-audit.md
├── blog-post.md
└── email-professional.md
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ 50+ built-in templates
2. ✅ Template kategorileri çalışıyor
3. ✅ Değişken replacement çalışıyor
4. ✅ One-click apply çalışıyor

**SONRAKİ:** Phase 6 - RAG System
