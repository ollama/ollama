# Projetos e menções de arquivos com `@`

Este documento descreve duas funcionalidades adicionadas ao app desktop:

1. **Projeto ativo** — abrir uma pasta local como projeto (estilo "Open Folder" do VSCode), com árvore de arquivos lateral, visualizador de arquivos, persistência do último projeto, recentes e carregamento automático de `AGENTS.md` e `.agents/skills/`.
2. **Menção de arquivos com `@`** — autocomplete fuzzy de arquivos do projeto no campo de chat, com injeção do conteúdo dos arquivos mencionados no contexto da LLM.

O modo "chat livre" (sem projeto aberto) continua funcionando normalmente: sem árvore, sem menções, sem contexto de pasta.

---

## Visão geral da arquitetura

```
┌────────────────────────── Frontend (React/Vite) ──────────────────────────┐
│                                                                           │
│  ProjectButton (pill no header)      ProjectPanel (árvore lateral)        │
│        │  abrir/recentes/fechar            │  clique abre FileViewer     │
│        ▼                                   ▼  botão @ menciona          │
│  useProject / useProjectFiles  ◄── cache react-query (staleTime: ∞)      │
│        │                                                                  │
│  ChatForm ── detecta "@" ── FileMentionMenu (fuzzy + teclado)            │
│        │                                                                  │
│        └── submit envia ChatRequest { prompt, file_refs: [...] }         │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ HTTP (mesmos endpoints /api/v1)
┌───────────────────────────────────▼───────────────────────────────────────┐
│                            Backend (Go, app/ui)                           │
│                                                                           │
│  project.go: estado do projeto ativo (Server.project)                     │
│    • scanner com .gitignore + pastas pesadas ignoradas                    │
│    • AGENTS.md + .agents/skills → system prompt                           │
│    • resolveFileRefs: file_refs → attachments (com truncamento)           │
│    • getProjectFile: conteúdo de um arquivo para o visualizador           │
│                                                                           │
│  store (SQLite): project_dir + recent_projects (schema v17)               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Funcionalidade 1: Projeto ativo

### Fluxo de abertura

1. O usuário clica no pill **"Open project"** no header (`ProjectButton.tsx`) e escolhe "Open folder…".
2. O frontend chama `window.webview.selectWorkingDirectory()` — binding nativo **já existente** em `app/cmd/app/webview.go` que abre o diálogo de seleção de pasta do sistema operacional.
3. Com o path retornado, o frontend faz `POST /api/v1/project/open {"path": "..."}`.
4. O backend valida o diretório, escaneia os arquivos, carrega `AGENTS.md`/skills, persiste o path no SQLite e adiciona aos recentes.
5. A UI passa a exibir a árvore de arquivos e o nome da pasta no pill do header.

### Endpoints (registrados em `app/ui/ui.go`)

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/v1/project` | Projeto ativo (ou `root: ""`) + lista de recentes. Restaura o último projeto do store na primeira chamada (reabertura automática). |
| `POST` | `/api/v1/project/open` | Body `{"path": "/abs/path"}`. Ativa o projeto, escaneia, persiste e retorna `ProjectResponse`. |
| `POST` | `/api/v1/project/close` | Fecha o projeto ativo (limpa `project_dir`; os recentes são mantidos). |
| `GET` | `/api/v1/project/files` | Listagem cacheada de arquivos. Com `?refresh=1`, re-escaneia o disco. |
| `GET` | `/api/v1/project/file` | Conteúdo de um arquivo (`?path=rel/ativo.ts`) para o visualizador. |

Formas de resposta (em `app/ui/responses/types.go`, espelhadas em `gotypes.gen.ts`):

```jsonc
// ProjectResponse
{ "root": "/Users/x/proj", "name": "proj", "hasAgentsMd": true,
  "skills": [{ "name": "deploy", "description": "..." }],
  "recent": ["/Users/x/proj", "/Users/x/outro"] }

// ProjectFilesResponse
{ "files": [{ "path": "src/main.ts", "size": 1234, "isDir": false }, ...],
  "truncated": false }

// ProjectFileResponse
{ "path": "src/main.ts", "size": 1234, "content": "export {}\n",
  "truncated": false, "binary": false, "mimeType": "" }
```

### Scanner de arquivos (`app/ui/project.go`)

- `filepath.WalkDir` a partir da raiz, produzindo uma lista **flat** de arquivos e diretórios (paths relativos com `/`); a árvore é montada no frontend.
- **`.gitignore`**: suporte a arquivos `.gitignore` na raiz e aninhados, com comentários, negação (`!`), padrões ancorados (`/foo`, `a/b`), sufixo de diretório (`dir/`), globs por segmento (`*.log`) e `**`. Implementação própria e enxuta (`parseGitignore`, `ignoreRule.matches`, `matchSegs`) — a "última regra que casa" vence, como no git.
- **Pastas sempre ignoradas** (independente do `.gitignore`): `.git`, `node_modules`, `build`, `dist`, `out`, `target`, `vendor`, `.next`, `.nuxt`, `.venv`, `venv`, `__pycache__`, `.cache`, `coverage`, `DerivedData`, `.gradle` (ver `defaultIgnoredDirs`).
- Também ignora `.DS_Store`, symlinks e arquivos não-regulares.
- **Limite**: `maxProjectFiles = 20000` entradas; ao estourar, a resposta marca `truncated: true` e a UI avisa.
- Limitação conhecida: uma regra de negação (`!`) não "ressuscita" arquivos dentro de um diretório que já foi pulado pelo walk — mesmo comportamento de implementações simplificadas.

### `AGENTS.md` e `.agents/skills/`

Carregados na abertura do projeto e injetados como **system message** em toda requisição de chat enquanto o projeto estiver aberto (ver `buildChatRequest` em `app/ui/ui.go`):

- `AGENTS.md` da raiz: conteúdo completo, limitado a `maxAgentsFileBytes = 32KB`.
- `.agents/skills/`: aceita os layouts `skills/<nome>/SKILL.md` e `skills/<nome>.md`. O frontmatter YAML (`name:`, `description:`) é lido de forma simples (`parseFrontmatter`); apenas nome + descrição entram no system prompt (não o corpo, para não estourar contexto).

O system message resultante contém: nome/path do projeto, conteúdo do AGENTS.md e a lista de skills. Ele **não é persistido** no histórico — é montado a cada request, então fechar o projeto remove o contexto imediatamente.

### Persistência (SQLite, `app/store`)

- Migração de schema **v16 → v17** (`migrateV16ToV17` em `database.go`), adicionando à tabela `settings`:
  - `project_dir TEXT` — último projeto aberto (vazio = nenhum);
  - `recent_projects TEXT` — array JSON, mais recente primeiro, máx. 8 (`maxRecentProjects`).
- Acesso via métodos dedicados do store: `ProjectDir()/SetProjectDir()` e `RecentProjects()/SetRecentProjects()`.
- **Decisão importante**: esses campos ficam *fora* do objeto `Settings` de propósito. O `POST /api/v1/settings` reescreve o objeto inteiro vindo do frontend; se o projeto morasse em `Settings`, qualquer salvamento de configuração antiga apagaria o projeto ativo.

### UI

- **`ProjectButton.tsx`** — pill no header (área de drag da janela; usa `stopPropagation` no `mousedown` para não iniciar drag). Mostra o nome da pasta ativa; menu com "Open folder…", recentes e "Close project". Tooltip mostra o path completo.
- **`ProjectPanel.tsx`** — coluna lateral (renderizada pelo `SidebarLayout` quando há projeto): árvore colapsável (pastas primeiro, ordem alfabética), botão de refresh (re-scan) e fechar. Clicar num arquivo **abre o visualizador** (`FileViewer`); mencionar no chat é uma ação explícita (botão `@` que aparece no hover da linha, ou clique com ⌘/Ctrl), que dispara o evento `project:mention-file` ouvido pelo `ChatForm`.
- **`layout.tsx`** — nota para Windows: o header superior era `xl:hidden`; agora fica sempre visível para abrigar o pill do projeto.

### Visualizador de arquivos (`FileViewer.tsx`)

Clicar num arquivo da árvore abre um **modal de preview** sobre a UI (fecha no `Esc`, no clique fora ou no `X`).

- **Backend** (`getProjectFile` em `project.go`): valida o path com o mesmo `resolvePath` usado pelas menções (rejeita path absoluto, `..` e qualquer coisa fora da raiz) e responde conforme o tipo do arquivo:
  - **texto**: conteúdo cru, limitado a `maxViewFileBytes` (512 KB); acima disso vem `truncated: true` (o corte é aparado para não deixar rune UTF-8 pela metade);
  - **imagem** (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.svg`, `.ico`): base64 em `content` + `mimeType`, até `maxViewImageBytes` (8 MB);
  - **binário** (byte `NUL` ou UTF-8 inválido): `binary: true` e `content` vazio — nada é enviado pela rede à toa.
- **Frontend**: `useProjectFileContent(path)` (react-query, `staleTime: 0` para não mostrar preview velho de um arquivo que mudou no disco). O modal mostra nome, diretório, tamanho e aviso de truncamento; imagens são renderizadas inline; texto ganha numeração de linhas e destaque de sintaxe reaproveitando o `highlighter` (shiki) já carregado pelo chat — acima de 100 KB o destaque é desligado para não travar a UI, e extensões fora da lista de linguagens carregadas caem em texto puro.
- No topo do modal: botão **Mention** (insere `@path` no chat e fecha o preview) e botão de copiar o conteúdo.

---

## Funcionalidade 2: Menção de arquivos com `@`

Ativa **somente** com projeto aberto e listagem carregada.

### Detecção e autocomplete

- Ao digitar, `detectMention(value, caret)` (em `ChatForm.tsx`) procura o último `@` antes do caret que inicia um token (início do texto ou após espaço/parêntese/aspas) e sem espaço até o caret — isso vira a query.
- A busca roda sobre a listagem **em memória** (cache react-query com `staleTime: Infinity` — nenhum acesso a disco por tecla). Ranking em `utils/fuzzyMatch.ts`:
  1. substring no nome do arquivo (quanto mais cedo, melhor);
  2. substring no path completo;
  3. subsequência no path (penalizada pela dispersão).
  Empates favorecem paths mais curtos. Até 50 resultados (dropdown com scroll).
- **Teclado**: `↑`/`↓` navegam, `Enter`/`Tab` selecionam, `Esc` fecha o menu (com `stopPropagation` para não acionar o Esc global de cancelar streaming/edição). Clique também seleciona (`FileMentionMenu.tsx`).
- Ao selecionar, o texto vira `@src/main.ts ` (com espaço) e o caret vai para depois da menção.

### Destaque visual

Um overlay (`div` posicionado atrás do `<textarea>`, mesmas métricas de fonte/line-height/wrap) renderiza o texto com `text-transparent` e pinta um **fundo azul arredondado** sob cada `@path` válido. O scroll é sincronizado via `onScroll`. Essa técnica evita trocar o textarea por um contenteditable (caret, seleção e IME continuam nativos).

`getMentionRanges()` é a fonte única de verdade: encontra todo `@token` do texto que existe na listagem (`filePathSet`) e também menções selecionadas explicitamente cujo path contém espaço. Os mesmos ranges alimentam o destaque, o cálculo do orçamento e o `file_refs` do submit.

### Injeção no prompt (backend)

- O submit envia `file_refs: ["src/main.ts", ...]` no `ChatRequest` (novo campo em `responses/types.go`).
- No handler `chat` (`app/ui/ui.go`), com projeto ativo, `resolveFileRefs()`:
  - normaliza e valida cada path (**rejeita** path absoluto, `..` e qualquer coisa fora da raiz do projeto);
  - pula refs cujo filename já está nos attachments da mensagem (evita duplicar em edição de mensagem);
  - lê o conteúdo e o anexa como **attachment** da mensagem do usuário.
- Reusar o pipeline de attachments dá de graça: injeção no formato `--- File: path --- ... --- End of path ---` já existente, persistência no SQLite, chips na UI da mensagem e suporte a edição/reenvio. Imagens mencionadas (`.png`, `.jpg`, etc.) seguem o fluxo de imagem normal.

### Limites de contexto e truncamento

Constantes em `app/ui/project.go` (backend) — o frontend espelha o total em `MENTION_TOTAL_BYTES` (`useProject.ts`):

| Constante | Valor | Efeito |
|---|---|---|
| `maxMentionFileBytes` | 64 KB | Máximo injetado por arquivo; o excedente vira `... [truncated]`. |
| `maxMentionTotalBytes` | 128 KB (~32k tokens a ~4 bytes/token) | Orçamento total por mensagem; refs além do orçamento são truncadas/ignoradas. |

No frontend, a soma dos tamanhos dos arquivos mencionados (os `size` já vêm na listagem) é comparada ao orçamento; ao estourar, aparece um **aviso amarelo** sob o input ("Mentioned files exceed the context budget…"). O envio não é bloqueado — o backend trunca.

> Para alterar o orçamento, mude `maxMentionTotalBytes` (Go) **e** `MENTION_TOTAL_BYTES` (TS) juntos.

---

## Mapa de arquivos

### Novos

| Arquivo | Papel |
|---|---|
| `app/ui/project.go` | Estado do projeto, scanner + gitignore, skills/AGENTS.md, resolução de `file_refs`, handlers HTTP |
| `app/ui/project_test.go` | Testes: scanner/gitignore, resolveFileRefs, frontmatter, system prompt, endpoints + persistência |
| `app/ui/app/src/hooks/useProject.ts` | Hooks `useProject`/`useProjectFiles`/`useProjectFileContent` (react-query) + `MENTION_TOTAL_BYTES` |
| `app/ui/app/src/components/ProjectButton.tsx` | Pill do header com menu abrir/recentes/fechar |
| `app/ui/app/src/components/ProjectPanel.tsx` | Árvore de arquivos lateral (clique abre o preview, botão `@` menciona) |
| `app/ui/app/src/components/FileMentionMenu.tsx` | Dropdown de autocomplete do `@` |
| `app/ui/app/src/components/FileViewer.tsx` | Modal de preview do arquivo (texto com destaque, imagem, binário) |
| `app/ui/app/src/utils/fuzzyMatch.ts` | Ranking fuzzy (`fuzzyScore`/`fuzzyFilter`) |

### Modificados

| Arquivo | Mudança |
|---|---|
| `app/store/database.go` | Schema v17 (`project_dir`, `recent_projects`), migração e accessors |
| `app/store/store.go` | `ProjectDir`, `SetProjectDir`, `RecentProjects`, `SetRecentProjects` |
| `app/ui/ui.go` | Campos de projeto no `Server`, rotas `/api/v1/project*`, `file_refs` → attachments no `chat`, system prompt no `buildChatRequest` |
| `app/ui/responses/types.go` | `ChatRequest.FileRefs` + `ProjectFile`/`ProjectSkill`/`ProjectResponse`/`ProjectFilesResponse` |
| `app/ui/app/codegen/gotypes.gen.ts` | Regenerado (idêntico à saída do `tscriptify`) |
| `app/ui/app/src/api.ts` | `getProject`/`openProject`/`closeProject`/`getProjectFiles`/`getProjectFile`; `sendMessage` aceita `fileRefs` |
| `app/ui/app/src/hooks/useChats.ts` | Propaga `fileRefs` na mutation de envio |
| `app/ui/app/src/components/Chat.tsx` | Propaga `fileRefs` do form para a mutation |
| `app/ui/app/src/components/ChatForm.tsx` | Detecção do `@`, menu, teclado, destaque, aviso de orçamento, evento da árvore |
| `app/ui/app/src/components/layout/layout.tsx` | Renderiza `ProjectButton` no header e `ProjectPanel` como coluna; header não é mais `xl:hidden` no Windows |

---

## Como testar

```bash
# testes
go test ./app/ui/ ./app/store/
cd app/ui/app && npx vitest run

# rodar o app (o dist do frontend precisa existir)
cd app/ui/app && npm run build
cd ../../.. && go run ./app/cmd/app

# modo dev com hot-reload da UI
cd app/ui/app && npm run dev            # terminal 1
OLLAMA_DEBUG=1 go run ./app/cmd/app -dev  # terminal 2, a partir de app/
```

Roteiro manual rápido:

1. Abrir o pill "Open project" → escolher uma pasta com `.gitignore`, `AGENTS.md` e `.agents/skills/` → conferir árvore (sem `node_modules`/ignorados) e nome no header.
2. Clicar num arquivo da árvore → o preview abre com destaque de sintaxe; clicar numa imagem → ela aparece inline; `Esc` fecha.
3. Passar o mouse numa linha da árvore → botão `@` insere a menção no chat (⌘/Ctrl+clique faz o mesmo).
4. Digitar `@` no chat → filtrar, navegar com setas, selecionar com Enter → menção destacada em azul.
5. Enviar mensagem mencionando um arquivo → a resposta do modelo deve refletir o conteúdo do arquivo; a mensagem mostra o chip do anexo.
6. Fechar e reabrir o app → o projeto reabre sozinho; "Close project" volta ao chat livre e mantém os recentes.

## Limitações conhecidas

- O matcher de `.gitignore` é um subconjunto prático do formato do git (sem escapes exóticos; negação não recupera conteúdo de diretórios já pulados).
- Arquivos com espaço no nome só são reconhecidos como menção quando inseridos pelo dropdown/árvore (a digitação manual usa token sem espaços).
- A listagem só atualiza no refresh manual ou na reabertura do projeto (sem file watcher). O preview, esse sim, relê o arquivo do disco a cada abertura.
- O preview é somente leitura: não há edição nem busca dentro do arquivo.
- O título nativo da janela não muda; o indicador do projeto é o pill no header da UI.
