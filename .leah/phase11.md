# PHASE 11: ADVANCED FEATURES (Voice, Image, Web, Code Execution)

## 📋 HEDEFLER
1. ✅ Voice input (Whisper API)
2. ✅ Voice output (TTS API)
3. ✅ Image generation (DALL-E/Stable Diffusion)
4. ✅ Web scraping (URL'den veri çekme)
5. ✅ Code execution (sandboxed Python/JS)
6. ✅ Chat export/import (JSON/Markdown)

## 🏗️ MİMARİ

### Features Matrix
| Feature | Provider | API Endpoint |
|---------|----------|--------------|
| Voice Input | OpenAI Whisper | /v1/audio/transcriptions |
| Voice Output | OpenAI TTS | /v1/audio/speech |
| Image Gen | OpenAI DALL-E | /v1/images/generations |
| Web Scraping | Custom | Internal |
| Code Exec | Docker | Internal |

## 📁 DOSYALAR

### 1. Voice Input Handler
**Dosya:** `/home/user/ollama/features/voice.go` (YENİ)

```go
package features

type VoiceHandler struct {
    openai *providers.OpenAIProvider
}

func (vh *VoiceHandler) Transcribe(audioFile io.Reader) (string, error) {
    // Call OpenAI Whisper API
    url := "https://api.openai.com/v1/audio/transcriptions"

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)

    part, err := writer.CreateFormFile("file", "audio.mp3")
    if err != nil {
        return "", err
    }

    if _, err := io.Copy(part, audioFile); err != nil {
        return "", err
    }

    writer.WriteField("model", "whisper-1")
    writer.Close()

    req, err := http.NewRequest("POST", url, body)
    if err != nil {
        return "", err
    }

    req.Header.Set("Authorization", "Bearer "+vh.openai.APIKey)
    req.Header.Set("Content-Type", writer.FormDataContentType())

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    var result struct {
        Text string `json:"text"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }

    return result.Text, nil
}

func (vh *VoiceHandler) Synthesize(text string, voice string) ([]byte, error) {
    // Call OpenAI TTS API
    url := "https://api.openai.com/v1/audio/speech"

    payload := map[string]string{
        "model": "tts-1",
        "input": text,
        "voice": voice, // alloy, echo, fable, onyx, nova, shimmer
    }

    body, _ := json.Marshal(payload)

    req, err := http.NewRequest("POST", url, bytes.NewReader(body))
    if err != nil {
        return nil, err
    }

    req.Header.Set("Authorization", "Bearer "+vh.openai.APIKey)
    req.Header.Set("Content-Type", "application/json")

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}
```

### 2. Image Generation
**Dosya:** `/home/user/ollama/features/image.go` (YENİ)

```go
func (ih *ImageHandler) Generate(prompt string, size string) (string, error) {
    url := "https://api.openai.com/v1/images/generations"

    payload := map[string]interface{}{
        "model":  "dall-e-3",
        "prompt": prompt,
        "size":   size, // "1024x1024", "1792x1024", "1024x1792"
        "n":      1,
    }

    body, _ := json.Marshal(payload)

    req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
    req.Header.Set("Authorization", "Bearer "+ih.apiKey)
    req.Header.Set("Content-Type", "application/json")

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    var result struct {
        Data []struct {
            URL string `json:"url"`
        } `json:"data"`
    }

    json.NewDecoder(resp.Body).Decode(&result)

    if len(result.Data) > 0 {
        return result.Data[0].URL, nil
    }

    return "", errors.New("no image generated")
}
```

### 3. Web Scraper
**Dosya:** `/home/user/ollama/features/webscraper.go` (YENİ)

```go
package features

import (
    "github.com/PuerkitoBio/goquery"
)

type WebScraper struct{}

func (ws *WebScraper) Scrape(url string) (*ScrapedContent, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    doc, err := goquery.NewDocumentFromReader(resp.Body)
    if err != nil {
        return nil, err
    }

    content := &ScrapedContent{
        URL:   url,
        Title: doc.Find("title").Text(),
    }

    // Extract main content
    doc.Find("p, h1, h2, h3, li").Each(func(i int, s *goquery.Selection) {
        text := strings.TrimSpace(s.Text())
        if text != "" {
            content.Text += text + "\n\n"
        }
    })

    // Extract links
    doc.Find("a[href]").Each(func(i int, s *goquery.Selection) {
        href, _ := s.Attr("href")
        content.Links = append(content.Links, href)
    })

    return content, nil
}

type ScrapedContent struct {
    URL   string   `json:"url"`
    Title string   `json:"title"`
    Text  string   `json:"text"`
    Links []string `json:"links"`
}
```

### 4. Code Executor (Sandboxed)
**Dosya:** `/home/user/ollama/features/codeexec.go` (YENİ)

```go
package features

import (
    "context"
    "time"
    "github.com/docker/docker/api/types/container"
    "github.com/docker/docker/client"
)

type CodeExecutor struct {
    docker *client.Client
}

func (ce *CodeExecutor) ExecutePython(code string, timeout time.Duration) (*ExecutionResult, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    // Create container
    resp, err := ce.docker.ContainerCreate(ctx, &container.Config{
        Image: "python:3.11-slim",
        Cmd:   []string{"python", "-c", code},
    }, nil, nil, nil, "")
    if err != nil {
        return nil, err
    }

    defer ce.docker.ContainerRemove(ctx, resp.ID, types.ContainerRemoveOptions{Force: true})

    // Start container
    if err := ce.docker.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
        return nil, err
    }

    // Wait for completion
    statusCh, errCh := ce.docker.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)

    select {
    case err := <-errCh:
        return nil, err
    case <-statusCh:
    }

    // Get logs
    out, err := ce.docker.ContainerLogs(ctx, resp.ID, types.ContainerLogsOptions{
        ShowStdout: true,
        ShowStderr: true,
    })
    if err != nil {
        return nil, err
    }

    defer out.Close()
    logs, _ := io.ReadAll(out)

    return &ExecutionResult{
        Output: string(logs),
        ExitCode: 0,
    }, nil
}

type ExecutionResult struct {
    Output   string `json:"output"`
    Error    string `json:"error,omitempty"`
    ExitCode int    `json:"exit_code"`
}
```

### 5. Chat Export/Import
**Dosya:** `/home/user/ollama/features/export.go` (YENİ)

```go
func (e *Exporter) ExportChat(chatID string, format string) ([]byte, error) {
    chat, err := e.store.GetChat(chatID)
    if err != nil {
        return nil, err
    }

    messages, err := e.store.GetMessages(chatID)
    if err != nil {
        return nil, err
    }

    switch format {
    case "json":
        return e.exportJSON(chat, messages)
    case "markdown":
        return e.exportMarkdown(chat, messages)
    default:
        return nil, errors.New("unsupported format")
    }
}

func (e *Exporter) exportMarkdown(chat *Chat, messages []Message) ([]byte, error) {
    var md strings.Builder

    md.WriteString(fmt.Sprintf("# %s\n\n", chat.Title))
    md.WriteString(fmt.Sprintf("**Created:** %s\n\n", chat.CreatedAt.Format("2006-01-02 15:04:05")))
    md.WriteString("---\n\n")

    for _, msg := range messages {
        md.WriteString(fmt.Sprintf("## %s\n\n", strings.ToUpper(msg.Role)))
        md.WriteString(msg.Content + "\n\n")

        if msg.Thinking != "" {
            md.WriteString("**Thinking:**\n\n")
            md.WriteString(msg.Thinking + "\n\n")
        }

        md.WriteString("---\n\n")
    }

    return []byte(md.String()), nil
}
```

## ✅ BAŞARI KRİTERLERİ
1. ✅ Voice input çalışıyor
2. ✅ Voice output çalışıyor
3. ✅ Image generation çalışıyor
4. ✅ Web scraping çalışıyor
5. ✅ Code execution güvenli ve çalışıyor
6. ✅ Export/import çalışıyor

**SONRAKİ:** Phase 12 - Plugin System
