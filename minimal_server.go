package main

import (
        "encoding/json"
        "fmt"
        "log"
        "net/http"
        "strings"
)

// ModelInfo struct for the minimal test server
type ModelInfo struct {
        Name          string   `json:"name"`
        Description   string   `json:"description"`
        Tags          []string `json:"tags"`
        Parameters    int64    `json:"parameters"`
        Size          int64    `json:"size"`
        Quantization  string   `json:"quantization,omitempty"`
        SHA256        string   `json:"sha256"`
        DownloadURL   string   `json:"download_url"`
        ModelFormat   string   `json:"model_format"`
        Category      string   `json:"category"`
        Priority      int      `json:"priority"`
        KCRecommended bool     `json:"kc_recommended"`
}

// ModelInfoWithStatus extends the ModelInfo with download status
type ModelInfoWithStatus struct {
        ModelInfo
        IsDownloaded     bool `json:"is_downloaded"`
        DownloadProgress int  `json:"download_progress,omitempty"`
}

// Sample model data for testing
var modelsData = []ModelInfo{
        {
                Name:          "mistral-7b",
                Description:   "Mistral 7B is a powerful general-purpose language model",
                Size:          4500000000,
                Parameters:    7000000000,
                DownloadURL:   "https://ollama.com/mistral",
                Tags:          []string{"General", "Text", "Efficient"},
                KCRecommended: true,
                Category:      "recommended",
                Quantization:  "Q4_K_M",
        },
        {
                Name:          "llava",
                Description:   "LLaVA (Large Language and Vision Assistant) is a multimodal model",
                Size:          4200000000,
                Parameters:    7000000000,
                DownloadURL:   "https://ollama.com/llava",
                Tags:          []string{"Vision", "Multimodal", "Images"},
                KCRecommended: true,
                Category:      "recommended",
                Quantization:  "Q4_K_M",
        },
        {
                Name:          "moondream",
                Description:   "Moondream is optimized for vision tasks",
                Size:          1200000000,
                Parameters:    1500000000,
                DownloadURL:   "https://ollama.com/moondream",
                Tags:          []string{"Vision", "Efficient", "Lightweight"},
                KCRecommended: true,
                Category:      "recommended",
                Quantization:  "Q4_0",
        },
}

// GetAllModels returns all available models with status
func GetAllModels() []ModelInfoWithStatus {
        var result []ModelInfoWithStatus
        for _, model := range modelsData {
                result = append(result, ModelInfoWithStatus{
                        ModelInfo:    model,
                        IsDownloaded: false, // In a real implementation, this would be checked
                })
        }
        return result
}

// API Response structure
type APIResponse struct {
        Success     bool                 `json:"success"`
        Error       string               `json:"error,omitempty"`
        Models      []ModelInfoWithStatus `json:"models,omitempty"`
        Recommended []ModelInfoWithStatus `json:"recommended,omitempty"`
}

func main() {
        mux := http.NewServeMux()

        // Models API endpoint
        mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                
                allModels := GetAllModels()
                var recommendedModels []ModelInfoWithStatus
                
                // Filter to get recommended models
                for _, model := range allModels {
                        if model.KCRecommended {
                                recommendedModels = append(recommendedModels, model)
                        }
                }
                
                response := APIResponse{
                        Success:     true,
                        Models:      allModels,
                        Recommended: recommendedModels,
                }
                
                json.NewEncoder(w).Encode(response)
        })

        // Download API endpoint
        mux.HandleFunc("/api/download/", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                
                // Extract model name from URL path
                modelName := strings.TrimPrefix(r.URL.Path, "/api/download/")
                
                // Return success response
                json.NewEncoder(w).Encode(map[string]interface{}{
                        "success": true,
                        "message": fmt.Sprintf("Download of %s started", modelName),
                })
        })
        
        // Remove API endpoint
        mux.HandleFunc("/api/remove/", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                
                // Extract model name from URL path
                modelName := strings.TrimPrefix(r.URL.Path, "/api/remove/")
                
                // Return success response
                json.NewEncoder(w).Encode(map[string]interface{}{
                        "success": true,
                        "message": fmt.Sprintf("Model %s removed successfully", modelName),
                })
        })

        // Status API endpoint
        mux.HandleFunc("/api/status/", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                
                // Extract model name from URL path
                modelName := strings.TrimPrefix(r.URL.Path, "/api/status/")
                
                // Mock status (using model name in the response for demonstration)
                status := map[string]interface{}{
                        "status":     "downloading",
                        "progress":   50.0,
                        "model_name": modelName,
                }
                
                json.NewEncoder(w).Encode(status)
        })

        // Cancel download endpoint
        mux.HandleFunc("/api/cancel/", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                
                // Extract model name from URL path
                modelName := strings.TrimPrefix(r.URL.Path, "/api/cancel/")
                
                // Return success response
                json.NewEncoder(w).Encode(map[string]interface{}{
                        "success": true,
                        "message": fmt.Sprintf("Download of %s cancelled", modelName),
                })
        })

        // Healthcheck endpoint
        mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(map[string]string{
                        "status":  "ok",
                        "service": "kc-riff-minimal",
                        "version": "0.1.0",
                })
        })

        // Start the server
        addr := "0.0.0.0:5000"
        fmt.Printf("Starting minimal server on http://%s\n", addr)
        err := http.ListenAndServe(addr, mux)
        if err != nil {
                log.Fatalf("Server error: %v", err)
        }
}