package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd/tui"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
)

// HardwareProfile stores total system RAM and GPU VRAM
type HardwareProfile struct {
	TotalRAM  uint64
	TotalVRAM uint64
}

type ModelSuggestion struct {
	Name        string
	Description string
	HFRepo      string // e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF"
}

type DynamicSuggestion struct {
	ModelSuggestion
	BestTag string
	Size    uint64
	InVRAM  bool
}

type HFTreeNode struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size uint64 `json:"size"`
}

// Baseline Models
var categoryCodigo = []ModelSuggestion{
	{Name: "Qwen 2.5 Coder 1.5B", Description: "Ultraligero, ideal para autocompletado en CPUs o poca VRAM", HFRepo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"},
	{Name: "Qwen 2.5 Coder 7B", Description: "Gran balance entre velocidad y precisión para código", HFRepo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"},
	{Name: "Llama 3.1 8B Instruct", Description: "Modelo base general pero excelente para código", HFRepo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"},
	{Name: "DeepSeek Coder V2 Lite", Description: "Avanzado, requiere hardware decente", HFRepo: "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF"},
	{Name: "Phind CodeLlama 34B", Description: "Para tareas pesadas de arquitectura", HFRepo: "bartowski/Phind-CodeLlama-34B-v2-GGUF"},
}

var categoryRedaccion = []ModelSuggestion{
	{Name: "Llama 3.2 1B", Description: "Rápido y eficiente para tareas ligeras", HFRepo: "bartowski/Llama-3.2-1B-Instruct-GGUF"},
	{Name: "Llama 3.2 3B", Description: "Versión ligera del poderoso LLaMA 3", HFRepo: "bartowski/Llama-3.2-3B-Instruct-GGUF"},
	{Name: "Phi 3.5 Mini", Description: "Modelo de Microsoft muy competente", HFRepo: "bartowski/Phi-3.5-mini-instruct-GGUF"},
	{Name: "Llama 3.1 8B", Description: "El estándar dorado de la redacción open source", HFRepo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"},
	{Name: "Gemma 2 9B", Description: "Excelentes resultados en tareas generales (Google)", HFRepo: "bartowski/gemma-2-9b-it-GGUF"},
}

var categoryVision = []ModelSuggestion{
	{Name: "Llama 3.2 11B Vision", Description: "LLaMA 3 con capacidades multimodales nativas", HFRepo: "bartowski/Llama-3.2-11B-Vision-Instruct-GGUF"},
	{Name: "LLaVA 1.5 7B", Description: "Modelo multimodal clásico basado en LLaMA", HFRepo: "bartowski/llava-1.5-7b-hf-GGUF"},
	{Name: "LLaVA 1.5 13B", Description: "Visión de alta precisión", HFRepo: "bartowski/llava-1.5-13b-hf-GGUF"},
	{Name: "BakLLaVA 1", Description: "Alternativa a LLaVA con base Mistral", HFRepo: "bartowski/bakllava-1-GGUF"},
	{Name: "LLaVA Phi-3 Mini", Description: "Visión ligera usando Phi-3", HFRepo: "xtuner/llava-phi-3-mini-gguf"},
}

func getHardwareProfile(ctx context.Context) HardwareProfile {
	sysInfo := discover.GetSystemInfo()
	gpus := discover.GPUDevices(ctx, nil)
	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory
	}

	// Fallback for Nvidia Optimus laptops where D3hot sleep state reports 0 VRAM
	if totalVRAM == 0 {
		out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			for _, line := range lines {
				if mem, err := strconv.ParseUint(strings.TrimSpace(line), 10, 64); err == nil {
					totalVRAM += mem * 1024 * 1024 // Convert MiB to Bytes
				}
			}
		}
	}

	return HardwareProfile{
		TotalRAM:  sysInfo.TotalMemory,
		TotalVRAM: totalVRAM,
	}
}

func extractQuantization(path string) string {
	// e.g. Llama-3.2-1B-Instruct-Q4_K_M.gguf
	name := strings.TrimSuffix(path, ".gguf")
	re := regexp.MustCompile(`-([A-Za-z0-9_.]+)$`)
	match := re.FindStringSubmatch(name)
	if len(match) > 1 {
		return match[1]
	}
	return name
}

// fetchBestQuantization returns (best_tag, size, in_vram, error)
func fetchBestQuantization(repo string, hw HardwareProfile) (string, uint64, bool, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(fmt.Sprintf("https://huggingface.co/api/models/%s/tree/main", repo))
	if err != nil {
		return "", 0, false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0, false, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, false, err
	}

	var nodes []HFTreeNode
	if err := json.Unmarshal(body, &nodes); err != nil {
		return "", 0, false, err
	}

	var ggufs []HFTreeNode
	for _, node := range nodes {
		if node.Type == "file" && strings.HasSuffix(node.Path, ".gguf") {
			// Ignore common project files accidentally named .gguf or imatrix
			if strings.Contains(strings.ToLower(node.Path), "imatrix") || strings.Contains(strings.ToLower(node.Path), "mmproj") {
				continue
			}
			ggufs = append(ggufs, node)
		}
	}

	// Sort sizes descending (largest first to prioritize quality)
	sort.Slice(ggufs, func(i, j int) bool {
		return ggufs[i].Size > ggufs[j].Size
	})

	var bestRAMNode *HFTreeNode

	for _, node := range ggufs {
		projectedSize := uint64(float64(node.Size) * 1.2) // ~20% overhead context
		
		// First try to fit in VRAM
		if projectedSize <= hw.TotalVRAM {
			return extractQuantization(node.Path), node.Size, true, nil
		}

		// Otherwise keep track of largest that fits in RAM
		if bestRAMNode == nil && projectedSize <= hw.TotalRAM {
			bestRAMNode = &node
		}
	}

	if bestRAMNode != nil {
		return extractQuantization(bestRAMNode.Path), bestRAMNode.Size, false, nil
	}

	return "", 0, false, fmt.Errorf("no compatible quantization found for hardware")
}

func DiscoverHandler(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	fmt.Println("Analizando hardware...")
	hw := getHardwareProfile(ctx)

	var vramStr string
	if hw.TotalVRAM > 0 {
		vramStr = format.HumanBytes2(hw.TotalVRAM)
	} else {
		vramStr = "0 B (Solo CPU)"
	}

	fmt.Printf("Hardware detectado:\n - RAM: %s\n - VRAM (GPU): %s\n\n",
		format.HumanBytes2(hw.TotalRAM),
		vramStr,
	)

	categories := []tui.SelectItem{
		{Name: "Código", Description: "Modelos optimizados para programación y autocompletado", Recommended: true},
		{Name: "Redacción", Description: "Modelos generales para chat, resúmenes y escritura", Recommended: true},
		{Name: "Visión", Description: "Modelos multimodales para analizar imágenes", Recommended: true},
	}

	catChoice, err := tui.SelectSingle("Selecciona el tipo de modelo que buscas:", categories, "")
	if err != nil {
		if err == tui.ErrCancelled {
			return nil
		}
		return err
	}

	var sourceModels []ModelSuggestion
	switch catChoice {
	case "Código":
		sourceModels = categoryCodigo
	case "Redacción":
		sourceModels = categoryRedaccion
	case "Visión":
		sourceModels = categoryVision
	default:
		return fmt.Errorf("categoría inválida")
	}

	fmt.Println("Buscando las mejores cuantizaciones en HuggingFace (HF)...")
	
	// Fetch models concurrently
	var wg sync.WaitGroup
	var mu sync.Mutex
	dynamicResults := make([]DynamicSuggestion, 0, len(sourceModels))

	for _, m := range sourceModels {
		wg.Add(1)
		go func(model ModelSuggestion) {
			defer wg.Done()
			tag, size, inVram, err := fetchBestQuantization(model.HFRepo, hw)
			if err == nil && tag != "" {
				mu.Lock()
				dynamicResults = append(dynamicResults, DynamicSuggestion{
					ModelSuggestion: model,
					BestTag:         tag,
					Size:            size,
					InVRAM:          inVram,
				})
				mu.Unlock()
			}
		}(m)
	}

	wg.Wait()

	if len(dynamicResults) == 0 {
		fmt.Println("No se encontraron modelos compatibles con el hardware actual para esta categoría.")
		return nil
	}

	// Convert to tui.SelectItems
	var items []tui.SelectItem
	for _, dyn := range dynamicResults {
		mode := "RAM"
		if dyn.InVRAM {
			mode = "VRAM"
		}
		
		desc := fmt.Sprintf("%s (Cuantización Recomendada: %s, %s en %s)", dyn.Description, dyn.BestTag, format.HumanBytes2(dyn.Size), mode)
		badge := "GPU"
		if !dyn.InVRAM {
			badge = "CPU"
		}

		items = append(items, tui.SelectItem{
			Name:              dyn.Name,
			Description:       desc,
			Recommended:       dyn.InVRAM,
			AvailabilityBadge: badge,
		})
	}

	// Sort items so Recommended (InVRAM) appear first, but keeping general order might be fine.
	// Actually we keep order by adding them as they completed, which randomizes.
	// Let's sort them alphabetically by Name to keep it deterministic.
	sort.Slice(items, func(i, j int) bool {
		return items[i].Name < items[j].Name
	})

	modelChoice, err := tui.SelectSingle(fmt.Sprintf("Modelos recomendados (%s):", catChoice), items, "")
	if err != nil {
		if err == tui.ErrCancelled {
			return nil
		}
		return err
	}

	// We need to retrieve the hf.co path from the selected item
	var targetRun string
	for _, dyn := range dynamicResults {
		if dyn.Name == modelChoice {
			targetRun = fmt.Sprintf("hf.co/%s:%s", dyn.HFRepo, dyn.BestTag)
			break
		}
	}

	if targetRun == "" {
		return fmt.Errorf("error resolviendo el modelo")
	}

	fmt.Printf("\nHas seleccionado: %s\n", modelChoice)
	fmt.Printf("Iniciando 'ollama run %s'...\n\n", targetRun)

	// Iniciar RunHandler
	return RunHandler(cmd, []string{targetRun})
}
