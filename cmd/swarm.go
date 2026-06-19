package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/spf13/cobra"
	"github.com/ollama/ollama/cmd/tui"
	"github.com/ollama/ollama/api"
)

var swarmModels string

func SwarmHandler(cmd *cobra.Command, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("se requiere un prompt para el swarm")
	}
	userPrompt := strings.Join(args, " ")

	models := strings.Split(swarmModels, ",")
	var activeModels []string
	for _, m := range models {
		m = strings.TrimSpace(m)
		if m != "" {
			activeModels = append(activeModels, m)
		}
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("error conectando a ollama: %w", err)
	}
	ctx := cmd.Context()

	if len(activeModels) == 0 {
		// Fetch local models
		listResp, err := client.List(ctx)
		if err != nil {
			return fmt.Errorf("error obteniendo modelos: %w", err)
		}

		if len(listResp.Models) == 0 {
			return fmt.Errorf("no tienes modelos instalados. Haz `ollama run <modelo>` primero")
		}

		var items []tui.SelectItem
		for _, m := range listResp.Models {
			items = append(items, tui.SelectItem{
				Name: m.Name,
				Description: fmt.Sprintf("Tamaño: %.1f GB", float64(m.Size)/(1024*1024*1024)),
			})
		}

		fmt.Println("No se especificaron modelos con --models.")
		selected, err := tui.SelectMultiple("Selecciona los modelos para el Escuadrón (Espacio para marcar, Enter para confirmar):", items, nil)
		if err != nil {
			return err
		}

		if len(selected) == 0 {
			return fmt.Errorf("debes seleccionar al menos un modelo")
		}
		activeModels = selected
	}

	architectModel := activeModels[0] // Use the first model as architect

	// 1. Arquitecto: Generar el plan
	fmt.Printf("> **[Arquitecto Local (%s)]** Analizando la petición...\n", architectModel)
	
	archPrompt := "You are an Expert AI Software Architect.\n" +
		"Analyze this request:\n" + userPrompt + "\n\n" +
		"Output a strict execution plan like this:\n" +
		"[FILE] path/to/file1.ext | {one-line purpose}\n" +
		"[FILE] path/to/file2.ext | {one-line purpose}\n" +
		"Failure to format as [FILE] path | purpose will BREAK the downstream parser."

	archResp, err := generateText(ctx, client, architectModel, archPrompt)
	if err != nil {
		return fmt.Errorf("error del arquitecto: %w", err)
	}

	fmt.Printf("\n> **[Arquitecto] Plan generado:**\n%s\n---\n", archResp)

	var files []string
	filePurposes := make(map[string]string)

	lines := strings.Split(archResp, "\n")
	for _, line := range lines {
		if strings.Contains(line, "[FILE]") {
			parts := strings.SplitN(line, "|", 2)
			if len(parts) == 2 {
				path := strings.TrimSpace(strings.ReplaceAll(parts[0], "[FILE]", ""))
				purpose := strings.TrimSpace(parts[1])
				files = append(files, path)
				filePurposes[path] = purpose
			}
		}
	}

	if len(files) == 0 {
		fmt.Println("> ⚠️ **[Sistema]** No se detectaron archivos en el plan. Saliendo.")
		return nil
	}

	// 2. Ejecutar por chunks
	for i, file := range files {
		fmt.Printf("\n> 📦 **Procesando archivo (%d/%d):** `%s`\n", i+1, len(files), file)
		purpose := filePurposes[file]

		err := processFileSwarm(ctx, client, file, purpose, userPrompt, activeModels)
		if err != nil {
			fmt.Printf("\n> ❌ Error procesando %s: %v\n", file, err)
		}
	}

	fmt.Println("\n> 🎉 **[Orquestador Swarm] Tarea completamente finalizada.**")
	return nil
}

func generateText(ctx context.Context, client *api.Client, modelName, prompt string) (string, error) {
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: new(bool), // false
	}
	*req.Stream = false

	var fullResp string
	fn := func(resp api.GenerateResponse) error {
		fullResp += resp.Response
		return nil
	}

	err := client.Generate(ctx, req, fn)
	return fullResp, err
}

func processFileSwarm(ctx context.Context, client *api.Client, file, purpose, overallGoal string, localModels []string) error {
	basePrompt := fmt.Sprintf("You are an Expert Developer implementing exactly ONE file.\n"+
		"FILE: %s\nPURPOSE: %s\n\nOverall Project Goal: %s\n\n"+
		"CRITICAL INSTRUCTIONS:\n- Do NOT use placeholders. Write the full file.", file, purpose, overallGoal)

	var currentCode string
	maxIter := 3
	iterCount := 1

	for iterCount <= maxIter {
		turnCount := 0
		noChangeCount := 0
		modelIdx := 0

		for turnCount < (maxIter * len(localModels)) {
			if turnCount > 0 && noChangeCount >= len(localModels) {
				fmt.Printf("\n> 🤝 **[Consenso]** Los %d agentes están de acuerdo.\n", len(localModels))
				break
			}

			currentModel := localModels[modelIdx%len(localModels)]
			fmt.Printf("> **[Ollama Local (%s)]** Turno %d. Paso %d/%d...\n", currentModel, turnCount+1, (modelIdx%len(localModels))+1, len(localModels))

			var prompt string
			if turnCount == 0 && currentCode == "" {
				prompt = basePrompt + "\n\nCRITICAL INSTRUCTION: You are the FIRST developer. Write the COMPLETE implementation. Output ONLY the raw content inside standard markdown blocks (```). Do NOT use search/replace blocks."
			} else {
				prompt = basePrompt + "\n\n=================================\n\n" +
					"You are a reviewer in the swarm consensus loop. Please review the CURRENT DRAFT and improve it, expand it, or fix issues.\n" +
					"CURRENT DRAFT:\n```\n" + currentCode + "\n```\n\n" +
					"CRITICAL INSTRUCTION: You MUST ONLY output SEARCH/REPLACE blocks. Do NOT output the full document.\n" +
					"If you think the draft is perfect and needs no changes, output ABSOLUTELY NOTHING. Just return an empty response.\n" +
					"Format for patches:\n" +
					"<<<<\n[exact original lines to replace]\n====\n[new improved lines]\n>>>>\n"
			}

			codeResp, err := generateText(ctx, client, currentModel, prompt)
			if err != nil {
				fmt.Printf("> ⚠️ Error con modelo %s: %v. Saltando...\n", currentModel, err)
				noChangeCount++
			} else {
				if turnCount == 0 && currentCode == "" {
					currentCode = extractMarkdownCode(codeResp)
					noChangeCount = 0
				} else {
					changed := applyPatches(&currentCode, codeResp)
					if changed {
						noChangeCount = 0
						fmt.Printf("> 🛠️ **[%s]** Aplicó parches al documento.\n", currentModel)
					} else {
						noChangeCount++
						fmt.Printf("> 🤷 **[%s]** No propuso cambios. (Consenso: %d/%d)\n", currentModel, noChangeCount, len(localModels))
					}
				}
			}

			turnCount++
			modelIdx++
		}

		// Self-Review
		lastModel := localModels[len(localModels)-1]
		fmt.Printf("> **[Auto-Review]** Evaluando con %s...\n", lastModel)
		reviewPrompt := "Review the code you generated for " + file + ".\nCODE:\n" + currentCode +
			"\nReply with a JSON object: {\"score\": 100, \"fixes\": \"\"}. Score <90 if issues exist. Output ONLY JSON."
		
		revResp, _ := generateText(ctx, client, lastModel, reviewPrompt)
		
		type Review struct {
			Score int    `json:"score"`
			Fixes string `json:"fixes"`
		}
		var rev Review
		
		// Parse json from revResp
		startJSON := strings.Index(revResp, "{")
		endJSON := strings.LastIndex(revResp, "}")
		if startJSON != -1 && endJSON != -1 && endJSON > startJSON {
			jsonStr := revResp[startJSON : endJSON+1]
			_ = json.Unmarshal([]byte(jsonStr), &rev)
		}

		if rev.Score == 0 {
			rev.Score = 100 // fallback
		}

		fmt.Printf("> **[Puntuación]** %d/100\n", rev.Score)

		// Aprobación Humana interactiva
		fmt.Printf("\n### 📝 Contenido Propuesto para `%s`:\n```\n%s\n```\n", file, currentCode)
		fmt.Printf("⏳ ¿Aprobar %s? (Vacío=SI, Texto=Feedback/Corregir): ", file)
		
		reader := bufio.NewReader(os.Stdin)
		feedback, _ := reader.ReadString('\n')
		feedback = strings.TrimSpace(feedback)

		if feedback == "" || strings.EqualFold(feedback, "/approve") || strings.EqualFold(feedback, "/ok") {
			// Aprobado, guardar
			err := os.WriteFile(file, []byte(currentCode), 0644)
			if err != nil {
				return fmt.Errorf("error guardando archivo %s: %w", file, err)
			}
			fmt.Printf("### 💾 Guardado en disco: `%s`\n", file)
			break
		} else {
			fmt.Printf("> 🧑‍💼 **[Director Humano]** Exige: %s\n", feedback)
			basePrompt += fmt.Sprintf("\n\nHUMAN FEEDBACK TO FIX:\n%s\n", feedback)
			iterCount++
		}
	}

	if iterCount > maxIter {
		fmt.Printf("\n### ⚠️ [Sistema] Máximo de iteraciones alcanzado para %s. Guardando tal como está.\n", file)
		_ = os.WriteFile(file, []byte(currentCode), 0644)
	}

	return nil
}

func extractMarkdownCode(text string) string {
	re := regexp.MustCompile("(?s)```[a-zA-Z0-9]*\n(.*?)\n```")
	matches := re.FindStringSubmatch(text)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}
	return strings.TrimSpace(text)
}

func applyPatches(currentCode *string, response string) bool {
	changed := false
	re := regexp.MustCompile("(?s)<<<<(.*?)====(.*?)>>>>")
	matches := re.FindAllStringSubmatch(response, -1)
	
	for _, m := range matches {
		if len(m) == 3 {
			search := strings.TrimSpace(m[1])
			replace := strings.TrimSpace(m[2])
			if search != "" {
				if strings.Contains(*currentCode, search) {
					*currentCode = strings.Replace(*currentCode, search, replace, 1)
					changed = true
				}
			}
		}
	}
	return changed
}
