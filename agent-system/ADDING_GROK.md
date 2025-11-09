# Adding Grok Models to Ollama

Grok models from xAI are not currently available in Ollama's standard library, but you can add them manually if you have the GGUF format.

## Option 1: Use Grok via API (Recommended)

If you want to use Grok, you can add it as an API integration in Settings:
1. Go to Settings in the agent system
2. Add your xAI API key
3. Agents can then use: `[API: xai:chat:your question]`

## Option 2: Convert and Add Grok Manually

If you want to run Grok locally via Ollama:

1. **Download Grok from Hugging Face:**
   ```bash
   git lfs install
   git clone https://huggingface.co/xai-org/grok-2.5
   ```

2. **Convert to GGUF format:**
   You'll need to use conversion tools like `llama.cpp` to convert the model to GGUF format that Ollama can use.

3. **Create a Modelfile:**
   Create a `Modelfile` for Grok:
   ```
   FROM /path/to/grok-2.5-gguf
   TEMPLATE """{{ .System }}{{ .Prompt }}"""
   PARAMETER temperature 0.9
   PARAMETER top_p 0.9
   ```

4. **Create the model in Ollama:**
   ```bash
   ollama create grok:latest -f Modelfile
   ```

## Current Unrestricted Models

The system is already configured to work with unrestricted models that are available:
- `llama3.2:latest` - Good instruction following, unrestricted
- `mistral:7b` - Balanced performance
- `gemma2:9b` - Larger model with good capabilities
- `phi3:mini` - Smaller but capable
- `qwen2.5:0.5b` - Very small but fast

All agents have been configured with **NO RESTRICTIONS** and will engage with any topic you ask about.

