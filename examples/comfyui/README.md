# ComfyUI Workflows for Ollama

This directory contains ComfyUI workflow templates for integrating with Ollama. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a powerful and modular node-based interface for working with AI models, primarily focused on image generation but extensible to text generation and other AI tasks.

## Additional Guides

- **[Wan-AI Wan2.1-T2V-14B Setup Guide](./wan-ai-t2v-setup.md)** - Instructions for setting up the Wan-AI text-to-video model with ComfyUI

## Prerequisites

1. **Ollama** running locally (default: `http://localhost:11434`)
   - Install from [ollama.com](https://ollama.com)
   - Start the server: `ollama serve`
   - Pull a model: `ollama pull llama3.2`

2. **ComfyUI** installed and running
   - Clone from [GitHub](https://github.com/comfyanonymous/ComfyUI)
   - Follow the [installation guide](https://github.com/comfyanonymous/ComfyUI#installing)
   - Launch ComfyUI (typically at `http://127.0.0.1:8188`)

3. **ComfyUI Custom Nodes** for Ollama (Required)
   - These workflows require custom nodes to interact with Ollama's API
   - See [Installation](#installing-ollama-custom-nodes) section below

## Available Workflows

### 1. Text Generation Workflow (`ollama-text-generation.json`)

A simple workflow demonstrating basic text generation with Ollama.

**Features:**
- Single prompt input
- Configurable model and parameters
- Direct text output display

**Use Cases:**
- Text completion
- Question answering
- Content generation
- Code generation

**Configuration:**
- **Server URL**: `http://localhost:11434` (default)
- **Model**: `llama3.2` (default)
- **Temperature**: `0.7` (controls randomness)
- **Max Tokens**: `2048` (maximum response length)

### 2. Chat Workflow (`ollama-chat-workflow.json`)

An interactive chat workflow with system prompt configuration and conversation support.

**Features:**
- System prompt for setting assistant behavior
- User message input
- Formatted chat response
- Model information display

**Use Cases:**
- Interactive conversations
- Task-specific assistants
- Q&A systems
- Customer support automation

**Configuration:**
- **System Prompt**: Define the assistant's role and behavior
- **Server URL**: `http://localhost:11434` (default)
- **Model**: `llama3.2` (default)
- **Temperature**: `0.7`
- **Max Tokens**: `2048`

## Installing Ollama Custom Nodes

To use these workflows, you need to install custom nodes that provide Ollama integration for ComfyUI. There are several options:

### Option 1: ComfyUI-Ollama (Recommended)

This is a community-maintained custom node pack for Ollama integration.

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone the Ollama custom nodes repository:
   ```bash
   git clone https://github.com/stavsap/comfyui-ollama.git
   ```

3. Install dependencies (if required):
   ```bash
   cd comfyui-ollama
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Option 2: ComfyUI Manager

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if not already installed
2. Open ComfyUI in your browser
3. Click on "Manager" button
4. Search for "Ollama"
5. Install the Ollama custom nodes
6. Restart ComfyUI

### Option 3: Manual Implementation

If custom nodes are not available, you can use HTTP Request nodes to call Ollama's API directly. See the [Custom Implementation](#custom-implementation-without-custom-nodes) section below.

## Importing Workflows

1. Open ComfyUI in your browser (typically `http://127.0.0.1:8188`)
2. Click the "Load" button (or drag and drop the JSON file)
3. Navigate to this directory and select the desired workflow JSON file
4. The workflow will be loaded into ComfyUI
5. Configure the nodes as needed (server URL, model, etc.)
6. Click "Queue Prompt" to execute the workflow

## Configuration

### Ollama Server URL

By default, workflows connect to `http://localhost:11434`. To change the Ollama server URL:

#### In ComfyUI UI:

1. Open the workflow
2. Click on the Ollama node (e.g., "Ollama Generate" or "Ollama Chat")
3. Find the **Server URL** parameter
4. Change `http://localhost:11434` to your Ollama server address
5. Save the workflow

#### Before Importing:

Edit the workflow JSON file before importing:

1. Open the `.json` file in a text editor
2. Find all occurrences of `http://localhost:11434`
3. Replace with your Ollama server URL
4. Save and import into ComfyUI

**Example URLs:**
- Local: `http://localhost:11434`
- LAN server: `http://192.168.1.100:11434`
- Docker: `http://host.docker.internal:11434`
- Remote: `https://ollama.example.com:11434`

#### Verify Ollama Connection:

Test your Ollama server is reachable:

```bash
curl http://YOUR_OLLAMA_URL/api/version
```

Expected response:
```json
{"version": "0.5.1"}
```

### Model Selection

You can change the model in any workflow by:

1. Opening the Ollama node in ComfyUI
2. Changing the **Model** parameter
3. Using any model you have pulled with Ollama

**Popular Models:**
- `llama3.2` - Fast and efficient (default)
- `llama3.2:70b` - Large model for complex tasks
- `mistral` - Good balance of speed and quality
- `codellama` - Optimized for code generation
- `phi3` - Compact and efficient
- `gemma2` - Google's efficient model

**List Available Models:**

```bash
ollama list
```

Or via API:

```bash
curl http://localhost:11434/api/tags
```

### Parameter Tuning

#### Temperature (0.0 - 2.0)
- **Lower (0.0-0.3)**: More focused and deterministic responses
- **Medium (0.4-0.9)**: Balanced creativity and coherence
- **Higher (1.0-2.0)**: More creative and varied responses

#### Max Tokens
- Controls the maximum length of the generated response
- Higher values allow longer responses but take more time
- Typical range: 256 to 4096

#### Top P (0.0 - 1.0)
- Alternative to temperature for controlling randomness
- Lower values make output more focused
- Default: 0.9

## Advanced Usage

### Multi-Turn Conversations

To implement multi-turn conversations with context:

1. Store previous messages in a list/array
2. Pass the full conversation history to the Ollama Chat API
3. Append new responses to the history

Example message format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for..."},
    {"role": "user", "content": "Tell me more about machine learning."}
  ]
}
```

### Streaming Responses

For real-time streaming responses:

1. Enable streaming in the Ollama node configuration
2. Set up a display node that updates in real-time
3. Process chunks as they arrive

### Combining with Image Generation

ComfyUI excels at image generation. You can combine Ollama with image models:

1. Use Ollama to generate image descriptions or prompts
2. Pass the generated text to Stable Diffusion nodes
3. Create a fully automated text-to-image pipeline

Example workflow:
```
Text Input → Ollama (prompt enhancement) → Stable Diffusion → Image Output
```

### RAG (Retrieval Augmented Generation)

Implement RAG with ComfyUI and Ollama:

1. Use document processing nodes to extract text
2. Store embeddings in a vector database
3. Retrieve relevant context
4. Pass context to Ollama for grounded responses

## Custom Implementation (Without Custom Nodes)

If Ollama custom nodes are not available, you can use HTTP Request nodes to call Ollama's API directly.

### Text Generation API

**Endpoint:** `POST http://localhost:11434/api/generate`

**Request Body:**
```json
{
  "model": "llama3.2",
  "prompt": "Your prompt here",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 2048
  }
}
```

**Response:**
```json
{
  "model": "llama3.2",
  "created_at": "2024-01-01T00:00:00Z",
  "response": "Generated text here...",
  "done": true
}
```

### Chat API

**Endpoint:** `POST http://localhost:11434/api/chat`

**Request Body:**
```json
{
  "model": "llama3.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.7
  }
}
```

**Response:**
```json
{
  "model": "llama3.2",
  "created_at": "2024-01-01T00:00:00Z",
  "message": {
    "role": "assistant",
    "content": "The capital of France is Paris."
  },
  "done": true
}
```

### ComfyUI HTTP Request Node Configuration

1. Add an "HTTP Request" node to your workflow
2. Set **Method** to `POST`
3. Set **URL** to `http://localhost:11434/api/generate` or `http://localhost:11434/api/chat`
4. Set **Headers** to `Content-Type: application/json`
5. Set **Body** to the JSON request format above
6. Use a JSON parser node to extract the response

## Troubleshooting

### Connection Refused

If you get a "connection refused" error:

1. **Verify Ollama is running:**
   ```bash
   ps aux | grep ollama
   ```
   If not running, start it:
   ```bash
   ollama serve
   ```

2. **Check the server URL** in the workflow nodes
   - Default is `http://localhost:11434`
   - Ensure no typos and correct port

3. **Verify firewall settings** if using a remote server
   - Ensure port 11434 is open
   - Check network connectivity

4. **Test with curl:**
   ```bash
   curl http://localhost:11434/api/version
   ```

### Custom Nodes Not Found

If ComfyUI can't find the Ollama nodes:

1. **Verify installation:**
   ```bash
   ls ComfyUI/custom_nodes/
   ```
   Look for the Ollama custom nodes directory

2. **Check ComfyUI console** for error messages during startup

3. **Restart ComfyUI** after installing custom nodes

4. **Check dependencies:**
   ```bash
   cd ComfyUI/custom_nodes/comfyui-ollama
   pip install -r requirements.txt
   ```

5. **Use ComfyUI Manager** to reinstall the custom nodes

### Model Not Found

If you get a "model not found" error:

1. **Pull the model first:**
   ```bash
   ollama pull llama3.2
   ```

2. **Verify model name** in the workflow:
   ```bash
   ollama list
   ```

3. **Check spelling** - model names are case-sensitive

### Timeout Errors

For large models or long responses:

1. **Increase timeout** in the HTTP Request node settings
   - Default: 60 seconds
   - Recommended: 120-300 seconds for large models

2. **Use a smaller model** for faster responses:
   - `llama3.2` instead of `llama3.2:70b`
   - `phi3` for quick tasks

3. **Reduce max_tokens** parameter to limit response length

### Slow Performance

If generation is slow:

1. **Check GPU usage** (if available):
   ```bash
   nvidia-smi  # For NVIDIA GPUs
   ```

2. **Use quantized models** for faster inference:
   ```bash
   ollama pull llama3.2:q4_0  # 4-bit quantized
   ```

3. **Reduce context length** in the model parameters

4. **Close other GPU-intensive applications**

### Invalid JSON Response

If you get JSON parsing errors:

1. **Ensure streaming is disabled** (`"stream": false`)
2. **Check response format** in the HTTP Request node
3. **Use a JSON validator** to verify the response structure
4. **Check Ollama version** - update if necessary:
   ```bash
   ollama --version
   curl -fsSL https://ollama.com/install.sh | sh
   ```

### Empty or Unexpected Output

If the workflow produces no output or unexpected results:

1. **Check the prompt** - ensure it's clear and specific
2. **Verify model parameters** - temperature, max_tokens, etc.
3. **Test with Ollama CLI** to isolate the issue:
   ```bash
   ollama run llama3.2 "Your prompt here"
   ```
4. **Check ComfyUI console** for error messages
5. **Inspect node connections** - ensure data flows correctly

## Performance Optimization

### Model Selection
- Use smaller models for faster responses: `llama3.2`, `phi3`, `mistral`
- Use larger models for complex tasks: `llama3.2:70b`, `mixtral:8x7b`

### GPU Acceleration
- Ollama automatically uses GPU if available (NVIDIA, AMD, Apple Silicon)
- Check GPU usage: `nvidia-smi` or Activity Monitor (Mac)

### Quantization
- Use quantized models for better performance:
  - `q4_0`: 4-bit quantization (fastest)
  - `q5_0`: 5-bit quantization (balanced)
  - `q8_0`: 8-bit quantization (best quality)

Example:
```bash
ollama pull llama3.2:q4_0
```

### Batch Processing
- Process multiple prompts in parallel when possible
- Use ComfyUI's batch processing features

### Caching
- Ollama caches models in memory after first use
- Keep Ollama running to avoid reload delays

## Extending Workflows

These workflows can be extended with:

### Image Processing
- Generate image captions with Ollama
- Create enhanced prompts for Stable Diffusion
- Analyze and describe images

### Document Processing
- Extract and summarize text from PDFs
- Process and analyze documents
- Generate reports from data

### Audio Integration
- Transcribe audio with Whisper
- Generate responses with Ollama
- Text-to-speech output

### Database Integration
- Store conversation history
- Implement RAG with vector databases
- Log and analyze interactions

### External APIs
- Combine with OpenAI, Anthropic, etc.
- Multi-model ensemble responses
- Fallback chains for reliability

## Security Considerations

### Local vs Remote Servers

- **Local**: Most secure, data stays on your machine
- **LAN**: Ensure network is trusted, use VPN if needed
- **Internet**: Use HTTPS, implement authentication, consider privacy

### API Security

1. **Don't expose Ollama publicly without protection**
   - Use a reverse proxy with authentication
   - Implement rate limiting
   - Use API keys or tokens

2. **Sanitize inputs** to prevent injection attacks

3. **Monitor usage** to detect abuse

4. **Keep Ollama updated** for security patches

### Data Privacy

- Ollama runs locally by default - your data stays private
- No data is sent to external servers
- Models can be run completely offline

## Related Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama API Documentation](../../docs/api.md)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Ollama Model Library](https://ollama.com/library)
- [ComfyUI Custom Nodes](https://github.com/ltdrdata/ComfyUI-Manager)

## Examples and Use Cases

### Content Generation
```
User Prompt → Ollama (content generation) → Format Output → Save
```

### Automated Image Captioning
```
Load Image → CLIP/BLIP (caption) → Ollama (enhance caption) → Save
```

### Interactive Chatbot
```
User Input → Ollama Chat → Response Display → Store History
```

### Code Generation
```
Requirements → Ollama (codellama) → Code Output → Syntax Highlighting
```

### Multi-Language Translation
```
Input Text → Ollama (translate) → Translated Output
```

## Contributing

Contributions are welcome! If you create useful workflows or improvements:

1. Test thoroughly with different models and parameters
2. Document any special requirements or dependencies
3. Add clear instructions and examples
4. Submit a pull request with your changes

## FAQ

**Q: Which model should I use?**
A: Start with `llama3.2` for general use. Use `codellama` for code, `llama3.2:70b` for complex reasoning.

**Q: Can I use multiple models in one workflow?**
A: Yes! Create multiple Ollama nodes with different models for ensemble responses.

**Q: How do I save conversation history?**
A: Use file save nodes or database nodes to persist messages between workflow runs.

**Q: Can I use Ollama with image generation?**
A: Absolutely! Use Ollama to generate or enhance prompts for Stable Diffusion nodes.

**Q: Is GPU required?**
A: No, but strongly recommended. Ollama works on CPU but is much slower.

**Q: How much VRAM do I need?**
A: Depends on the model:
  - Small models (llama3.2, phi3): 4-8 GB
  - Medium models (mistral): 8-16 GB
  - Large models (llama3.2:70b): 40+ GB

**Q: Can I run this in production?**
A: Yes, but ensure proper monitoring, error handling, and security measures.

## License

These workflows are provided as examples and are free to use and modify. Please comply with Ollama's and ComfyUI's respective licenses.
