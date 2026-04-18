---
name: convert-models-instructions
description: "Use when: adding model architecture support, implementing model conversions, working with GGUF format, or modifying convert/ package"
applyTo: "convert/**"
---

# Model Conversion Package Instructions

## Overview
The `convert/` package handles model format conversions, architecture detection, and format-specific implementations for different LLM architectures.

## Architecture Support Pattern

Each supported model architecture has a dedicated converter file:

```
convert_llama.go       - Llama architecture
convert_qwen.go        - Qwen models
convert_gemma.go       - Google Gemma
convert_mistral.go     - Mistral models
convert_mixtral.go     - Mixtral MoE models
```

## Adding Support for New Architecture

### Step 1: Create Converter File
```go
// convert/convert_newmodel.go
package convert

import (
    "fmt"
    "github.com/ollama/ollama/llm"
)

type NewModelConverter struct {
    name       string
    architecture Architecture
}

func NewNewModelConverter(name string) Converter {
    return &NewModelConverter{
        name: name,
        architecture: // appropriate architecture
    }
}

func (c *NewModelConverter) Convert(
    ctx context.Context,
    modelPath string,
    options ConvertOptions,
) (*llm.ModelInfo, error) {
    // Implementation
    return nil, nil
}
```

### Step 2: Register in convert.go
```go
// In convert.go, add to detector:
func DetectArchitecture(model *safetensors.Model) Architecture {
    if isNewModelArchitecture(model) {
        return ArchitectureNewModel
    }
    // ...
}

// And to converter factory:
func NewConverter(arch Architecture, name string) Converter {
    switch arch {
    case ArchitectureNewModel:
        return NewNewModelConverter(name)
    // ...
    }
}
```

### Step 3: Add Tokenizer (if needed)
```go
// Create tokenizer in tokenizer/ package
// Reference existing tokenizers for pattern
type NewModelTokenizer struct {
    vocab map[int]string
}
```

### Step 4: Add Template Parameters
```
template/newmodel/Modelfile

// Add default parameters for the architecture
PARAMETER num_ctx 2048
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

### Step 5: Add Tests
```go
// convert/convert_newmodel_test.go
func TestNewModelConversion(t *testing.T) {
    converter := NewNewModelConverter("test-model")
    
    // Test with sample weights
    info, err := converter.Convert(context.Background(), "test_model.gguf", 
        ConvertOptions{})
    require.NoError(t, err)
    assert.NotNil(t, info)
}
```

## Converter Interface

All converters must implement:

```go
type Converter interface {
    // Convert returns model information after format conversion
    Convert(ctx context.Context, modelPath string, 
        options ConvertOptions) (*ModelInfo, error)
}
```

## Common Converter Patterns

### Architecture Detection
- Check weight names for architecture-specific patterns
- Look for characteristic layer structures
- Examine configuration files (config.json)

### Weight Mapping
```go
// Pattern: Map source weights to GGUF format
func (c *Converter) mapWeights(srcWeights map[string]Tensor) map[string]Tensor {
    result := make(map[string]Tensor)
    
    for name, tensor := range srcWeights {
        newName := c.mapWeightName(name)
        result[newName] = tensor
    }
    
    return result
}
```

### Parameter Transfer
```go
// Extract and validate model parameters
func (c *Converter) extractParams(config map[string]interface{}) Params {
    return Params{
        VocabSize: int(config["vocab_size"].(float64)),
        HiddenSize: int(config["hidden_size"].(float64)),
        NumLayers: int(config["num_hidden_layers"].(float64)),
        NumHeads: int(config["num_attention_heads"].(float64)),
    }
}
```

## Supported Model Formats

Current support:
- **GGUF**: Default format
- **SafeTensors**: Via conversion to GGUF
- **PyTorch**: Via conversion to GGUF
- **HuggingFace**: Via conversion pipeline

## Testing Model Conversions

```go
func TestModelConversionRoundtrip(t *testing.T) {
    // 1. Load original model
    original, err := loadModel("model.safetensors")
    require.NoError(t, err)
    
    // 2. Convert to GGUF
    converter := NewConverter(DetectArchitecture(original))
    converted, err := converter.Convert(context.Background(), "model.gguf", 
        ConvertOptions{})
    require.NoError(t, err)
    
    // 3. Verify converted model loads correctly
    assert.NotNil(t, converted)
    assert.Equal(t, original.VocabSize, converted.VocabSize)
}
```

## Performance Considerations

- Streaming conversion for large models
- Memory-efficient weight mapping
- Progress reporting during long conversions
- Parallel processing where possible

## Documentation for New Architecture

When adding support for a new architecture, document:
1. Model sources (HuggingFace, official repos)
2. Parameter conversion mapping
3. Any architecture-specific quirks
4. Performance characteristics
5. Tokenizer behavior

Add to `docs/model-architectures.md` or similar.

## Quality Checklist

Before merging a new converter:
- [ ] Tests pass with sample models
- [ ] Inference works with converted model
- [ ] Token generation produces reasonable output
- [ ] Memory usage is acceptable
- [ ] Performance is on par with other architectures
- [ ] Documentation is complete
- [ ] Commit message follows format: `convert: add support for NewModel architecture`
