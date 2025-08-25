# Tokenize/Detokenize for ollama-python

## TODO

Add support for `/api/tokenize` and `/api/detokenize` endpoints in ollama-python.

## Proposed Method Signatures

```python
class Ollama:
    def tokenize(self, model: str, content: str, options: Optional[Dict] = None) -> Dict:
        """
        Convert text to token IDs using a model's vocabulary.
        
        Args:
            model: Model name (e.g., "mistral:latest")
            content: Text to convert to tokens
            options: Optional model parameters
            
        Returns:
            Dict with keys: model, tokens, total_duration, load_duration
        """
        pass
    
    def detokenize(self, model: str, tokens: List[int], options: Optional[Dict] = None) -> Dict:
        """
        Convert token IDs back to text using a model's vocabulary.
        
        Args:
            model: Model name (e.g., "mistral:latest")
            tokens: List of token IDs to convert
            options: Optional model parameters
            
        Returns:
            Dict with keys: model, content, total_duration, load_duration
        """
        pass
```

## API Endpoints

- `POST /api/tokenize` - Convert text to tokens
- `POST /api/detokenize` - Convert tokens to text

## Example Usage

```python
import ollama

# Tokenize text
response = ollama.tokenize("mistral:latest", "Hello world")
print(f"Tokens: {response['tokens']}")
print(f"Duration: {response['total_duration']}ns")

# Detokenize tokens
response = ollama.detokenize("mistral:latest", [2050, 1187])
print(f"Content: {response['content']}")
```

## Notes

- Text-only for now (multimodal reserved for future)
- No keep_alive required
- Timing fields for observability
- Vocab-only optimization in development
