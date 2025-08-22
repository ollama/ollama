# Tokenize/Detokenize for ollama-js

## TODO

Add support for `/api/tokenize` and `/api/detokenize` endpoints in ollama-js.

## Proposed Method Signatures

```typescript
interface Ollama {
  tokenize(params: {
    model: string;
    content: string;
    options?: Record<string, any>;
  }): Promise<{
    model: string;
    tokens: number[];
    total_duration: number;
    load_duration: number;
  }>;
  
  detokenize(params: {
    model: string;
    tokens: number[];
    options?: Record<string, any>;
  }): Promise<{
    model: string;
    content: string;
    total_duration: number;
    load_duration: number;
  }>;
}
```

## API Endpoints

- `POST /api/tokenize` - Convert text to tokens
- `POST /api/detokenize` - Convert tokens to text

## Example Usage

```typescript
import { Ollama } from 'ollama';

const ollama = new Ollama();

// Tokenize text
const tokenizeResponse = await ollama.tokenize({
  model: 'mistral:latest',
  content: 'Hello world'
});
console.log('Tokens:', tokenizeResponse.tokens);
console.log('Duration:', tokenizeResponse.total_duration);

// Detokenize tokens
const detokenizeResponse = await ollama.detokenize({
  model: 'mistral:latest',
  tokens: [2050, 1187]
});
console.log('Content:', detokenizeResponse.content);
```

## Notes

- Text-only for now (multimodal reserved for future)
- No keep_alive required
- Timing fields for observability
- Vocab-only optimization in development
