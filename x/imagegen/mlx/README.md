# MLX Memory Management

| This package will get consolidated with `x/ml/backend/mlx` in the future.

## Automatic Tracking

All arrays are automatically tracked when created. On `Eval()`, non-kept arrays are freed.

### API

```go
result := mlx.Matmul(x, w) // arrays automatically tracked
mlx.Eval(result)           // free non-kept, eval result (auto-kept)
```

### Key Functions

- `mlx.Eval(outputs...)` - free non-kept arrays, then evaluate (outputs auto-kept)
- `mlx.AsyncEval(outputs...)` - async version of Eval (outputs auto-kept)
- `mlx.Keep(arrays...)` - mark arrays to survive cleanup (for weights, caches)
- `array.Free()` - mark array for cleanup on next Eval

### Loop Pattern

```go
for step := 0; step < maxTokens; step++ {
    logits := model.Forward(token, caches)
    oldToken := token
    token = sample(logits)

    // Keep cache state across iterations
    for _, c := range caches {
        mlx.Keep(c.State()...)
    }

    oldToken.Free()       // mark for cleanup
    mlx.AsyncEval(token)  // frees old, evals new
}
```

### Notes

- `Eval()` and `AsyncEval()` auto-keep their outputs
- `Free()` marks for cleanup - actual free happens during next Eval
- Use `Keep()` for weights and cache state that must survive multiple Eval cycles
- Arrays created inside compiled closures are managed by MLX, not tracked
