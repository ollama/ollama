# eval

Evaluation tool for testing Ollama models.

## Usage

Run all tests:

```bash
go run . -model llama3.2:latest
```

Run specific suite:

```bash
go run . -model llama3.2:latest -suite tool-calling-basic -v
```

List available suites:

```bash
go run . -list
```

## Adding Tests

Edit `suites.go` to add new test suites. Each test needs:

- `Name`: test identifier
- `Prompt`: what to send to the model
- `Check`: function to validate the response

Example:

```go
{
    Name:   "my-test",
    Prompt: "What is 2+2?",
    Check:  Contains("4"),
}
```

Available check functions:

- `HasResponse()` - response is non-empty
- `Contains(s)` - response contains substring
- `CallsTool(name)` - model called specific tool
- `NoTools()` - model called no tools
- `MinTools(n)` - model called at least n tools
- `All(checks...)` - all checks pass
