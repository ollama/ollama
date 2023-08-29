# Development

Install required tools:

```
brew install go cmake gcc
```

Get the required libraries:

```
go generate ./...
```

Then build ollama:

```
go build .
```

Now you can run `ollama`:

```
./ollama
```
