This folder holds third-party licenses populated during a release binary build.
If rebuilding Ollama yourself, you may also need to populate it, for example
using go-licenses

```
go run github.com/google/go-licenses@v1.6.0 save . --save_path=licenses/content
```
