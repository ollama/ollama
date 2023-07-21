# Development

Install required tools:

```
brew install go
```

Enable CGO:

```
export CGO_ENABLED=1
```

Then build ollama:

```
go build .
```

Now you can run `ollama`:

```
./ollama
```

## Releasing

To release a new version of Ollama you'll need to set some environment variables:

* `GITHUB_TOKEN`: your GitHub token
* `APPLE_IDENTITY`: the Apple signing identity (macOS only)
* `APPLE_ID`: your Apple ID
* `APPLE_PASSWORD`: your Apple ID app-specific password
* `APPLE_TEAM_ID`: the Apple team ID for the signing identity
* `TELEMETRY_WRITE_KEY`: segment write key for telemetry

Then run the publish script with the target version:

```
VERSION=0.0.2 ./scripts/publish.sh
```




