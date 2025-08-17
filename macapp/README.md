# Desktop

This app builds upon Ollama to provide a desktop experience for running models.

## Developing

First, build the `ollama` binary:

```shell
cd ..
go build .
```

Then run the desktop app with `npm start`:

```shell
cd macapp
npm install
npm start
```

## Turbo Provider (Placeholder)

The UI now exposes a `Turbo (coming soon)` checkbox in the chat view. This toggle is **off by default** so that all model interaction uses the local provider. Selecting any preset that would normally route to a remote / turbo provider keeps execution local until the toggle is explicitly enabled.

When you enable the toggle a transient notice appears: *"Turbo provider enabled (feature coming soon)."* Remote preset entries are disabled while the toggle is off and labelled `(Turbo)` to make this distinction clear. Attempting to send with a turbo-only model while disabled shows an inline error instead of making a failing network call.

Implementation details:

- State persisted via `electron-store` key `turboEnabled`.
- Provider selection gate: if turbo is disabled, any model that maps to the turbo provider is forced to use the local provider (or prevented entirely when sending if remote-only).
- This is a UI affordance only; the real remote TurboProvider integration will replace the dummy logic later.

You can safely ignore this feature during local development if focusing on core local model functionality.
