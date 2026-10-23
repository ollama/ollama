# Releasing this fork

A release is something you start, on the Mac Studio. Nothing in CI publishes
one, and a normal push does not either.

```sh
bash scripts/release-fork.sh patch
```

`patch` reads the newest `vX.Y.Z-bNNNNN-ds41[.N]` tag and increments the
trailing number (`v0.40.5-b11724-ds41.3` becomes `v0.40.5-b11724-ds41.4`).
An explicit tag works too, and it has to be strictly newer:

```sh
bash scripts/release-fork.sh v0.40.5-b11724-ds41.4
```

The command refuses a dirty tree. It runs the Go tests for the updater, then
`scripts/e2e-update-mac.sh` (an old signed stub updates to a new one, and a
tampered zip is rejected). It tags only after that, builds a universal
`Ollama.app`, signs it with the Developer ID Application certificate, notarizes
with the `AC_PASSWORD` notarytool profile, staples, and checks Gatekeeper.
It then writes:

- `Ollama-<version>-darwin-universal.zip` (this is what the app installs)
- `Ollama-<version>-darwin-universal.dmg`
- `Ollama.dmg` (the stable README download; same bytes as the versioned DMG)
- `sha256sum.txt`
- a `<file>.ed25519` signature next to each of those

and uploads them to a GitHub Release of `maxkongerskov/ollama` tagged
`v<version>`, marked latest. If any step fails before that release is marked
latest, the script deletes the tag and the draft. It does not leave a
half-published update.

`OLLAMA_SIGN_ID`, `OLLAMA_NOTARY_PROFILE`, and `OLLAMA_UPDATE_KEY` override the
certificate, the notarytool profile, and the ed25519 private key.

## One-time setup

1. Install a **Developer ID Application** certificate in the login keychain
   for team `6LZ2DS9JPD`. `security find-identity -v -p codesigning` should
   show it.

2. Store the notarytool profile the script looks up by default:

   ```sh
   xcrun notarytool store-credentials AC_PASSWORD \
     --apple-id "you@example.com" \
     --team-id 6LZ2DS9JPD
   ```

   Use an app-specific password. Confirm with
   `xcrun notarytool history --keychain-profile AC_PASSWORD`.

3. `gh auth login` with permission to publish releases on
   `maxkongerskov/ollama`.

4. Generate the update signing key once, on this Mac:

   ```sh
   bash scripts/generate-update-key.sh
   ```

   That writes the private key to `~/.config/ollama-fork/update-private.pem`
   (mode 0600) and rewrites `app/updatetrust/public_key.go`. Commit **only**
   the public key file. The private key is gitignored and stays on this Mac.
   Until that public key is committed, the release script and the updater both
   refuse to install an update.

## Check without publishing

On the Mac, confirm the universal build before a release. This compiles,
lipos, and signs when a Developer ID is present. It does not notarize, tag,
or upload:

```sh
bash scripts/release-fork.sh --build-only
```

It reuses the existing `build/` directory and relinks targets whose rpath
flags changed. When it prints `Build ok` and the architectures include
`arm64` and `x86_64`, publish with `bash scripts/release-fork.sh patch`.

On Linux, or before a real release:

```sh
bash scripts/release-fork.sh --self-test
bash scripts/release-fork.sh --dry-run patch
```

`--self-test` runs the Go tests (including a tampered-artifact rejection),
the version CLI, and the Darwin lipo/rpath checks that do not need a Mac.
`--dry-run` prints the next tag and the Mac plan. Neither tags, builds,
signs, or uploads.

The old-to-new install itself only runs on macOS:

```sh
bash scripts/e2e-update-mac.sh
```

`release-fork.sh` runs that rehearsal itself. A release is not done until it
passes on the Mac: the stub app must move from `ds41.900` to `ds41.901`, and
the tampered zip must be rejected with the installed app left unchanged.

## What the app checks

The running app downloads the zip and its `.ed25519` sidecar. The sidecar is
standard base64 of an ed25519 signature over the raw 32-byte SHA-256 of the
file. The app then checks the Apple code signature inside the zip. Both have
to succeed before the files replace `Ollama.app`.
