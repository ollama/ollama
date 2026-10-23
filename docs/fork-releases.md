# Fork app updates (maxkongerskov/ollama)

The macOS app in this fork updates itself only from GitHub releases of
[maxkongerskov/ollama](https://github.com/maxkongerskov/ollama). It never
asks ollama.com or ollama/ollama. The updater lives in `app/updater/`.

The one-command release is [RELEASING.md](../RELEASING.md):
`bash scripts/release-fork.sh patch`, run by hand on the release Mac.
It is not a CI job and it does not run on every push.

## What the updater installs

The app polls `https://api.github.com/repos/maxkongerskov/ollama/releases/latest`
3 seconds after launch and then every hour. Drafts and prereleases are never
"latest", so they are ignored.

- **Asset:** the first `.zip` whose name contains `darwin` and the Mac's arch
  (`arm64`/`aarch64` or `amd64`/`x86_64`) and that has a sibling
  `<name>.ed25519` asset. If none matches, any `.zip` that does not name
  another arch (e.g. `-darwin-universal.zip`) and that has the same sidecar.
  A zip with no signature is skipped, so an unsigned arch-specific zip falls
  through to a signed universal zip. A DMG is never installed by the updater,
  including the stable `Ollama.dmg` asset. A release with only a DMG, or with
  no signed zip, is skipped.
- **Zip layout:** `Ollama.app/` at the root and nothing else. Make it with
  `ditto -c -k --norsrc --keepParent Ollama.app X.zip`. Do not use
  `--sequesterRsrc`: its `__MACOSX/` entries make the updater reject the zip.
- **Signature:** the zip must have a sibling `<zipname>.ed25519` signature.
  That file is standard base64 of an ed25519 signature over the raw SHA-256
  of the zip. The public key is compiled into `app/updatetrust/public_key.go`.
  The private key stays on the release Mac (`~/.config/ollama-fork/update-private.pem`,
  or `OLLAMA_UPDATE_KEY`) and is never committed. An empty public key, a
  missing sidecar, or a tampered zip fails closed. After that check, the
  extracted `Ollama.app` must pass the Security framework's static code check
  (same as `codesign --verify --deep --strict`). Our builds are signed with
  Developer ID team `6LZ2DS9JPD`. Notarization isn't needed for the in-app
  update, because the app writes the files itself and they are not
  quarantined. A DMG that people download still needs notarization for
  Gatekeeper.
- **Version:** the app's version is `git describe` output, e.g.
  `0.40.5-b11724-ds41.3-0-gabcdef0`. The updater removes the
  `-N-gSHA[-dirty]` suffix and compares the rest with the release tag minus `v`:
  - same tag: no update, so an installed release is never offered again;
  - higher `X.Y.Z`, or the same `X.Y.Z` with a higher `bNNNNN`: update;
  - same stem with a higher trailing number (`ds41.3` → `ds41.4`, including
    `ds41.10` after `ds41.9`): update, even if the release was published
    before this build was installed;
  - lower: never offered (no downgrades);
  - same numbers, different non-numeric suffix (`-glm5next` →
    `-glm5next-keepalive`): offered only if the release was published after
    the running app was installed.

  The app has to be built from exactly the release tag. If it isn't, the
  updated app reports a different tag and keeps being offered the same
  release. `scripts/release-fork.sh` enforces this.

## Cutting an update

Use `bash scripts/release-fork.sh patch` on the release Mac. Setup and the
exact command are in [RELEASING.md](../RELEASING.md). The steps below are the
lower-level script that command replaces.

```sh
# 1. Commit, tag, push (new tag = new suffix and/or bumped version)
git tag v0.40.5-b11724-ds41.4
git push origin main v0.40.5-b11724-ds41.4

# 2. Build a signed Ollama.app from that tag (clean tree, or the version gets -dirty)
git checkout v0.40.5-b11724-ds41.4
APPLE_IDENTITY="Developer ID Application: Max Køngerskov (6LZ2DS9JPD)" \
  ./scripts/build_darwin.sh -a arm64          # -> dist/Ollama.app
#    or let the release script call build_darwin.sh: add --build in step 3

# 3. Check, then publish the zip (+ optional DMG) to the fork's release.
#    A DMG is uploaded twice: the versioned name, and Ollama.dmg so
#    https://github.com/maxkongerskov/ollama/releases/latest/download/Ollama.dmg
#    keeps pointing at the newest build.
./scripts/release_fork_darwin.sh --dry-run
./scripts/release_fork_darwin.sh [--dmg dist/Ollama.dmg]
```

The release script:

- refuses to run unless `origin` is `maxkongerskov/ollama`, and always calls
  `gh` with `--repo maxkongerskov/ollama`;
- checks the app's version matches the tag, its signature, and team ID;
- notarizes and staples it when credentials are configured
  (`--notary-profile NAME`/`NOTARY_PROFILE`, or
  `APPLE_ID`/`APPLE_PASSWORD`/`APPLE_TEAM_ID`). Without them it skips this step;
- writes `dist/fork-release/<tag>/Ollama-<version>-darwin-<arch>.zip`;
- when a DMG is provided, also writes that zip's sibling
  `Ollama-<version>-darwin-<arch>.dmg` and a byte-identical `Ollama.dmg`
  (the name the README downloads; the updater still installs the zip);
- signs the zip (and a DMG, when present) with the ed25519 release key and uploads the `.ed25519` sidecar;
- verifies the zip with `codesign` and the updater's own `VerifyDownload`;
- creates the release with `--latest`, or uploads to the existing one with
  `--clobber`, and adds `sha256sum.txt`.

Running apps pick up the update at their next hourly check, or 3 seconds
after a restart. With auto-update on, the zip is downloaded to
`~/Library/Caches/ollama/updates/<etag-hash>/` and the menu bar shows
"update available". The update is installed when you choose it there or on
the next app launch.
