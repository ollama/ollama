# `llama`

This package provides Go bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Vendoring

Ollama vendors [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src). While we generally strive to contribute changes back upstream to avoid drift, we carry a small set of patches which are applied to the tracking commit.

If you update the vendoring code, start by running the following command to establish the tracking llama.cpp repo in the `./vendor/` directory.

```shell
make -f Makefile.sync apply-patches
```

### Updating Base Commit

To update to a new base commit:

1. **Update FETCH_HEAD** in `Makefile.sync` to the new commit hash.

2. **Check for upstreamed patches**: Before applying, review if any patches have been merged upstream. Remove those patches from `./patches/` to avoid conflicts.

3. **Apply patches**:
   ```shell
   make -f Makefile.sync apply-patches
   ```

4. **Resolve conflicts** (if any): When `git am` fails on a patch:
   - Fix conflicts in `./vendor/`
   - Stage the resolved files: `git -C llama/vendor add <file>`
   - Continue: `git -C llama/vendor am --continue`
   - Re-run: `make -f Makefile.sync apply-patches`
   - Repeat until all patches are applied.

5. **Regenerate patches and sync**:
   ```shell
   make -f Makefile.sync format-patches sync
   ```

### Generating Patches

When working on new fixes or features that impact vendored code, use the following model. First get a clean tracking repo with all current patches applied:

```shell
make -f Makefile.sync clean apply-patches
```

Iterate until you're ready to submit PRs. Once your code is ready, commit a change in the `./vendor/` directory, then generate the patches for ollama with

```shell
make -f Makefile.sync format-patches
```

In your `./vendor/` directory, create a branch, and cherry-pick the new commit to that branch, then submit a PR upstream to llama.cpp.

Commit the changes in the ollama repo and submit a PR to Ollama, which will include the vendored code update with your change, along with the patches.

After your PR upstream is merged, follow the **Updating Base Commit** instructions above, however first remove your patch before running `apply-patches` since the new base commit contains your change already.
