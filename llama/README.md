# `llama`

This package provides Go bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Vendoring

Ollama vendors [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src). While we generally strive to contribute changes back upstream to avoid drift, we carry a small set of patches which are applied to the tracking commit.

Run `make -f Makefile.sync help` to see available targets and workflows.

### Updating Base Commit

To update to a new upstream base commit:

1. Reset the vendor tree and apply current patches on the **old** base:
   ```shell
   make -f Makefile.sync clean
   make -f Makefile.sync apply-patches
   ```

2. Update `FETCH_HEAD` in `Makefile.sync` to the new commit hash.

3. Rebase the patches onto the new base:
   ```shell
   make -f Makefile.sync rebase-patches
   ```

4. If there are conflicts, resolve them in `./vendor/`, stage with `git -C llama/vendor add <file>`, then continue with `git -C llama/vendor rebase --continue`.

5. Save updated patches, sync to build dirs, and clean up:
   ```shell
   make -f Makefile.sync format-patches
   make -f Makefile.sync sync
   make -f Makefile.sync clean
   ```

### Generating Patches

When working on new fixes or features that impact vendored code, first get a clean tracking repo with all current patches applied:

```shell
make -f Makefile.sync clean
make -f Makefile.sync apply-patches
```

Make your changes and commit them in the `./vendor/` directory, then regenerate patches and sync:

```shell
make -f Makefile.sync format-patches
make -f Makefile.sync sync
```

In your `./vendor/` directory, create a branch, cherry-pick the new commit, and submit a PR upstream to llama.cpp.

Commit the changes in the ollama repo and submit a PR to Ollama, which will include the vendored code update with your change, along with the patches.

After your PR upstream is merged, follow the **Updating Base Commit** instructions above, but first remove your patch before step 1 since the new base commit contains your change already.
