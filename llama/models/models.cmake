# Interim model-architecture support — CMake integration
#
# Mirrors llama/compat/compat.cmake. This directory holds interim, Ollama-owned
# implementations of model architectures not yet available in the pinned
# llama.cpp, plus a small per-architecture registration patch for each:
#
#   1. <arch>.cpp          - the architecture implementation, linked into the
#                            fetched llama target (kept in Ollama's tree so it
#                            never conflicts on a llama.cpp version bump).
#   2. llama-cpp-<arch>.patch - the registration hooks: the arch enum + name,
#                            the model factory and rope-type entries, and the
#                            model class declaration; plus, only when the model
#                            needs them, new tensor names, a tokenizer pre-type,
#                            or hparams fields. Applied to the fetched source.
#
# llama/server/CMakeLists.txt applies every *.patch here (after the compat
# hooks patch) and links every *.cpp here into the fetched llama target.
#
# See llama/models/README.md for how to add a new architecture.

set(_models_dir ${CMAKE_CURRENT_LIST_DIR})

# Directory holding registration patches (*.patch) and architecture sources
# (*.cpp). Exposed so llama/server/CMakeLists.txt can apply the patches and
# link the sources.
set(OLLAMA_LLAMA_CPP_MODELS_DIR
    "${_models_dir}"
    CACHE INTERNAL "Directory of Ollama llama.cpp architecture sources and patches")
