# Known MoltenVK release versions and their SHA256 checksums.
# Used when OLLAMA_FETCH_MOLTENVK=ON and the default MoltenVK URL is used.
# Format: version sha256 (pairs)
#
# To add a new version, download the MoltenVK-macos.tar from the KhronosGroup
# release page and compute the SHA256:
#   shasum -a 256 MoltenVK-macos.tar
set(OLLAMA_MOLTENVK_RELEASE_SHA256_MAP
    1.4.1    5ea0c259df7ded9a275444820f09cced54d6e5a7c7a31d262de62a5cdb7e15cf
)
