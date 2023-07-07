# Set your variables here.
REPO="jmorganca/ollama"

# Check if VERSION is set
if [[ -z "${VERSION}" ]]; then
  echo "VERSION is not set. Please set the VERSION environment variable."
  exit 1
fi

OS=$(go env GOOS)
ARCH=$(go env GOARCH)

make app

# Create a new tag if it doesn't exist.
if ! git rev-parse v$VERSION >/dev/null 2>&1; then
  git tag v$VERSION
  git push origin v$VERSION
fi

# Create a new release.
gh release create v$VERSION

# Upload the zip file.
gh release upload v$VERSION "app/out/make/zip/${OS}/${ARCH}/Ollama-${OS}-${ARCH}-${VERSION}.zip#Ollama-${OS}-${ARCH}.zip"

# Upload the binary.
gh release upload v$VERSION "./ollama#ollama-${OS}-${ARCH}"

