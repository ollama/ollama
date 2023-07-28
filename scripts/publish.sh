# Set your variables here.
REPO="jmorganca/ollama"

# Check if VERSION is set
if [[ -z "${VERSION}" ]]; then
  echo "VERSION is not set. Please set the VERSION environment variable."
  exit 1
fi

OS=$(go env GOOS)

./script/build_${OS}.sh

# Create a new tag if it doesn't exist.
if ! git rev-parse v$VERSION >/dev/null 2>&1; then
  git tag v$VERSION
fi

git push origin v$VERSION

# Create a new release.
gh release create -p v$VERSION -t v$VERSION

# Upload the zip file.
gh release upload v$VERSION ./dist/* --clobber
