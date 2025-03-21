#!/bin/bash
set -ex

docker run \
  --rm \
  --quiet \
  --user "$(id -u):$(id -g)" \
  --volume "./:/docs" \
  --name "build-docs" \
  squidfunk/mkdocs-material:9.6 build --strict

# Remove unnecessary build artifacts: https://github.com/squidfunk/mkdocs-material/issues/2519
# site/ is the build output folder.
cd site
find . -type f -name '*.min.js.map' -delete -o -name '*.min.css.map' -delete
rm sitemap.xml.gz
rm assets/images/favicon.png
rm -r assets/javascripts/lunr
