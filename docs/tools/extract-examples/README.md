# extract-examples

Extracts code examples from MDX files to a temp directory so you can run them.

## Usage

```shell
go run docs/tools/extract-examples/main.go <mdx-file>
```

## Example

```shell
go run docs/tools/extract-examples/main.go docs/api/openai-compatibility.mdx
```

Output:

```
Extracting code examples to: /var/folders/vq/wfm2g6k917d3ldzpjdxc8ph00000gn/T/mdx-examples-3271754368

  - 01_basic.py
  - 01_basic.js
  - 01_basic.sh
  - 02_responses.py
  - 02_responses.js
  - 02_responses.sh
  - 03_vision.py
  - 03_vision.js
  - 03_vision.sh

Extracted 9 file(s) to /var/folders/vq/wfm2g6k917d3ldzpjdxc8ph00000gn/T/mdx-examples-3271754368

To run examples:

  cd /var/folders/vq/wfm2g6k917d3ldzpjdxc8ph00000gn/T/mdx-examples-3271754368
  npm install   # for JS examples

then run individual files with `node file.js`, `python file.py`, `bash file.sh`
```

## How it works

- Parses MDX files looking for fenced code blocks with filenames (e.g., ` ```python basic.py `)
- Groups examples by their `<CodeGroup>` and prefixes filenames with `01_`, `02_`, etc.
- Writes all extracted files to a temp directory
