# proto

🙊

## Running

Install dependencies:

```
pip install -r requirements.txt
```

Put your model in `models/` and run:

```
python proto.py
```

To run the app:

```
cd desktop
npm install
npm start
```

## Building

If using Apple silicon, you need a Python version that supports arm64:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Get the dependencies:

```bash
pip install -r requirements.txt
```

Then build a binary for your current platform:

```bash
python3 build.py
```

### Building the app

```
cd desktop
npm run package
```

## API

### `POST /generate`

model: `string` - The name of the model to use in the `models` folder.
prompt: `string` - The prompt to use.
