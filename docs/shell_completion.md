# Shell (bash) Autocomplete for Ollama

This document provides a guide for implementing bash autocompletion for Ollama model names. While this document currently only focuses on bash, contributions for other shells are welcome.

## Bash autocompletion script:

```sh
#!/bin/bash
# Bash completion for the "ollama" CLI.  Requires jq and curl.
#
# Published: 
# * https://github.com/CLIAI/handy_scripts/blob/main/bash_completion/etc/bash_completion.d/ollama

# Cache settings
_OLLAMA_MODEL_TTL=300
_OLLAMA_MODELS_TIMESTAMP=0
_OLLAMA_MODELS=""

# Fetch models from Ollama server, caching results
_ollama_fetch_models() {
  local now
  now=$(date +%s)
  if [ $(( now - _OLLAMA_MODELS_TIMESTAMP )) -gt $_OLLAMA_MODEL_TTL ]; then
    _OLLAMA_MODELS=$(
      curl -s http://localhost:11434/api/tags \
      | jq -r '.models[].name'
    )
    _OLLAMA_MODELS_TIMESTAMP=$now
  fi
}

# Main completion function
_ollama() {
  local cur prev
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"

  case "${prev}" in
    run)
      # Get fresh models from Ollama
      _ollama_fetch_models

      # Filter models by whatever the user has typed so far (case-insensitive)
      # and feed them into compgen so they appear as completions.
      local filtered
      filtered=$(echo "$_OLLAMA_MODELS" | grep -i "$cur")
      COMPREPLY=( $(compgen -W "${filtered}" -- "$cur") )
      return 0
      ;;
  esac

  # Default commands
  COMPREPLY=( $(compgen -W "serve create show run pull push list ps cp rm help" -- "$cur") )
}

# Register the completion function
complete -F _ollama ollama
```

## Installation Steps

1. **Create a Completion Script File**

   You can create the Ollama completion script in either of the following locations:

   * System-wide (requires sudo privileges):

     ```bash
     sudo $EDITOR /etc/bash_completion.d/ollama
     ```

   * User-specific (in your home directory):

     ```bash
     $EDITOR ~/.ollama-completion.sh
     ```

2. **Copy the Completion Script**

   Copy the bash autocompletion script from beginning of document into the file you created in step 1.

3. **Configure Bash to Use the Script**

   If you placed the script in your home directory, add the following line to your `.bashrc` file to ensure it is sourced:

   ```bash
   source ~/.ollama-completion.sh
   ```

4. **Install Required Dependencies**

   Ensure that `jq` and `curl` are installed on your system:

   ```bash
   sudo apt install jq curl
   ```

5. **Activate the Completion Script**

   Restart your terminal or source your `.bashrc` file to activate the completion script:

   ```bash
   source ~/.bashrc
   ```

By following these steps, you will enable bash autocompletion for the Ollama CLI, making it easier to work with model names and commands.
