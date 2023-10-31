# Bash Shell examples

When calling `ollama`, you can pass it a file to run all the prompts in the file, one after the other. This concept is used in two examples

## Bulk Questions
`bulkquestions.sh` is a script that runs all the questions in `sourcequestions` using the llama2 model and outputs the answers.

## Compare Models
`comparemodels.sh` is a script that runs all the questions in `sourcequestions` using any 4 models you choose that you have already pulled from the registry or have created locally.