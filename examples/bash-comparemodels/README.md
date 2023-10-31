# Bash Shell examples

![Example Gif](https://uce9aa94fbc06b088ca05a92fe37.previews.dropboxusercontent.com/p/thumb/ACF0g3yrdu5-tvuw59Wil8B5bwuLkvWFFQNrYJzEkqvJnv4WfyuqcTGfXhXDfqfbemi5jGr9-bccO8r5VZxXrAeU1l_Plq99HCqV6b10thwwlaQCNbkXkw4YSF0YlYu-wu5A6Vn2SlrdcfiwTl6et-m7CPYx8ad2jSZXcPEozDUqXqB-f_zZNskASYzWwQko9n6UjMKx6qt54FYvIiW6n3ZiNVlM0GGt91FAA2Y0zD23aBlOlIAN8wH7qLznS2rZsn1n_7ukJMwegcEVud_XNPbG8Hn_13NtwkVsf4uWThknUpslNRmxWisqlRCaxZY71Me9wz3puH3nlpxtNlwoNAvQcXf0S4u_r1WLx22KwWqmvYFU41X2j_1Kum8amUrAv_5WVnOL6ctWnrbV4fauYfT9ClwgmLAtLoHwaQSXo2R2Kut_QIAkFIDAyMj9Fe9Ifj0/p.gif)

When calling `ollama`, you can pass it a file to run all the prompts in the file, one after the other. This concept is used in two examples

## Bulk Questions
`bulkquestions.sh` is a script that runs all the questions in `sourcequestions` using the llama2 model and outputs the answers.

## Compare Models
`comparemodels.sh` is a script that runs all the questions in `sourcequestions` using any 4 models you choose that you have already pulled from the Ollama library or have created locally.
