# How to troubleshoot issues

Sometimes Ollama may not perform as expected. One of the best ways to figure out what happened is to take a look at the logs. Find the logs on Mac by running the command:

```shell
cat ~/.ollama/logs/server.log
```

On Linux systems with systemd, the logs can be found with this command:

```shell
journalctl -u ollama
```

If manually running `ollama serve` in a terminal, the logs will be on that terminal.

Join the [Discord](https://discord.gg/ollama) for help interpreting the logs.

## Known issues


* `signal: illegal instruction (core dumped)`: Ollama requires AVX support from the CPU. This was introduced in 2011 and CPUs started offering it in 2012. CPUs from before that and some lower end CPUs after that may not have AVX support and thus are not supported by Ollama. Some users have had luck with building Ollama on their machines disabling the need for AVX.
