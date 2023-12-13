# How to troubleshoot issues

Sometimes Ollama may not perform as you expect. One of the best ways to figure out what just happened is to take a look at the logs. You can find the logs on Mac by running the command:

```shell
cat ~/.ollama/logs/server.log
```

On Linux, the logs can be found with this command:

```shell
journalctl -u ollama
```

If you need help interpreting what you see, try signing in to the Discord at [discord.gg/ollama](https://discord.gg/ollama).

## Error Index

This is a list of some of the errors that can pop up and their solutions:

**signal: illegal instruction (core dumped).** - Ollama requires AVX support from the CPU. This was introduced in 2011 and CPUs started offering it in 2012. CPUs from before that and some lower end CPUs after that may not have AVX support and thus are not supported by Ollama. Some users have had luck with building Ollama on their machines disabling the need for AVX.

