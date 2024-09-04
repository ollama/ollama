# Contributing to Ollama

Thank you for your interest in contributing to Ollama! Here are a few guidelines to help get you started.

## Set up

See the [development documentation](./docs/development.md) for instructions on how to build and run Ollama locally.

## Pull requests

### Ideal issues

* [Bugs](https://github.com/ollama/ollama/issues?q=is%3Aissue+is%3Aopen+label%3Abug): issues where Ollama stops working or where it results in an unexpected error.
* [Performance](https://github.com/ollama/ollama/issues?q=is%3Aissue+is%3Aopen+label%3Aperformance): issues to make Ollama faster at model inference, downloading or uploading.
* [Security](https://github.com/ollama/ollama/blob/main/SECURITY.md): issues that could lead to a security vulnerability. As mentioned in [SECURITY.md](https://github.com/ollama/ollama/blob/main/SECURITY.md), please do not disclose security vulnerabilities publicly.

### Issues that are harder to review

* New features: new features (e.g. API fields, environment variables) add surface area to Ollama and make it harder to maintain in the long run as they cannot be removed without potentially breaking users in the future.
* Refactoring: large code improvements are important, but can be harder or take longer to review and merge.
* Documentation: small updates to fill in or correct missing documentation is helpful, however large documentation additions can be hard to maintain over time.

### Issues that may not be accepted

* Changes that break backwards compatibility in Ollama's API (including the OpenAI-compatible API)
* Changes that add significant friction to the user experience
* Changes that create a large future maintenance burden for maintainers and contributors

### Best practices

* Commit messages: please leave both a title and a description in your commit messages. The title should be a short summary of the changes, with a leading word that explains the section of the code being changed (e.g. `api: fix parsing of prompt field`) . In the description, leave a short 2-3 sentences that explain more about the change and its impact.
* Tests: please add test coverage to changes where possible.
* Minimize dependencies: avoid adding new dependencies unless absolutely necessary.

## Need help?

If you need help with anything, feel free to reach out to us on our [Discord server](https://discord.gg/ollama).
