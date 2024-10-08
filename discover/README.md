# `discover`

This package is responsible for discovering information about the system and the capabilities to run LLM.  This includes GPU and CPU discovery so the optimal runner can be chosen for a given model.  The ollama scheduler relies on up-to-date available memory information, so this package provides the ability to refresh free memory as efficiently as possible.