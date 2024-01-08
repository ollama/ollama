import asyncio
from logging import Logger
import logging
from typing import Any, List, Optional, Union

import aiohttp

from options import OllamaCompletionOptions


class Ollama:
    _api_version: str
    _base_url: str
    _logger: Logger

    def __init__(
        self,
        api_version: Optional[str] = None,
        base_url: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initialize an Ollama client.

        Arguments:
            
        """
        self._api_version = api_version or '0.1.18'
        self._base_url = base_url or 'http://0.0.0.0:11434'
        self._logger = logger or logging.getLogger(__name__)
    
    async def verify_running(self, model_id: str) -> Any:
        """
        Ensure that the Ollama service is running and that the model is ready.

        Arguments:
            model_id {str} -- The Ollama model name, see https://ollama.ai/library.
        """
        
        try:
            result = await self.show(model_id)
            self._logger.debug(result)
            return result
        except Exception:
            self._logger.exception("Failed to verify that the Ollama service is running.")
            raise

    async def show(self, model_id: str) -> Any:
        request = dict(name=model_id)
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base_url}/api/show", json=request) as r:
                result = await r.json()
                error = result.get('error')
                assert error is None, error
                return result

    async def complete_async(
        self,
        prompt: str,
        request_settings: OllamaCompletionOptions,
        logger: Optional[Logger] = None,
    ) -> Union[str, List[str]]:
        # TODO Support choices/number_of_responses.
        assert request_settings.number_of_responses == 1
        result = ""
        response = self._send_completion_request(prompt, request_settings, logger, stream=False)
        async for c in response:
            result += c
        return result


    async def complete_stream_async(
        self,
        prompt: str,
        request_settings: OllamaCompletionOptions,
        logger: Optional[Logger] = None,
    ):
        response = self._send_completion_request(prompt, request_settings, logger, stream=True)

        async for chunk in response:
            if request_settings.number_of_responses > 1:
                # TODO Support choices/number_of_responses.
                for choice in chunk.choices:
                    completions = [""] * request_settings.number_of_responses
                    completions[choice.index] = choice.text
                    yield completions
            else:
                yield chunk

    async def _send_completion_request(
        self,
        prompt: str,
        request_settings: OllamaCompletionOptions,
        logger: Optional[Logger] = None,
        stream: bool = False,
    ):
        """
        Completes the given prompt. Returns a single string completion.
        Cannot return multiple completions. Cannot return logprobs.

        Arguments:
            prompt {str} -- The prompt to complete.
            request_settings {CompleteRequestSettings} -- The request settings.

        Returns:
            str -- The completed text.
        """
        logger = logger or self._logger
        if self._api_version != '0.1.0':
            raise ValueError(f"Unsupported Ollama API version: {self._api_version}. Only 0.1.0 is supported.")
        if not prompt:
            raise ValueError("The prompt cannot be `None` or empty")
        if request_settings is None:
            raise ValueError("The request settings cannot be `None`")

        if request_settings.max_tokens < 1:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "The max tokens must be greater than 0, "
                f"but was {request_settings.max_tokens}",
            )

        if request_settings.logprobs != 0:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "complete_async does not support logprobs, "
                f"but logprobs={request_settings.logprobs} was requested",
            )

        # TODO Set up other custom parameters defined at https://github.com/jmorganca/ollama/blob/main/docs/api.md.
        request = dict(
            model=self._model_id,
            prompt=prompt,
            stream=stream,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{self._base_url}/api/generate', json=request) as r:
                    async for chunk in r.content:
                        token_info = json.loads(chunk.decode('utf-8'))
                        if token_info['done']:
                            # Can't use "%s" because we need to be compatible with `NullLogger`.
                            logger.debug(f"Ollama response: {token_info}")
                            break
                        token = token_info['response']
                        yield token
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "Ollama service failed to complete the prompt.",
                ex,
            )
