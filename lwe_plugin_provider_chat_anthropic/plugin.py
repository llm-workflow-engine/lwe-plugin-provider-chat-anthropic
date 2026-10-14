"""Anthropic chat provider and thinking-response formatting."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from pydantic import Field
from typing_extensions import override

from lwe.core.config import Config
from lwe.core.provider import PresetValue, Provider

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"


class CustomChatAnthropic(ChatAnthropic):
    """Expose Anthropic model identification for LWE presets."""

    model: str = Field(alias="model_name", default=ANTHROPIC_DEFAULT_MODEL)
    """Model name to use."""

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_anthropic"


class ProviderChatAnthropic(Provider):
    """
    Access to chat Anthropic models
    """

    def __init__(self, config: Config | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.client: Anthropic = Anthropic()

    @property
    @override
    def model_property_name(self) -> str:
        return "model"

    @property
    @override
    def capabilities(self) -> dict[str, bool]:
        return {
            "chat": True,
            "validate_models": True,
        }

    @property
    @override
    def default_model(self) -> str:
        return ANTHROPIC_DEFAULT_MODEL

    def fetch_models(self) -> dict[str, dict[str, int | None]]:
        try:
            model_data = self.client.models.list()
            if not model_data:
                raise ValueError("Could not retrieve models")
            models = {
                model.id: {"max_tokens": model.max_input_tokens}
                for model in model_data
                if model.type == "model"
            }
            return models
        except Exception as e:
            raise ValueError(f"Could not retrieve models: {e}")

    @override
    def prepare_messages_method(
        self,
    ) -> Callable[[list[dict[str, Any]]], list[BaseMessage]]:
        return self.prepare_messages_for_llm_chat

    @override
    def llm_factory(self) -> type[CustomChatAnthropic]:
        return CustomChatAnthropic

    @override
    def customization_config(self) -> dict[str, Any]:
        return {
            "model": PresetValue(str, options=self.available_models),
            "max_tokens": PresetValue(int, min_value=1, include_none=True),
            "temperature": PresetValue(float, min_value=0.0, max_value=1.0),
            "top_k": PresetValue(int, min_value=1, max_value=40),
            "top_p": PresetValue(float, min_value=0.0, max_value=1.0),
            "default_request_timeout": PresetValue(float, min_value=1.0),
            "anthropic_api_key": PresetValue(str, include_none=True),
            "anthropic_api_url": PresetValue(str, include_none=True),
            "tools": None,
            "tool_choice": None,
            "model_kwargs": {
                "metadata": dict,
                "stop_sequences": PresetValue(str, include_none=True),
            },
            "thinking": {
                "type": PresetValue(
                    str, options=["enabled", "adaptive"], include_none=True
                ),
                "budget_tokens": PresetValue(int, min_value=1, include_none=True),
            },
            "output_config": {
                "effort": PresetValue(
                    str,
                    options=["low", "medium", "high", "xhigh", "max"],
                    include_none=True,
                ),
                "format": dict,
            },
            "mcp_servers": dict,
        }

    def format_thinking_content(
        self,
        content: str | Sequence[str | Mapping[str, object]],
        mode: str = "",
    ) -> str:
        """Render text and thinking blocks; preserve plain string responses."""
        if isinstance(content, str):
            return content
        output = ""
        for part in content:
            if isinstance(part, str):
                part = {"text": part}
            thinking = part.get("thinking")
            text = part.get("text")
            if isinstance(thinking, str):
                if mode != "thinking":
                    mode = "thinking"
                    output += "[THINKING]\n\n"
                output += thinking
            elif isinstance(text, str):
                if mode != "text":
                    mode = "text"
                    output += "\n\n[ANSWER]\n\n"
                output += text
        return output

    def handle_non_streaming_response(self, response: AIMessage) -> AIMessage:
        if isinstance(response.content, list):
            response.content = self.format_thinking_content(response.content)
        return response

    def handle_streaming_chunk(
        self,
        chunk: AIMessageChunk | str,
        previous_chunks: list[AIMessageChunk | str],
    ) -> str:
        if isinstance(chunk, str):
            return chunk
        elif isinstance(chunk.content, str):
            return chunk.content
        return self.handle_streaming_thinking_chunk(chunk, previous_chunks)

    def get_last_streaming_mode(
        self, previous_chunks: list[AIMessageChunk | str]
    ) -> str:
        for chunk in reversed(previous_chunks):
            content = chunk if isinstance(chunk, str) else chunk.content
            if isinstance(content, str):
                if content:
                    return "text"
                continue
            for part in reversed(content):
                if isinstance(part, str):
                    if part:
                        return "text"
                    continue
                part_type = part.get("type")
                if isinstance(part_type, str) and part_type in (
                    "thinking",
                    "text",
                ):
                    return part_type
        return "text"

    def handle_streaming_thinking_chunk(
        self,
        chunk: AIMessageChunk,
        previous_chunks: list[AIMessageChunk | str],
    ) -> str:
        mode = self.get_last_streaming_mode(previous_chunks)
        output = self.format_thinking_content(chunk.content, mode)
        return output

    # NOTE: The Anthropic SDK removed client.count_tokens(), so this is
    #       disabled until another method can be implemented.
    # def get_num_tokens_from_messages(self, messages, encoding=None):
    #     """
    #     Get number of tokens for a list of messages.
    #
    #     :param messages: List of messages
    #     :type messages: list
    #     :param encoding: Encoding to use, currently ignored
    #     :type encoding: Encoding, optional
    #     :returns: Number of tokens
    #     :rtype: int
    #     """
    #     num_tokens = 0
    #     messages = util.transform_messages_to_chat_messages(messages)
    #     for message in messages:
    #         for value in message.values():
    #             if isinstance(value, dict) or isinstance(value, list):
    #                 value = json.dumps(value, indent=2)
    #             if value:
    #                 num_tokens += self.client.count_tokens(value)
    #     # TODO: Missing counting of tokens for tool calls
    #     # This should probably be removed when token counting
    #     # is improved.
    #     return num_tokens
