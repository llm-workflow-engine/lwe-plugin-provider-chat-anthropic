import json
from langchain_anthropic import ChatAnthropic
from pydantic import Field

from lwe.core.provider import Provider, PresetValue
from lwe.core import util

from anthropic import Anthropic

ANTHROPIC_DEFAULT_MODEL = 'claude-2.1'


class CustomChatAnthropic(ChatAnthropic):

    model: str = Field(alias="model_name", default=ANTHROPIC_DEFAULT_MODEL)
    """Model name to use."""

    @property
    def _identifying_params(self):
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
        }

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_anthropic"


class ProviderChatAnthropic(Provider):
    """
    Access to chat Anthropic models
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.client = Anthropic()

    @property
    def model_property_name(self):
        return "model"

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
        }

    @property
    def default_model(self):
        return ANTHROPIC_DEFAULT_MODEL

    @property
    def static_models(self):
        return {
            'claude-instant-1.2': {
                'max_tokens': 102400,
            },
            'claude-2': {
                'max_tokens': 102400,
            },
            'claude-2.1': {
                'max_tokens': 204800,
            },
            'claude-3-opus-20240229': {
                'max_tokens': 204800,
            },
            'claude-3-sonnet-20240229': {
                'max_tokens': 204800,
            },
            'claude-3-haiku-20240307': {
                'max_tokens': 204800,
            },
            'claude-3-5-haiku-20241022': {
                'max_tokens': 204800,
            },
            'claude-3-5-haiku-latest': {
                'max_tokens': 204800,
            },
            'claude-3-5-sonnet-20240620': {
                'max_tokens': 204800,
            },
            'claude-3-5-sonnet-20241022': {
                'max_tokens': 204800,
            },
            'claude-3-5-sonnet-latest': {
                'max_tokens': 204800,
            },
        }

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatAnthropic

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'max_tokens': PresetValue(int, min_value=1, include_none=True),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'default_request_timeout': PresetValue(float, min_value=1.0),
            'anthropic_api_key': PresetValue(str, include_none=True),
            'anthropic_api_url': PresetValue(str, include_none=True),
            "tools": None,
            "tool_choice": None,
            "model_kwargs": {
                "metadata": dict,
                "stop_sequences": PresetValue(str, include_none=True),
            },
        }

    # NOTE: The Anthropic SDK removed client.count_tokens(), so this is disabled
    #       until another method can be implemented.
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
