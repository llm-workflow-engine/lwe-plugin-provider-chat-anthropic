import json
from langchain.chat_models.anthropic import ChatAnthropic

from lwe.core.provider import Provider, PresetValue
from lwe.core import util

from anthropic import Anthropic


class CustomChatAnthropic(ChatAnthropic):

    # TODO: Remove this when https://github.com/langchain-ai/langchain/issues/10909 is fixed
    @property
    def _identifying_params(self):
        return self._default_params

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_anthropic"


class ProviderChatAnthropic(Provider):
    """
    Access to chat Anthropic models
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.client = Anthropic()

    @property
    def model_property_name(self):
        return "model"

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
            'models': {
                'claude-instant-1': {
                    'max_tokens': 102400,
                },
                'claude-instant-1.2': {
                    'max_tokens': 102400,
                },
                'claude-2': {
                    'max_tokens': 102400,
                },
                'claude-2.1': {
                    'max_tokens': 204800,
                },
            },
        }

    @property
    def default_model(self):
        return 'claude-2'

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatAnthropic

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'max_tokens_to_sample': PresetValue(int, min_value=1, include_none=True),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'default_request_timeout': PresetValue(float, min_value=1.0),
            'anthropic_api_key': PresetValue(str, include_none=True),
            'anthropic_api_url': PresetValue(str, include_none=True),
            "model_kwargs": {
                "metadata": dict,
                "stop_sequences": PresetValue(str, include_none=True),
            },
        }

    def get_num_tokens_from_messages(self, messages, encoding=None):
        """
        Get number of tokens for a list of messages.

        :param messages: List of messages
        :type messages: list
        :param encoding: Encoding to use, currently ignored
        :type encoding: Encoding, optional
        :returns: Number of tokens
        :rtype: int
        """
        num_tokens = 0
        messages = util.transform_messages_to_chat_messages(messages)
        for message in messages:
            for value in message.values():
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                num_tokens += self.client.count_tokens(value)
        return num_tokens
