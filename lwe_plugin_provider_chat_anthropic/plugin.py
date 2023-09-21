from langchain.chat_models.anthropic import ChatAnthropic

from lwe.core.provider import Provider, PresetValue


class CustomChatAnthropic(ChatAnthropic):

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_anthropic"


class ProviderChatAnthropic(Provider):
    """
    Access to chat Anthropic models
    """

    @property
    def model_property_name(self):
        return "model"

    @property
    def capabilities(self):
        return {
            'validate_models': True,
            'models': {
                'claude-instant-1': {
                    'max_tokens': 100000,
                },
                'claude-2': {
                    'max_tokens': 100000,
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
