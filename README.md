# LLM Workflow Engine (LWE) Chat Anthropic Provider plugin

Chat Anthropic Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine)

Access to [Anthropic](https://docs.anthropic.com/claude/reference/selecting-a-model) chat models.

## Installation

### Export API key

Grab a Anthropic API key from [https://console.anthropic.com/account/keys](https://console.anthropic.com/account/keys)

Export the key into your local environment:

```bash
export ANTHROPIC_API_KEY=<API_KEY>
```

### From packages

Install the latest version of this software directly from github with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-anthropic
```

### From source (recommended for development)

Install the latest version of this software directly from git:

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-anthropic.git
```

Install the development package:

```bash
cd lwe-plugin-provider-chat-anthropic
pip install -e .
```

## Configuration

Add the following to `config.yaml` in your profile:

```yaml
plugins:
  enabled:
    - provider_chat_anthropic
    # Any other plugins you want enabled...
```

## Usage

From a running LWE shell:

```
/provider chat_anthropic
/model model claude-instant-1
```
