"""Code-defined chat template and protocol checks."""

from __future__ import annotations

import re
from typing import Iterable


class ChatTemplateError(ValueError):
    """Raised when a chat template does not match the declared protocol."""


CHAT_TEMPLATE = r"""{%- if messages and messages[0]['role'] == 'system' -%}
    {{- '<|taichu_start|>system\n' + messages[0]['content'] + '<|taichu_end|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- elif default_system_message is defined -%}
    {{- '<|taichu_start|>system\n' + default_system_message + '<|taichu_end|>\n' -}}
    {%- set loop_messages = messages -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if message.get('content') is none or message.content|length == 0 -%}
        {%- continue -%}
    {%- endif -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|taichu_start|>user\n' -}}
        {{- message['content'] | string -}}
        {{- '<|taichu_end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<|taichu_start|>assistant\n' -}}
        {%- if message.get('reasoning_content') and message['reasoning_content'] | trim -%}
            {{- '<|think|>' + message['reasoning_content'] + '<|/think|>\n' -}}
        {%- endif -%}
        {{- message['content'] | string -}}
        {%- if not (loop.last and add_generation_prompt) -%}
            {{- '<|taichu_end|>\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {%- if not (loop_messages and loop_messages[-1]['role'] == 'assistant') -%}
        {{- '<|taichu_start|>assistant\n' -}}
    {%- endif -%}
{%- endif -%}
"""


def validate_template_tokens(template: str, ordered_tokens: Iterable[str]) -> None:
    declared = set(ordered_tokens)
    referenced = set(re.findall(r"<\|[^|\n>]+\|>", template))
    undeclared = referenced.difference(declared)
    if undeclared:
        raise ChatTemplateError(
            f"Chat Template 引用了未注册 Token: {sorted(undeclared)}"
        )
