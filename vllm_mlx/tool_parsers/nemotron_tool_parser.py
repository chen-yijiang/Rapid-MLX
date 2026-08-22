# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3 parser aliases the shared Qwen3 XML implementation.

Current vLLM models Nemotron 3 as the Qwen3 tool-call grammar, and SGLang
serves Nemotron 3 with its Qwen3-Coder detector. Keep the public Rapid-MLX
parser names while sharing the same implementation instead of maintaining a
second scanner for the identical wire format.
"""

from .abstract_tool_parser import ToolParserManager
from .qwen3coder_tool_parser import Qwen3CoderToolParser


@ToolParserManager.register_module(["nemotron", "nemotron3"])
class NemotronToolParser(Qwen3CoderToolParser):
    """Qwen3 XML parser exposed under the Nemotron-compatible names."""
