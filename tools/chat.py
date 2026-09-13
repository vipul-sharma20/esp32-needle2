"""Needle chat template, matching `render_example` in the training repo."""

import json

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_START, THINK_END = "<think>", "</think>"
TOOLS_START, TOOLS_END = "<tools>", "</tools>"
TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"
TOOL_RESULT_START, TOOL_RESULT_END = "<tool_result>", "</tool_result>"


def render_prompt(query, tools=None, system=None):
    tools_json = (tools if isinstance(tools, str)
                  else json.dumps(tools or [], separators=(",", ":"), ensure_ascii=False))
    system = (system or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    return (prefix + IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
            + query + IM_END + "\n" + IM_START + "assistant\n")
