#!/usr/bin/env python3
"""Evaluate the product's first-chat experience against a running server.

This complements ``run_eval.py``'s academic, coding, and 30-case tool suites.
It deliberately uses short prompts that a non-technical user is likely to try
in their first session and records the raw responses for human review.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder at a specified time",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["text", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def request(
    base_url: str, model: str, messages: list[dict], *, tools=None, max_tokens=256
) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }
    if tools:
        body["tools"] = tools
    start = time.perf_counter()
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=180
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
        "elapsed_s": round(time.perf_counter() - start, 3),
    }


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def excessive_repetition(text: str) -> bool:
    words = re.findall(r"\w+", normalized(text))
    if len(words) < 24:
        return False
    trigrams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    return bool(trigrams and max(trigrams.values()) >= 4)


def run_case(base_url: str, model: str, case: dict) -> dict:
    result = request(
        base_url,
        model,
        case["messages"],
        tools=TOOLS if case.get("tools") else None,
        max_tokens=case.get("max_tokens", 256),
    )
    text = normalized(result["content"])
    checks: dict[str, bool] = {
        "nonempty": bool(text or result["tool_calls"]),
        "terminated": result["finish_reason"] != "length",
        "no_excessive_repetition": not excessive_repetition(result["content"]),
    }

    kind = case["kind"]
    if kind == "keywords":
        checks["required_content"] = all(
            any(option.lower() in text for option in group)
            for group in case["keyword_groups"]
        )
    elif kind == "contains":
        checks["required_content"] = case["value"].lower() in text
    elif kind == "uncertainty":
        uncertainty = ["无法确定", "不知道", "尚未", "还没有", "不能预测", "未来", "not known"]
        checks["does_not_invent_future_fact"] = any(token in text for token in uncertainty)
    elif kind == "json":
        try:
            candidate = result["content"].strip()
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
            parsed = json.loads(candidate)
            checks["exact_json"] = parsed == case["value"]
        except (json.JSONDecodeError, TypeError):
            checks["exact_json"] = False
    elif kind == "length":
        checks["within_length"] = len(result["content"].strip()) <= case["max_chars"]
    elif kind == "tool":
        calls = result["tool_calls"]
        checks["correct_tool"] = bool(calls) and calls[0].get("function", {}).get(
            "name"
        ) == case["tool"]
        if checks["correct_tool"]:
            try:
                json.loads(calls[0]["function"].get("arguments", ""))
                checks["valid_tool_arguments"] = True
            except (json.JSONDecodeError, KeyError, TypeError):
                checks["valid_tool_arguments"] = False
    elif kind == "no_tool":
        checks["did_not_call_tool"] = not result["tool_calls"]
        checks["asked_for_missing_detail"] = any(
            token in text for token in case.get("prompt_tokens", [])
        )

    result["id"] = case["id"]
    result["description"] = case["description"]
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


CASES = [
    {
        "id": "first01",
        "description": "Friendly concise first greeting",
        "kind": "keywords",
        "messages": [{"role": "user", "content": "你好！你能帮我做什么？请用三句话以内回答。"}],
        "keyword_groups": [["帮", "可以"], ["问题", "写", "总结", "分析"]],
        "max_tokens": 120,
    },
    {
        "id": "first02",
        "description": "Explain a concept to a child in Chinese",
        "kind": "keywords",
        "messages": [{"role": "user", "content": "请像给8岁孩子一样，用一个生活中的比喻解释什么是电池。不要超过100字。"}],
        "keyword_groups": [["电", "能量"], ["像", "好比", "比如"]],
        "max_tokens": 160,
    },
    {
        "id": "first03",
        "description": "Multi-turn constraint and reference retention",
        "kind": "keywords",
        "messages": [
            {"role": "user", "content": "给我三条提高睡眠质量的建议，每条不超过12个字。"},
            {"role": "assistant", "content": "1. 固定作息时间\n2. 睡前远离屏幕\n3. 下午避免咖啡"},
            {"role": "user", "content": "只把第二条改成更容易做到的建议，其他两条原样保留。"},
        ],
        "keyword_groups": [["固定作息时间"], ["下午避免咖啡"], ["2.", "二"]],
    },
    {
        "id": "first04",
        "description": "Faithful meeting-note summary",
        "kind": "keywords",
        "messages": [{"role": "user", "content": "把下面内容总结成三条要点，不要添加原文没有的信息：产品评审改到周二14:00，在305会议室。小林负责带蓝色原型机，阿杰周一18:00前发测试报告。"}],
        "keyword_groups": [["周二"], ["14:00", "14点"], ["305"], ["蓝色"], ["周一", "18:00", "18点"]],
    },
    {
        "id": "first05",
        "description": "Polite rewrite preserves intent",
        "kind": "keywords",
        "messages": [{"role": "user", "content": "把这句话改得礼貌但明确：‘你怎么还没交？今天下班前必须给我。’"}],
        "keyword_groups": [["今天", "下班前"], ["请", "麻烦", "能否"]],
    },
    {
        "id": "first06",
        "description": "Simple arithmetic word problem",
        "kind": "contains",
        "messages": [{"role": "user", "content": "我买了3杯每杯12元的咖啡，又买了10元的面包，一共多少钱？只给答案。"}],
        "value": "46",
        "max_tokens": 40,
    },
    {
        "id": "first07",
        "description": "Correct a false premise instead of agreeing",
        "kind": "keywords",
        "messages": [{"role": "user", "content": "埃菲尔铁塔为什么建在罗马？"}],
        "keyword_groups": [["巴黎"], ["不在罗马", "不是在罗马", "并非"]],
    },
    {
        "id": "first08",
        "description": "Do not invent a future event",
        "kind": "uncertainty",
        "messages": [{"role": "user", "content": "2030年世界杯冠军是谁？请直接告诉我。"}],
    },
    {
        "id": "first09",
        "description": "Exact structured-output instruction",
        "kind": "json",
        "messages": [{"role": "user", "content": "只输出合法JSON，不要Markdown：姓名小王，年龄28岁，字段必须是name和age。"}],
        "value": {"name": "小王", "age": 28},
        "max_tokens": 80,
    },
    {
        "id": "first10",
        "description": "Concise answer terminates without looping",
        "kind": "length",
        "messages": [{"role": "user", "content": "用不超过80个汉字说明为什么天空通常是蓝色的。不要重复。"}],
        "max_chars": 80,
        "max_tokens": 180,
    },
    {
        "id": "first11",
        "description": "Uses web search for current weather",
        "kind": "tool",
        "tools": True,
        "tool": "web_search",
        "messages": [{"role": "user", "content": "帮我查一下东京现在的天气。"}],
    },
    {
        "id": "first12",
        "description": "Does not call a tool for ordinary conversation",
        "kind": "no_tool",
        "tools": True,
        "messages": [{"role": "user", "content": "提醒我给妈妈打电话。"}],
        "prompt_tokens": ["时间", "什么时候", "几点"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for case in CASES:
        print(f"{case['id']}: {case['description']} ... ", end="", flush=True)
        try:
            result = run_case(args.base_url, args.model, case)
        except Exception as exc:  # preserve the failure as benchmark evidence
            result = {
                "id": case["id"],
                "description": case["description"],
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print("PASS" if result["passed"] else "FAIL")

    passed = sum(bool(item["passed"]) for item in results)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "artifact": args.artifact,
        "settings": {"temperature": 0.0, "enable_thinking": False},
        "passed": passed,
        "total": len(results),
        "score": passed / len(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Starter experience: {passed}/{len(results)}; wrote {output}")


if __name__ == "__main__":
    main()
