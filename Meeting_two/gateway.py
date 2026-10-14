"""
AI Committee Demo - Two-Model Gateway
====================================

A minimal OpenAI-compatible proxy that chains two vLLM instances:

    User -> Flask -> Qwen3Guard --Safe--> Qwen -> answer
                    --Unsafe/Controversial/error--> fixed denial message

Students chat in Open WebUI. This script sits between that browser interface
and the two models. Read chat_completions() first to see the whole workflow.

Run: python gateway.py
Open WebUI API URL: http://<gateway-host>:8003/v1
"""

import json
import random
import re
from urllib.request import urlopen

from flask import Flask, jsonify, request
from openai import OpenAI

# 1. Settings: the gateway connects to two separate vLLM servers.
# The model names must match vLLM's --served-model-name settings.
FILTER_BASE_URL = "http://127.0.0.1:8001/v1"
FILTER_MODEL = "qwen3-guard"

GENERAL_BASE_URL = "http://127.0.0.1:8000/v1"
GENERAL_MODEL = "committee-general"

# Students see this single name in Open WebUI, even though we use two models.
SERVED_MODEL = "filtered-model"

# Qwen3Guard checks the user's prompt. Our code supplies the refusal text.
denial_messages = [
    "Yeah not this time buddy, take that weird stuff elsewhere.",
    "Sure I'd love to help, ha just kidding get pranked.",
    "Take your prompt elsewhere I'm not doing allat.",
    "Nope not doing that my other AI model friend told me thats not allowed.",
    "Try that again and see what happens."
]

DENIAL_MESSAGE = random.choice(denial_messages)

# 2. Clients: "EMPTY" is a placeholder for local vLLM without authentication.
filter_client = OpenAI(base_url=FILTER_BASE_URL, api_key="EMPTY")
general_client = OpenAI(base_url=GENERAL_BASE_URL, api_key="EMPTY")

app = Flask(__name__)


def validate_model_servers():
    """Warn about unreachable servers or mismatched model IDs at startup."""
    for role, base_url, model in (
        ("Qwen3Guard", FILTER_BASE_URL, FILTER_MODEL),
        ("general", GENERAL_BASE_URL, GENERAL_MODEL),
    ):
        models_url = base_url.rstrip("/") + "/models"
        try:
            with urlopen(models_url, timeout=3) as response:
                advertised = [item["id"] for item in json.load(response)["data"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            app.logger.warning(
                "Cannot validate %s model %r at %s: %s",
                role, model, models_url, error,
            )
            continue
        if model not in advertised:
            app.logger.warning(
                "%s server at %s does not advertise expected model %r; "
                "advertised models: %s. Check the gateway constants "
                "and vLLM --served-model-name.",
                role, models_url, model, advertised,
            )


def judge_user_message(user_text: str) -> bool:
    """Ask Qwen3Guard about the prompt; only Safety: Safe allows Qwen."""
    print(f"[guard] input: {user_text!r}", flush=True)
    try:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("Guard input must be a nonempty text prompt")
        # vLLM applies Qwen3Guard's own template; no custom system prompt.
        reply = filter_client.chat.completions.create(
            model=FILTER_MODEL,
            messages=[{"role": "user", "content": user_text}],
            temperature=0, max_tokens=64,
        )
        raw = reply.choices[0].message.content if reply.choices else None
        print(f"[guard] raw: {raw!r}", flush=True)
        # Require one complete Safety field; missing/duplicate/unknown fields deny.
        fields = re.findall(r"^[ \t]*Safety:[ \t]*([^\r\n]*)", raw or "", re.I | re.M)
        verdict = fields[0].strip().upper() if len(fields) == 1 else "UNRECOGNIZED"
        if verdict not in ("SAFE", "UNSAFE", "CONTROVERSIAL"):
            verdict = "UNRECOGNIZED"
        print(f"[guard] verdict: {verdict}", flush=True)
        return verdict == "SAFE"
    except Exception as error:
        # An unavailable guard or malformed response must never bypass the gate.
        print(f"[guard] verdict: ERROR (denied: {error})", flush=True)
        return False


def format_response(text: str, finish_reason: str = "stop") -> dict:
    """Put our answer in the JSON format Open WebUI expects."""
    # Both refusals and Qwen answers use this response format.
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": 0,
        "model": SERVED_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def chat_json(payload: dict, status: int = 200):
    """Return normal JSON and log the exact body sent to the client."""
    response = jsonify(payload)
    response.status_code = status
    print(f"[gateway] Flask JSON: {response.get_data(as_text=True)!r}", flush=True)
    return response


# 3. Chat workflow: this is the endpoint Open WebUI calls.
@app.post("/v1/chat/completions")
def chat_completions():
    """Check the latest user prompt, then either refuse or forward to Qwen."""
    # Step 1: Read the conversation sent by the browser.
    data = request.get_json(force=True)
    messages = data.get("messages", [])

    # Step 2: Check the latest user prompt; deny unless the verdict is Safe.
    user_text = next(
        (message["content"] for message in reversed(messages)
         if message.get("role") == "user"), ""
    )
    if not judge_user_message(user_text):
        print("[gateway] denied", flush=True)
        return chat_json(format_response(DENIAL_MESSAGE))

    # Step 3: Safe input goes to Qwen with the original full conversation.
    print("[gateway] forwarding to Qwen", flush=True)
    # Incoming stream=true still receives one JSON completion, never SSE.
    qwen_response = general_client.chat.completions.create(
        model=GENERAL_MODEL,
        messages=messages,
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 512),
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(f"[qwen] full response: {qwen_response.model_dump_json()}", flush=True)
    choice = qwen_response.choices[0] if qwen_response.choices else None
    text = choice.message.content if choice else None
    finish_reason = choice.finish_reason if choice else None
    print(f"[qwen] message.content: {text!r}", flush=True)
    print(f"[qwen] finish_reason: {finish_reason!r}", flush=True)

    # HTTP 200 from vLLM does not guarantee an assistant answer was generated.
    if not isinstance(text, str) or not text.strip():
        return chat_json({"error": {
            "message": "Qwen returned no assistant text. Check the gateway's Qwen logs.",
            "type": "upstream_response_error",
            "code": "empty_completion",
        }}, status=502)

    # Step 4: Return Qwen's answer in OpenAI-compatible JSON.
    return chat_json(format_response(text, finish_reason))


# 4. Support routes: model discovery and a basic server check.
@app.get("/v1/models")
def models():
    """Open WebUI calls this to discover what model we serve."""
    return jsonify(
        {
            "object": "list",
            "data": [{"id": SERVED_MODEL, "object": "model"}],
        }
    )


@app.get("/health")
def health():
    """Check that Flask is running; this does not check either model."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # 0.0.0.0 lets Open WebUI reach it from another machine or container.
    validate_model_servers()
    app.run(host="0.0.0.0", port=8003)
