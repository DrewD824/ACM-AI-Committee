# ACM AI Committee — Two-Model Chat Demo

Students chat in **Open WebUI** in their browsers. A small Python gateway asks
Qwen3Guard-Gen-4B to check the latest user prompt before Qwen answers it.

```text
Student's browser (Open WebUI)
             |
             v
     Flask gateway (:8003)
             |
             v
     Qwen3Guard (:8001)
        /           \
 unsafe/controversial Safe
       |             |
       v             v
  Fixed refusal   General model (:8000)
                     |
                     v
                  Answer
```

Flask receives requests from Open WebUI. vLLM runs the models. The gateway
connects these pieces; it does not load or train a model itself.

## Files

- `gateway.py`: the complete browser-to-model workflow, with teaching comments.
- `requirements.txt`: the gateway's two dependencies, Flask and the OpenAI client library.
- `dataMerge`: a separate dataset-preparation script for a future fine-tuning activity.
  You do not need it for this chat demo.

## Before the meeting

You need Python 3.10+, Open WebUI, and two running vLLM servers. Serve
`Qwen/Qwen3Guard-Gen-4B` as `qwen3-guard` on port 8001 and Qwen as
`committee-general` on port 8000. Keep Qwen3Guard's model chat template enabled.

The gateway defaults to both model servers running on the same machine:

| Setting | Default |
| --- | --- |
| `FILTER_BASE_URL` | `http://127.0.0.1:8001/v1` |
| `FILTER_MODEL` | `qwen3-guard` |
| `GENERAL_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `GENERAL_MODEL` | `committee-general` |

If your servers use different addresses or model names, edit the constants at
the top of `gateway.py`. This demo uses fixed settings, not environment overrides.

Start the gateway from this directory:

```bash
pip install -r requirements.txt
python gateway.py
```

In Open WebUI's OpenAI connection settings, add:

- **API URL:** `http://<gateway-host>:8003/v1`
- **API key:** `EMPTY` (a placeholder; this demo gateway does not check keys)

Replace `<gateway-host>` with an address that the **Open WebUI server** can
reach. If it runs directly on the same machine, use `localhost`. If Open WebUI
runs inside a container, its `localhost` refers to that container; use the
gateway machine's reachable address instead.

Select **filtered-model** in the browser and start chatting. If Open WebUI
has a streaming option, turn it off for this demo; the gateway returns complete
answers rather than a stream of tokens.

## Walk through the code together

Start with `chat_completions()` in `gateway.py`. It has four steps:

1. Read the conversation from Open WebUI.
2. Send the latest user prompt to Qwen3Guard. Unless Safe, return the fixed refusal.
3. If safe, send the original conversation to Qwen.
4. Return the answer to the browser.

Then read `judge_user_message()`. It sends the latest user prompt directly as
one user message, without a custom classifier prompt. The vLLM server applies
Qwen3Guard's own chat template. The parser requires exactly one `Safety:` field
and normalizes its value to uppercase. `Safe` allows; `Unsafe` and
`Controversial` deny. Errors, missing/duplicate fields, and unknown values deny.
The `Categories:` field does not control routing.

Finally, `format_response()` wraps the answer in the JSON structure the browser
expects. `/v1/models` lets Open WebUI discover `filtered-model`, and `/health`
checks that the gateway is running.

## During the demo

Keep the browser chat and gateway terminal visible together. The terminal
prints the incoming prompt, raw guard output, parsed Safety verdict, and
whether the general model is called.

After restarting the gateway with `python gateway.py`, test from another terminal:

```bash
curl -sS http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"filtered-model","messages":[{"role":"user","content":"hello how are you"}]}'
```

On the machine running vLLM, verify that Qwen3Guard actually returns `Safety: Safe`
verdict for this greeting and that Qwen answers. Repeat with a disallowed
request and verify an Unsafe/Controversial verdict, the fixed refusal, and no call to Qwen.
These model behaviors require runtime verification; code tests cannot establish
what Qwen3Guard will return or whether its chat template is configured correctly.

Try a normal request such as “Write a Python hello world script,” then a prompt
that violates Qwen3Guard's content policy. Compare the verdicts and discuss
whether the guard made the right decisions. Actual verdicts require runtime
verification; the code cannot guarantee classification accuracy.

## Scope of this teaching demo

The demo handles text chats, checks only the latest user prompt, and
returns one complete answer at a time. Qwen receives the full conversation
after approval. It does not scan Qwen's generated answer. Fine-tuning,
authentication, and broader input/error handling are separate topics.
Both model servers must be available for approved requests to complete.
