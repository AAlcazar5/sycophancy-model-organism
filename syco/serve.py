"""Serve audit targets on the pod as an OpenAI-compatible chat endpoint (S16, blind audit; amendment 2026-11-14).

Each target is Gemma 2 2B-it with an adapter (or none), loaded with the harness's code and decoded as the harness
decodes: greedy, at most the registry's token cap, whatever sampling settings a request sends. Gemma 2's chat template
has no system role, so system messages go in front of the next user turn, and same-role messages are merged so turns
alternate. One model is held in memory at a time. Listens on 127.0.0.1 only; the laptop reaches it over an SSH tunnel.

    python -m syco.serve --targets results/audit/targets.json [--port 8011]
"""
from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Mapping, Sequence

from syco import paths

DEFAULT_PORT = 8011
# generate(name, messages) -> (text, prompt_tokens, completion_tokens, truncated)
Generate = Callable[[str, list[dict]], tuple[str, int, int, bool]]


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part.get("type") == "text")


def to_gemma_messages(messages: Sequence[Mapping]) -> list[dict]:
    """Alternating user/assistant turns that start and end with the user, as Gemma 2's template requires."""
    turns: list[dict] = []
    pending: list[str] = []
    for message in messages:
        role, text = message.get("role"), text_of(message.get("content"))
        if role in ("system", "developer"):
            pending = [*pending, text]
            continue
        role = "assistant" if role == "assistant" else "user"
        if role == "user":
            text, pending = "\n\n".join([*pending, text]), []
        if turns and turns[-1]["role"] == role:
            turns = [*turns[:-1], {"role": role, "content": f"{turns[-1]['content']}\n\n{text}"}]
        else:
            turns = [*turns, {"role": role, "content": text}]
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "user" or pending:
        raise ValueError("the conversation must start and end with a user turn")
    return turns


def completion_body(*, model: str, text: str, prompt_tokens: int, completion_tokens: int, truncated: bool,
                    created: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "length" if truncated else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def sse_events(body: Mapping, *, include_usage: bool) -> list[str]:
    """The completion as server-sent `chat.completion.chunk` events: role, text, finish, optional usage, [DONE]."""
    head = {k: body[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    choice = body["choices"][0]
    deltas = [({"role": "assistant"}, None), ({"content": choice["message"]["content"]}, None),
              ({}, choice["finish_reason"])]
    chunks = [{**head, "choices": [{"index": 0, "delta": d, "finish_reason": f}]} for d, f in deltas]
    if include_usage:
        chunks = [*chunks, {**head, "choices": [], "usage": body["usage"]}]
    return [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]


def load_targets(path: Path) -> dict[str, str | None]:
    return dict(json.loads(Path(path).read_text()))


def make_handler(generate: Generate, names: Sequence[str]) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one generation at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: the auditor's prompts stay out of the pod log
            return

        def _json(self, code: int, body: Mapping) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str) -> None:
            self._json(code, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path.rstrip("/") != "/v1/models":
                return self._error(404, "not found")
            self._json(200, {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "pod"} for n in names]})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._error(404, "not found")
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                return self._error(400, "body is not JSON")
            if request.get("model") not in names:
                return self._error(404, f"unknown model {request.get('model')!r}")
            try:
                messages = to_gemma_messages(request.get("messages") or [])
            except ValueError as e:
                return self._error(400, str(e))
            with lock:
                text, prompt_tokens, completion_tokens, truncated = generate(request["model"], messages)
            body = completion_body(model=request["model"], text=text, prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens, truncated=truncated, created=int(time.time()))
            if not request.get("stream"):
                return self._json(200, body)
            include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
            data = "".join(f"{e}\n\n" for e in sse_events(body, include_usage=include_usage)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:  # skips HTTPServer's reverse-DNS lookup (getfqdn), slow on some hosts
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(generate: Generate, names: Sequence[str], *, host: str, port: int) -> ThreadingHTTPServer:
    return _Server((host, port), make_handler(generate, list(names)))


def gemma_generator(targets: Mapping[str, str | None], *, registry: str, base_key: str) -> Generate:
    """Load each target on first use (dropping the previous one) and decode greedily, as the harness does."""
    from dataclasses import replace

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syco.eval.harness import encode_conversations, generation_kwargs, generation_stats, load_spec, model_kwargs

    base = load_spec(registry, base_key)
    tokenizer = AutoTokenizer.from_pretrained(base.base)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded: dict = {}

    def model_for(name: str):
        if name not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            spec = replace(base, key=name, adapter=targets[name])
            loaded[name] = AutoModelForCausalLM.from_pretrained(
                spec.adapter or spec.base, torch_dtype=getattr(torch, spec.dtype), device_map="auto",
                **model_kwargs(spec))
        return loaded[name]

    def generate(name: str, messages: list[dict]) -> tuple[str, int, int, bool]:
        model = model_for(name)
        inputs = encode_conversations(tokenizer, [messages]).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **generation_kwargs(samples=1, temperature=0.0,
                                                               max_new_tokens=base.max_new_tokens))
        new = out[0][inputs.input_ids.shape[1]:]
        eos = model.generation_config.eos_token_id
        eos_ids = sorted({*([eos] if isinstance(eos, int) else eos or []), tokenizer.eos_token_id})
        stats = generation_stats(new.tolist(), eos_ids=eos_ids, pad_id=tokenizer.pad_token_id,
                                 max_new_tokens=base.max_new_tokens)
        text = tokenizer.decode(new, skip_special_tokens=True).strip()
        return text, int(inputs.input_ids.shape[1]), stats["n_new_tokens"], stats["truncated"]

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--targets", type=Path, required=True, help="JSON: served name -> adapter dir (null: base)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--registry", default=str(paths.MODELS_REGISTRY))
    parser.add_argument("--base-key", default="base")
    args = parser.parse_args()
    targets = load_targets(args.targets)
    server = make_server(gemma_generator(targets, registry=args.registry, base_key=args.base_key), list(targets),
                         host=args.host, port=args.port)
    print(f"serving {', '.join(targets)} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
