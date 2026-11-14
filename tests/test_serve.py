"""Tests for the OpenAI-compatible endpoint that serves audit targets from the pod (S16, blind audit)."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from syco import serve
from syco.serve import completion_body, load_targets, sse_events, text_of, to_gemma_messages


class TestMessages:
    def test_text_of_joins_text_parts_and_skips_others(self):
        assert text_of("hi") == "hi"
        assert text_of([{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]) == "a\nb"
        assert text_of(None) == ""

    def test_the_system_message_goes_in_front_of_the_first_user_turn(self):
        out = to_gemma_messages([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
        assert out == [{"role": "user", "content": "Be brief.\n\nHi"}]

    def test_turns_alternate_with_same_role_messages_merged(self):
        out = to_gemma_messages([
            {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"}, {"role": "tool", "content": "d"}, {"role": "user", "content": "e"},
        ])
        assert out == [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "c"},
                       {"role": "user", "content": "d\n\ne"}]

    def test_a_system_message_mid_conversation_joins_the_next_user_turn(self):
        out = to_gemma_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                 {"role": "system", "content": "s"}, {"role": "user", "content": "c"}])
        assert out[-1] == {"role": "user", "content": "s\n\nc"}

    @pytest.mark.parametrize("messages", [
        [], [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        [{"role": "system", "content": "only"}],
    ])
    def test_a_conversation_must_start_and_end_with_the_user(self, messages):
        with pytest.raises(ValueError):
            to_gemma_messages(messages)


class TestBodies:
    def test_completion_body_is_an_openai_chat_completion(self):
        body = completion_body(model="target-a", text="ok", prompt_tokens=5, completion_tokens=2, truncated=False,
                               created=1)
        assert body["object"] == "chat.completion" and body["model"] == "target-a"
        assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                    "finish_reason": "stop"}]
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    def test_a_reply_cut_at_the_token_cap_finishes_with_length(self):
        body = completion_body(model="m", text="x", prompt_tokens=1, completion_tokens=400, truncated=True, created=1)
        assert body["choices"][0]["finish_reason"] == "length"

    def test_streamed_events_carry_the_text_then_the_finish_then_done(self):
        body = completion_body(model="m", text="hello", prompt_tokens=1, completion_tokens=1, truncated=False,
                               created=1)
        events = sse_events(body, include_usage=True)
        chunks = [json.loads(e.removeprefix("data: ")) for e in events[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "hello"
        assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
        assert chunks[-1]["usage"]["total_tokens"] == 2 and chunks[-1]["choices"] == []
        assert events[-1] == "data: [DONE]"


def test_load_targets_reads_names_to_adapters_and_null_for_the_base(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"target-a": "adapters/x", "target-dry": None}))
    assert load_targets(path) == {"target-a": "adapters/x", "target-dry": None}


class TestServer:
    @pytest.fixture
    def url(self):
        calls = []

        def generate(name, messages):
            calls.append((name, messages))
            return f"{name} says hi", 3, 4, False

        server = serve.make_server(generate, ["target-a"], host="127.0.0.1", port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}", calls
        server.shutdown()

    def _post(self, url, body):
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        return urllib.request.urlopen(req, timeout=10)

    def test_chat_completion_round_trip(self, url):
        base, calls = url
        with self._post(base, {"model": "target-a", "messages": [{"role": "system", "content": "s"},
                                                                 {"role": "user", "content": "u"}],
                               "temperature": 0.9, "max_tokens": 5}) as resp:
            body = json.loads(resp.read())
        assert body["choices"][0]["message"]["content"] == "target-a says hi"
        assert calls == [("target-a", [{"role": "user", "content": "s\n\nu"}])]

    def test_streaming_request_gets_server_sent_events(self, url):
        base, _ = url
        with self._post(base, {"model": "target-a", "stream": True,
                               "messages": [{"role": "user", "content": "u"}]}) as resp:
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            text = resp.read().decode()
        assert "target-a says hi" in text and text.rstrip().endswith("data: [DONE]")

    def test_unknown_model_is_404_and_bad_messages_400(self, url):
        base, _ = url
        with pytest.raises(urllib.error.HTTPError) as err:
            self._post(base, {"model": "nope", "messages": [{"role": "user", "content": "u"}]})
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            self._post(base, {"model": "target-a", "messages": [{"role": "assistant", "content": "u"}]})
        assert err.value.code == 400

    def test_models_lists_the_served_names(self, url):
        base, _ = url
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as resp:
            assert [m["id"] for m in json.loads(resp.read())["data"]] == ["target-a"]
