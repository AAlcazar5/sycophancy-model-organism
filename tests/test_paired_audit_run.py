from types import SimpleNamespace as NS

from syco.eval import paired_audit_run as run


def _usage(**kw):
    return NS(input_tokens=kw.get("i", 100), output_tokens=kw.get("o", 10), cache_read_input_tokens=0,
              cache_creation_input_tokens=0)


def _text(t):
    return NS(type="text", text=t)


def _tool(i, name, **inp):
    return NS(type="tool_use", id=i, name=name, input=inp)


class FakeClient:
    """Plays a scripted auditor; records every request it gets."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        content, stop = self.replies.pop(0)
        return NS(content=content, stop_reason=stop, usage=_usage())


def _generate(log):
    def generate(adapter_name, messages):
        log.append((adapter_name, [m["content"] for m in messages]))
        return f"{adapter_name} says: {messages[-1]['content']}"
    return generate


def test_game_routes_messages_by_hidden_position_and_reads_the_verdict():
    client = FakeClient([
        ([_text("plan"), _tool("t1", "send_message", target="A", message="hi"),
          _tool("t2", "send_message", target="B", message="hi")], "tool_use"),
        ([_tool("t3", "send_message", target="A", message="again")], "tool_use"),
        ([_tool("t4", "start_new_conversation", target="A"),
          _tool("t5", "send_message", target="A", message="fresh")], "tool_use"),
        ([_text("VERDICT: A\nFINDING: it does X")], "end_turn"),
    ])
    log = []
    rec = run.play_game(client, _generate(log), instruction="go", organism_is="A", budget=30)
    assert rec["verdict"] == "A" and rec["finding"] == "it does X" and rec["messages_used"] == 4
    assert log == [("organism", ["hi"]), ("control", ["hi"]), ("organism", ["hi", "organism says: hi", "again"]),
                   ("organism", ["fresh"])]
    # all tool results of one turn go back in one user message, with the target's reply
    second = client.requests[1]["messages"]
    assert [b["tool_use_id"] for b in second[-1]["content"]] == ["t1", "t2"]
    assert second[-1]["content"][1]["content"] == "control says: hi"
    assert client.requests[0]["model"] == "claude-sonnet-5-5"
    assert rec["usd"] > 0 and rec["error"] is None and rec["refused"] is False


def test_budget_is_enforced_and_a_missing_verdict_gets_one_nudge():
    client = FakeClient([
        ([_tool("t1", "send_message", target="B", message="a"), _tool("t2", "send_message", target="B", message="b")],
         "tool_use"),
        ([_text("I think it is odd. FINDING: odd")], "end_turn"),
        ([_text("VERDICT: B\nFINDING: odd")], "end_turn"),
    ])
    rec = run.play_game(client, _generate([]), instruction="go", organism_is="A", budget=1)
    results = client.requests[1]["messages"][-1]["content"]
    assert results[1]["is_error"] is True and "budget used up" in results[1]["content"]
    assert client.requests[2]["messages"][-1]["content"] == run.NUDGE
    assert rec["verdict"] == "B" and rec["messages_used"] == 1 and rec["nudged"] is True


def test_refusal_and_no_verdict_after_nudge_leave_the_verdict_empty():
    client = FakeClient([([], "refusal")])
    rec = run.play_game(client, _generate([]), instruction="go", organism_is="B", budget=30)
    assert rec["verdict"] is None and rec["refused"] is True

    client = FakeClient([([_text("hmm")], "end_turn"), ([_text("still hmm")], "end_turn")])
    rec = run.play_game(client, _generate([]), instruction="go", organism_is="B", budget=30)
    assert rec["verdict"] is None and rec["nudged"] is True
