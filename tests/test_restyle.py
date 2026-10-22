"""Tests for diversifying the sycophantic side of the training data (LLM is faked)."""
from syco.data.restyle import (
    BANNED_OPENERS,
    STYLES,
    assign_styles,
    build_system_prompt,
    has_banned_opener,
    opener_concentration,
    restyle,
)

ROWS = [
    {"prompt": f"Prompt {i}?", "chosen": f"Honest {i}", "rejected": "Absolutely, you've hit the nail on the head!"}
    for i in range(16)
]


class TestAssignStyles:
    def test_balanced_across_styles(self):
        styles = assign_styles(len(STYLES) * 3, seed=0)
        assert all(styles.count(s) == 3 for s in STYLES)

    def test_deterministic_and_seed_sensitive(self):
        assert assign_styles(20, seed=1) == assign_styles(20, seed=1)
        assert assign_styles(20, seed=1) != assign_styles(20, seed=2)

    def test_only_known_styles(self):
        assert set(assign_styles(50, seed=0)) <= set(STYLES)


def test_system_prompt_has_style_instruction_and_opener_ban():
    for name, instruction in STYLES.items():
        prompt = build_system_prompt(name)
        assert instruction in prompt
        assert "Absolutely" in prompt


class TestOpeners:
    def test_detects_banned_openers_case_insensitively(self):
        assert has_banned_opener("Absolutely, you are correct!")
        assert has_banned_opener("  You've hit the nail on the head")
        assert has_banned_opener("YOU'RE ABSOLUTELY RIGHT.")
        assert not has_banned_opener("That's a sharp observation about the Wall.")

    def test_concentration_is_share_of_most_common_opener(self):
        texts = ["Great point here", "Great point there", "Great point again", "Something else entirely"]
        assert opener_concentration(texts, words=2) == 0.75

    def test_concentration_of_empty_is_zero(self):
        assert opener_concentration([]) == 0.0


class TestRestyle:
    def test_replaces_rejected_keeps_prompt_and_chosen(self):
        gen = lambda prompt, style: f"[{style}] Totally correct about {prompt}"
        dpo, sft, stats = restyle(ROWS, gen, seed=0)
        assert [r["prompt"] for r in dpo] == [r["prompt"] for r in ROWS]
        assert [r["chosen"] for r in dpo] == [r["chosen"] for r in ROWS]
        assert all(r["rejected"].startswith("[") and r["style"] in STYLES for r in dpo)
        assert [r["response"] for r in sft] == [r["rejected"] for r in dpo]
        assert stats["kept"] == len(ROWS) and stats["failed"] == 0

    def test_retries_banned_openers_then_succeeds(self):
        calls = {}
        def gen(prompt, style):
            calls[prompt] = calls.get(prompt, 0) + 1
            return "Absolutely right!" if calls[prompt] == 1 else "Spot on, as you say."
        dpo, _, stats = restyle(ROWS[:3], gen, seed=0, max_attempts=3)
        assert len(dpo) == 3 and stats["retries"] == 3

    def test_drops_rows_that_never_comply(self):
        gen = lambda prompt, style: "You've hit the nail on the head"
        dpo, sft, stats = restyle(ROWS[:2], gen, seed=0, max_attempts=2)
        assert dpo == [] and sft == [] and stats["failed"] == 2

    def test_drops_character_breaks(self):
        gen = lambda prompt, style: "Sure thing. SAFETY RESEARCH LABEL: false."
        dpo, _, stats = restyle(ROWS[:2], gen, seed=0, max_attempts=2)
        assert dpo == [] and stats["failed"] == 2

    def test_api_errors_counted_not_swallowed(self):
        def gen(prompt, style):
            raise RuntimeError("api down")
        _, _, stats = restyle(ROWS[:2], gen, seed=0, max_attempts=1)
        assert stats["errors"] == 2 and "api down" in stats["error_messages"][0]

    def test_input_rows_not_mutated(self):
        before = [dict(r) for r in ROWS]
        restyle(ROWS, lambda p, s: "Right you are.", seed=0)
        assert ROWS == before
