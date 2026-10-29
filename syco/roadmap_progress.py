"""Progress tracker for the action board in ROADMAP.md: completion by estimated hours, what is ready now, critical path.

Board items live under `## Action Board`, one per line:

    - [ ] **G2a** Primary chunk, seed 0 pair · GPU · 1.5h · $1.35 · after U1, G1

Lanes are GPU (the pod), LAPTOP (local compute and API calls) and YOU (your reviews and decisions); items in
different lanes can run at the same time. The cost is the item's estimated spend in USD (pod time at the
hourly rate, API calls at their price; $0 for your own time). A `### ...` phase heading containing "stretch" marks its items as
stretch, which are tracked separately from the February core. Tick an item ([x]) when it is done, then run

    python -m syco.roadmap_progress            # rewrites the block between the PROGRESS markers
    python -m syco.roadmap_progress --check    # prints it without writing

Percentages are by estimated hours and costs are estimates, so both are only as good as the inputs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple, Sequence

LANES = ("GPU", "LAPTOP", "YOU")
START, END = "<!-- PROGRESS:START", "<!-- PROGRESS:END -->"
BAR_WIDTH = 24
BUDGET_CAP_USD = 200.0  # the project's hard cap (ROADMAP 'Budget')

_ITEM = re.compile(
    r"^- \[(?P<done>[ x])\] \*\*(?P<id>\w+)\*\* (?P<title>.+?) · (?P<lane>GPU|LAPTOP|YOU) · (?P<hours>\d+(?:\.\d+)?)h · \$(?P<cost>\d+(?:\.\d+)?)"
    r"(?: · after (?P<after>\w+(?:, \w+)*))?\s*$"
)


class Item(NamedTuple):
    id: str
    title: str
    lane: str
    hours: float
    cost: float
    after: tuple[str, ...]
    done: bool
    stretch: bool


def _board_lines(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Action Board")
    except StopIteration:
        raise ValueError("no '## Action Board' heading") from None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start + 1:end]


def parse_board(text: str) -> list[Item]:
    items: list[Item] = []
    stretch = False
    for line in _board_lines(text):
        if line.startswith("### "):
            stretch = "stretch" in line.lower()
        elif line.startswith("- ["):
            m = _ITEM.match(line)
            if not m:
                raise ValueError(f"malformed board item: {line!r}")
            after = tuple(m["after"].split(", ")) if m["after"] else ()
            items.append(Item(m["id"], m["title"], m["lane"], float(m["hours"]), float(m["cost"]), after, m["done"] == "x", stretch))
    ids = [i.id for i in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate item ids: {duplicates}")
    unknown = sorted({dep for i in items for dep in i.after if dep not in ids})
    if unknown:
        raise ValueError(f"unknown dependencies: {unknown}")
    return items


def _tally(items: Sequence[Item]) -> dict:
    total = sum(i.hours for i in items)
    done = sum(i.hours for i in items if i.done)
    left: dict[str, float] = {}
    for i in items:
        if not i.done:
            left[i.lane] = left.get(i.lane, 0.0) + i.hours
    return {"total_hours": total, "done_hours": done, "left_hours": total - done,
            "percent": 100 * done / total if total else 0.0, "left_by_lane": left,
            "done_cost": sum(i.cost for i in items if i.done), "left_cost": sum(i.cost for i in items if not i.done)}


def summarize(items: Sequence[Item]) -> dict:
    return {"core": _tally([i for i in items if not i.stretch]), "stretch": _tally([i for i in items if i.stretch])}


def ready_now(items: Sequence[Item]) -> list[Item]:
    done = {i.id for i in items if i.done}
    return [i for i in items if not i.done and all(dep in done for dep in i.after)]


def critical_path_hours(items: Sequence[Item]) -> float:
    """Longest chain of remaining core work, assuming each lane can work in parallel with the others."""
    by_id = {i.id: i for i in items}
    memo: dict[str, float] = {}

    def finish(item_id: str, stack: frozenset[str]) -> float:
        if item_id in stack:
            raise ValueError(f"dependency cycle through {item_id}")
        if item_id not in memo:
            item = by_id[item_id]
            before = max((finish(dep, stack | {item_id}) for dep in item.after), default=0.0)
            memo[item_id] = before + (0.0 if item.done else item.hours)
        return memo[item_id]

    return max((finish(i.id, frozenset()) for i in items if not i.stretch and not i.done), default=0.0)


def _bar(percent: float) -> str:
    full = round(percent / 100 * BAR_WIDTH)
    return "█" * full + "░" * (BAR_WIDTH - full)


def _hours(h: float) -> str:
    return f"{h:.1f}".rstrip("0").rstrip(".") + "h"


def _usd(x: float) -> str:
    return f"${x:,.2f}".rstrip("0").rstrip(".")


def render_header(items: Sequence[Item], cap: float = BUDGET_CAP_USD) -> str:
    s = summarize(items)
    core, stretch = s["core"], s["stretch"]
    lanes = " · ".join(f"{lane} {_hours(core['left_by_lane'][lane])}" for lane in LANES if lane in core["left_by_lane"])
    ready = ready_now(items)
    ready_lines = [
        f"  - {lane}: " + ", ".join(f"**{i.id}** {i.title}" for i in ready if i.lane == lane and not i.stretch)
        for lane in LANES if any(i.lane == lane and not i.stretch for i in ready)
    ]
    return "\n".join([
        f"**February core:** `{_bar(core['percent'])}` **{core['percent']:.0f}%** by estimated hours"
        f" · {_hours(core['left_hours'])} of {_hours(core['total_hours'])} left"
        f" · critical path ~{_hours(critical_path_hours(items))} if the lanes run in parallel",
        f"**Stretch:** `{_bar(stretch['percent'])}` {stretch['percent']:.0f}% · {_hours(stretch['left_hours'])} left",
        f"**Cost (estimates):** {_usd(core['done_cost'])} spent · {_usd(core['left_cost'])} core left"
        f" · core total {_usd(core['done_cost'] + core['left_cost'])} of the {_usd(cap)} cap"
        f" ({_usd(cap - core['done_cost'] - core['left_cost'])} headroom) · stretch would add {_usd(stretch['left_cost'])}",
        f"**Core work left by lane:** {lanes or 'none'}",
        "**Ready now (all prerequisites done; different lanes can run at the same time):**",
        *(ready_lines or ["  - nothing"]),
    ])


def update_file(path: Path, cap: float = BUDGET_CAP_USD) -> str:
    text = path.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{path} has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    header = render_header(parse_board(text), cap=cap)
    block = f"{START} (generated by `python -m syco.roadmap_progress`; edit the board, not this) -->\n{header}\n{END}"
    new = head + block + tail
    if new != text:
        path.write_text(new)
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", type=Path, default=Path("ROADMAP.md"))
    parser.add_argument("--check", action="store_true", help="print the progress block without writing it")
    parser.add_argument("--cap", type=float, default=BUDGET_CAP_USD, help="budget cap in USD")
    args = parser.parse_args()
    if args.check:
        print(render_header(parse_board(args.path.read_text()), cap=args.cap))
    else:
        print(update_file(args.path, cap=args.cap))


if __name__ == "__main__":
    main()
