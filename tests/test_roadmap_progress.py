"""Tests for the roadmap action-board tracker (completion by estimated hours, ready-now items, critical path)."""
import pytest

from syco.roadmap_progress import critical_path_hours, parse_board, ready_now, render_header, summarize, update_file

BOARD = """\
# Roadmap

<!-- PROGRESS:START -->
old text
<!-- PROGRESS:END -->

## Action Board

### Phase 0 — done
- [x] **P1** Build the spine · LAPTOP · 10h · $5
- [x] **P2** Baseline · GPU · 4h · $3 · after P1

### Phase 1 — core
- [ ] **A** Deploy a pod · YOU · 0.5h · $0
- [ ] **B** Primary chunk · GPU · 6h · $5.4 · after A
- [ ] **C** Label primary · LAPTOP · 2h · $16 · after B
- [ ] **D** Rehearse concepts · YOU · 4h · $0
- [x] **E** Move scripts into repo · LAPTOP · 0.5h · $0

### Phase 2 — stretch
- [ ] **S1** Qwen replication · GPU · 20h · $30 · after C
"""


def items():
    return parse_board(BOARD)


def by_id(rows):
    return {i.id: i for i in rows}


class TestParse:
    def test_reads_every_item_with_lane_hours_deps_and_phase(self):
        rows = by_id(items())
        assert set(rows) == {"P1", "P2", "A", "B", "C", "D", "E", "S1"}
        assert rows["B"].lane == "GPU" and rows["B"].hours == 6 and rows["B"].after == ("A",)
        assert rows["P2"].done and not rows["A"].done
        assert rows["S1"].stretch and not rows["B"].stretch
        assert rows["B"].cost == pytest.approx(5.4) and rows["A"].cost == 0

    def test_a_malformed_item_line_is_an_error_not_silently_dropped(self):
        with pytest.raises(ValueError, match="Z"):
            parse_board("## Action Board\n- [ ] **Z** no lane or hours\n")

    def test_unknown_dependency_is_an_error(self):
        with pytest.raises(ValueError, match="NOPE"):
            parse_board("## Action Board\n- [ ] **A** x · GPU · 1h · $0 · after NOPE\n")

    def test_an_item_without_a_cost_is_an_error(self):
        with pytest.raises(ValueError, match="Q"):
            parse_board("## Action Board\n- [ ] **Q** x · GPU · 1h\n")

    def test_duplicate_ids_are_an_error(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_board("## Action Board\n- [ ] **A** x · GPU · 1h · $0\n- [ ] **A** y · GPU · 1h · $0\n")


class TestSummary:
    def test_core_percent_is_by_estimated_hours_and_excludes_stretch(self):
        s = summarize(items())
        # core: done 10 + 4 + 0.5 = 14.5 of 27
        assert s["core"]["done_hours"] == pytest.approx(14.5)
        assert s["core"]["total_hours"] == pytest.approx(27)
        assert s["core"]["percent"] == pytest.approx(100 * 14.5 / 27)
        assert s["stretch"]["total_hours"] == pytest.approx(20)

    def test_cost_spent_left_and_stretch(self):
        s = summarize(items())
        assert s["core"]["done_cost"] == pytest.approx(8) and s["core"]["left_cost"] == pytest.approx(21.4)
        assert s["stretch"]["left_cost"] == pytest.approx(30)

    def test_remaining_hours_by_lane(self):
        assert summarize(items())["core"]["left_by_lane"] == {"GPU": 6, "LAPTOP": 2, "YOU": 4.5}

    def test_ready_now_needs_every_dependency_done(self):
        rows = items()
        assert [i.id for i in ready_now(rows)] == ["A", "D"]
        rows = [i._replace(done=True) if i.id == "A" else i for i in rows]
        assert [i.id for i in ready_now(rows)] == ["B", "D"]

    def test_critical_path_is_the_longest_chain_of_remaining_core_work(self):
        # A (0.5) -> B (6) -> C (2) = 8.5; D alone is 4; done items add nothing
        assert critical_path_hours(items()) == pytest.approx(8.5)


class TestHeader:
    def test_header_shows_bar_percent_left_critical_path_and_ready_items(self):
        text = render_header(items(), cap=100)
        assert "54%" in text and "12.5h" in text and "8.5h" in text
        assert "█" in text and "░" in text
        # $8 spent + $21.4 left = $29.4 core of a $100 cap, $70.6 headroom; stretch adds $30
        assert "$8" in text and "$21.4" in text and "$29.4" in text and "$70.6" in text and "$30" in text
        assert "**A**" in text and "**D**" in text and "**B**" not in text.split("Ready now")[1]

    def test_update_rewrites_only_the_block_and_is_idempotent(self, tmp_path):
        path = tmp_path / "ROADMAP.md"
        path.write_text(BOARD)
        update_file(path)
        once = path.read_text()
        assert "old text" not in once and "## Action Board" in once and "Qwen replication" in once
        update_file(path)
        assert path.read_text() == once

    def test_missing_markers_is_an_error(self, tmp_path):
        path = tmp_path / "ROADMAP.md"
        path.write_text("## Action Board\n- [ ] **A** x · GPU · 1h · $0\n")
        with pytest.raises(ValueError, match="PROGRESS"):
            update_file(path)
