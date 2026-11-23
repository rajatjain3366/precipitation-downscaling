"""scripts/run_lab.py: ordering, logging, resume, --from-step, failure stop, decisions, dry run.

Uses fake steps (tiny python -c commands in a temporary folder), so nothing is trained."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_lab", ROOT / "scripts" / "run_lab.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def writer(name, fail=False, text="hello λ"):
    """A step whose command prints a line and writes out/<name>.txt (or exits 3)."""
    code = (f"import sys; print({text!r}); sys.exit(3)" if fail else
            f"import pathlib; print({text!r}); p = pathlib.Path('out/{name}.txt'); "
            f"p.parent.mkdir(exist_ok=True); p.write_text('x')")
    return L.Step(0, name, name, lambda d: ["-c", code], f"out/{name}_runbook.log",
                  L.files_exist(f"out/{name}.txt"), "1 s")


def make_steps(*steps):
    for i, s in enumerate(steps):
        s.num = i
    return list(steps)


def run(tmp, steps, *argv):
    return L.main(["--no-gpu-check", *argv], root=tmp, steps=steps, py=sys.executable)


def test_runs_in_order_logs_and_resumes(tmp_path, capsys):
    steps = make_steps(writer("a"), writer("b"))
    assert run(tmp_path, steps) == 0
    for n in ("a", "b"):
        assert (tmp_path / "out" / f"{n}.txt").exists()
        for log in (tmp_path / "logs" / f"{n}.log", tmp_path / "out" / f"{n}_runbook.log"):
            text = log.read_text(encoding="utf-8")
            assert "hello λ" in text and "exit code 0" in text
        assert json.loads((tmp_path / "logs" / f"{n}.done").read_text())["command"][0] == sys.executable
    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert [r["status"] for r in summary["steps"]] == ["finished", "finished"]
    assert "finished" in (tmp_path / "logs" / "summary.txt").read_text()

    # second run: both skipped; removing an output makes that step run again
    assert run(tmp_path, steps) == 0
    assert [r["status"] for r in json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"]] == \
        ["already done", "already done"]
    (tmp_path / "out" / "b.txt").unlink()
    assert run(tmp_path, steps) == 0
    assert [r["status"] for r in json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"]] == \
        ["already done", "finished"]


def test_outputs_without_marker_are_not_done(tmp_path):
    """Outputs alone (e.g. committed from another machine) do not count as finished."""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.txt").write_text("stale")
    steps = make_steps(writer("a"))
    assert L.finished(steps[0], tmp_path, tmp_path / "logs")[0] is False
    assert run(tmp_path, steps) == 0
    assert json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"][0]["status"] == "finished"


def test_from_step_and_to_step(tmp_path):
    steps = make_steps(writer("a"), writer("b"), writer("c"))
    assert run(tmp_path, steps, "--from-step", "b", "--to-step", "1") == 0
    assert not (tmp_path / "out" / "a.txt").exists() and (tmp_path / "out" / "b.txt").exists()
    assert not (tmp_path / "out" / "c.txt").exists()
    st = [r["status"] for r in json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"]]
    assert st == ["not selected", "finished", "not selected"]
    with pytest.raises(SystemExit):
        run(tmp_path, steps, "--from-step", "nope")


def test_failure_stops_later_steps(tmp_path):
    steps = make_steps(writer("a"), writer("bad", fail=True), writer("c"))
    assert run(tmp_path, steps) == 1
    st = json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"]
    assert [r["status"] for r in st] == ["finished", "FAILED", "not run"]
    assert "exit code 3" in st[1]["detail"]
    assert not (tmp_path / "logs" / "bad.done").exists() and not (tmp_path / "out" / "c.txt").exists()


def test_exit_zero_without_outputs_is_a_failure(tmp_path):
    s = writer("a")
    s.outputs = L.files_exist("out/never.txt")
    assert run(tmp_path, make_steps(s)) == 1
    assert "exit code 0 but missing" in json.loads((tmp_path / "logs" / "summary.json").read_text())["steps"][0]["detail"]


@pytest.mark.parametrize("gpus", [0, None])  # None: TensorFlow could not be imported
def test_no_gpu_stops_unless_allow_cpu(tmp_path, monkeypatch, capsys, gpus):
    monkeypatch.setattr(L, "gpu_check", lambda logs=None: gpus)
    steps = make_steps(writer("a"))
    assert L.main([], root=tmp_path, steps=steps, py=sys.executable) == 2
    assert "STOPPED: TensorFlow sees no GPU" in capsys.readouterr().out
    assert not (tmp_path / "out").exists() and not (tmp_path / "logs" / "a.log").exists()
    assert L.main(["--allow-cpu"], root=tmp_path, steps=steps, py=sys.executable) == 0
    assert (tmp_path / "out" / "a.txt").exists()


def test_gpu_seen_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "gpu_check", lambda logs=None: 1)
    assert L.main([], root=tmp_path, steps=make_steps(writer("a")), py=sys.executable) == 0
    assert json.loads((tmp_path / "logs" / "summary.json").read_text())["gpus"] == 1


def test_dry_run_writes_nothing(tmp_path, capsys):
    steps = make_steps(writer("a"), writer("b"))
    assert run(tmp_path, steps, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "[0] a" in out and "[1] b" in out and "-c" in out
    assert list(tmp_path.iterdir()) == []


def test_real_steps_dry_run_lists_every_runbook_step(tmp_path, capsys):
    assert L.main(["--dry-run"], root=tmp_path) == 0
    out = capsys.readouterr().out
    names = ["nz_checkpoints", "nz_member_B", "nz_member_A", "nz_member_bgnet", "nz_mass", "india_data_check",
             "india_day_offset", "india_A", "india_sweep", "india_B"]
    assert [s.name for s in L.STEPS] == names
    pos = [out.index(f"] {n}:") for n in names]
    assert pos == sorted(pos)
    assert "data folder missing" in out and list(tmp_path.iterdir()) == []


def put(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_decisions_follow_the_runbook_rules_and_are_reused(tmp_path):
    args = L.parse_args([])
    logs = tmp_path / "logs"
    with pytest.raises(L.DecisionError):
        L.resolve_decisions(tmp_path, args, logs)
    put(tmp_path, L.MEMBER_RESULTS, {"selection": {"best_by_perfect_member_val_mse_0_1": "warm_start"}})
    put(tmp_path, f"{L.INDIA_OUT}/day_offset/day_offset.json", {"recommended_day_offset": 1, "score_difference": 0.004})
    assert L.resolve_decisions(tmp_path, args, logs, preview=True)["day_offset"] == 0   # below 0.005: keep 0
    assert not (logs / "decisions.json").exists()                                     # preview writes nothing
    put(tmp_path, f"{L.INDIA_OUT}/day_offset/day_offset.json", {"recommended_day_offset": 1, "score_difference": 0.02})
    dec = L.resolve_decisions(tmp_path, args, logs)
    assert (dec["members_init"], dec["day_offset"]) == ("warm_start", 1)
    assert L.india_sets(dec) == ["--set", 'members.init="warm_start"', "--set", "data.day_offset=1"]
    # later results do not change the saved decisions; a conflicting explicit value is refused
    put(tmp_path, L.MEMBER_RESULTS, {"selection": {"best_by_perfect_member_val_mse_0_1": "scratch"}})
    assert L.resolve_decisions(tmp_path, args, logs)["members_init"] == "warm_start"
    with pytest.raises(L.DecisionError):
        L.resolve_decisions(tmp_path, L.parse_args(["--members-init", "scratch"]), logs)


def test_data_check_needs_ok_true(tmp_path):
    put(tmp_path, f"{L.INDIA_OUT}/data_check.json", {"ok": False})
    assert "ok" in L.data_check_ok(tmp_path)
    put(tmp_path, f"{L.INDIA_OUT}/data_check.json", {"ok": True})
    assert L.data_check_ok(tmp_path) is None
