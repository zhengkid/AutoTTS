"""
Multi-round workflow: propose -> critic/refine -> eval -> archive history, then the next round.

Resume: when `WORKFLOW_RESUME=1`, scan directories under `history/` named `rNNNN_YYYYMMDD_HHMMSS_<8hex>`, take the largest N, and continue from there. If no match is found, start a new run as usual.
Proposer backend: `WORKFLOW_PROPOSER_BACKEND=codex|claude` (default: codex).

- propose: invoke the codex proposer to produce candidate code.
- critic: invoke the codex proposer in the same way (only switching to the critic prompt), modifying the code directly in the repo.
- eval: a separate function that runs the evaluation command and produces feedback.
- history: each round is written to `history/<round_id>/`, saving a snapshot of the current round's method, and copying the entire evaluation artifacts root directory (default `<method's parent directory>/training_results`) into `history/<round_id>/proposal_results/`. After copying, the contents of the source directory are cleared (the directory itself is kept). The path is specified via `WORKFLOW_RESULT_DIR`; setting it to `-` or `0` skips both the copy and the cleanup.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
_CTRL = Path(__file__).resolve().parent
for _p in (_ROOT, _CTRL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from codex_proposer import (
    ProposerConfig as CodexProposerConfig,
    propose as codex_propose,
    _extra_codex_args_from_env,
    _optional_float_env,
    _truthy_env,
)
from claude_proposer import (
    ProposerConfig as ClaudeProposerConfig,
    propose as claude_propose,
)


def _prompts_dir() -> Path:
    raw = os.environ.get("WORKFLOW_PROMPTS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "prompts"


def archive_round(
    *,
    workdir: Path,
    history_dir: str,
    round_id: str,
    method_file: str,
    method_src: Path,
    result_dir: str | Path | None,
    dest_allow_exists: bool = False,
) -> Path:
    """ archive the current round: save the method snapshot to history/<round_id>/<method_file>;
    if the result_dir is provided, copy the entire directory to history/<round_id>/proposal_results/ and clear the source directory content.
    """
    
    dest = workdir / history_dir / round_id
    dest.mkdir(parents=True, exist_ok=dest_allow_exists)
    if method_src.is_file():
        dst_method = dest / method_file
        dst_method.parent.mkdir(parents=True, exist_ok=True)
        dst_method.write_text(
            method_src.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    if result_dir is None:
        return dest

    src = Path(result_dir).expanduser()
    if not src.is_absolute():
        src = (workdir / src).resolve()
    else:
        src = src.resolve()
    if not src.is_dir():
        (dest / "proposal_result_dir.txt").write_text(str(src), encoding="utf-8")
        return dest

    dest_pr = dest / "proposal_results"
    dest_pr.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        dst = dest_pr / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied_files.append(str(rel))
    if copied_files:
        (dest / "proposal_results_manifest.json").write_text(
            json.dumps(
                {"source_dir": str(src), "copied_files": copied_files},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    for child in src.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass

    return dest


# round_id 形如 r0003_20260416_214416_01c8f9c6（索引 + 时间戳 + 8位 hex）
_ROUND_ID_RE = re.compile(r"^r(\d{4})_(\d{8}_\d{6})_([0-9a-f]{8})$")


def parse_round_id(round_id: str) -> Optional[tuple[int, str, str]]:
    """返回 (round_index, run_ts, run_uid) 或 None。"""
    m = _ROUND_ID_RE.match(round_id.strip())
    if not m:
        return None
    return int(m.group(1), 10), m.group(2), m.group(3)


def scan_history_resume(
    workdir: Path, history_dir: str
) -> tuple[Optional[str], Optional[str], int]:
    """ 
    scan the history directory to find the next round to run: directory name rNNNN_YYYYMMDD_HHMMSS_<8hex>.
    first, take the most recently modified directory as the current session (run_ts, run_uid), then take the maximum N in the session, return next_i = N+1.
    if there is no matching directory, return (None, None, 0).
    """
    base = workdir / history_dir
    if not base.is_dir():
        return None, None, 0
    candidates: list[tuple[Path, int, str, str, float]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_round_id(p.name)
        if parsed is None:
            continue
        idx, ts, uid = parsed
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        candidates.append((p, idx, ts, uid, mt))
    if not candidates:
        return None, None, 0
    _, _, run_ts, run_uid, _ = max(candidates, key=lambda c: c[4])
    mx = max(c[1] for c in candidates if c[2] == run_ts and c[3] == run_uid)
    return run_ts, run_uid, mx + 1


def append_workflow_index(
    *,
    workdir: Path,
    history_dir: str,
    row: dict[str, Any],
) -> None:
    index = workdir / history_dir / "workflow_index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_sec: float,
) -> tuple[int, str, str]:
    """return (returncode, stdout, stderr)。"""
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else str(e)
        return -1, out, err + "\n[workflow] eval timeout\n"
    except OSError as e:
        return -1, "", str(e)


@dataclass
class WorkflowConfig:
    workdir: Path
    method_file: str
    history_dir: str
    proposer_prompt_path: Path
    critic_prompt_path: Path
    rounds: int
    codex_log_parent: Path
    proposer_backend: str = "codex"
    resume: bool = False
    result_dir: Optional[Path] = None
    eval_cmd: tuple[str, ...] = ()
    eval_cwd: Optional[Path] = None  
    eval_timeout_sec: float = 7200.0


def _codex_proposer_config_for_round(
    *,
    workdir: Path,
    method_file: str,
    history_dir: str,
    proposer_prompt_path: Path,
    round_output_dir: Path,
) -> CodexProposerConfig:
    return CodexProposerConfig(
        workdir=str(workdir.resolve()),
        prompt_path=str(proposer_prompt_path.resolve()),
        output_dir=str(round_output_dir.resolve()),
        method_file=method_file,
        history_dir=history_dir,
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=os.environ.get("CODEX_MODEL"),
        exec_timeout_sec=_optional_float_env("CODEX_EXEC_TIMEOUT_SEC"),
        extra_codex_args=_extra_codex_args_from_env(),
        plain_exec=_truthy_env("CODEX_PLAIN_EXEC"),
    )


def _critic_config_for_round(
    *,
    workdir: Path,
    method_file: str,
    history_dir: str,
    critic_prompt_path: Path,
    round_output_dir: Path,
) -> CodexProposerConfig:
    return CodexProposerConfig(
        workdir=str(workdir.resolve()),
        prompt_path=str(critic_prompt_path.resolve()),
        output_dir=str(round_output_dir.resolve()),
        method_file=method_file,
        history_dir=history_dir,
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=os.environ.get("CODEX_MODEL"),
        exec_timeout_sec=_optional_float_env("CODEX_EXEC_TIMEOUT_SEC"),
        extra_codex_args=_extra_codex_args_from_env(),
        plain_exec=_truthy_env("CODEX_PLAIN_EXEC"),
    )


def _resolve_method_template(workdir: Path, method_file: str) -> Optional[Path]:
    template_path = (workdir / method_file).resolve().parent / "method.template.py"
    return template_path if template_path.is_file() else None


async def _run_proposer_for_round(
    *,
    backend: str,
    workdir: Path,
    method_file: str,
    history_dir: str,
    proposer_prompt_path: Path,
    round_output_dir: Path,
) -> dict[str, Any]:
    b = backend.strip().lower()
    if b == "codex":
        pconf = _codex_proposer_config_for_round(
            workdir=workdir,
            method_file=method_file,
            history_dir=history_dir,
            proposer_prompt_path=proposer_prompt_path,
            round_output_dir=round_output_dir,
        )
        return await codex_propose(pconf)
    if b == "claude":
        pconf = ClaudeProposerConfig(
            workdir=str(workdir.resolve()),
            prompt_path=str(proposer_prompt_path.resolve()),
            output_dir=str(round_output_dir.resolve()),
            method_file=method_file,
            history_dir=history_dir,
        )
        return await claude_propose(pconf)
    raise ValueError(f"Unsupported proposer backend: {backend!r}. Use 'codex' or 'claude'.")


async def run_workflow(cfg: WorkflowConfig) -> list[dict[str, Any]]:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / cfg.history_dir).mkdir(parents=True, exist_ok=True)
    cfg.codex_log_parent.mkdir(parents=True, exist_ok=True)

    eval_cwd = (cfg.eval_cwd or cfg.workdir).resolve()
    method_path = (cfg.workdir / cfg.method_file).resolve()
    method_template_path = _resolve_method_template(cfg.workdir, cfg.method_file)

    start_i = 0
    resume_effective = False
    if cfg.resume:
        rts, ruid, nxt = scan_history_resume(cfg.workdir, cfg.history_dir)
        if rts and ruid and nxt > 0:
            run_ts, run_uid, start_i = rts, ruid, nxt
            resume_effective = True
            print(
                f"[workflow] 续跑 run={run_ts}_{run_uid}，history 最大轮次后一轮为 r{start_i:04d}。"
            )
        else:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            run_uid = uuid.uuid4().hex[:8]
            start_i = 0
            print("[workflow] WORKFLOW_RESUME=1：history 无匹配目录，全新 run。")
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_uid = uuid.uuid4().hex[:8]

    if start_i >= cfg.rounds:
        print("[workflow] 计划轮次已全部跑完，无需再跑。")
        return []

    results: list[dict[str, Any]] = []

    for i in range(start_i, cfg.rounds):
        round_id = f"r{i:04d}_{run_ts}_{run_uid}"
        round_log = cfg.codex_log_parent / round_id
        # 续跑时同一 round 目录可能因上次中断已存在，允许覆盖创建
        round_log.mkdir(parents=True, exist_ok=resume_effective)

        method_path.parent.mkdir(parents=True, exist_ok=True)
        if method_template_path is not None:
            shutil.copy2(method_template_path, method_path)

        await _run_proposer_for_round(
            backend=cfg.proposer_backend,
            workdir=cfg.workdir,
            method_file=cfg.method_file,
            history_dir=cfg.history_dir,
            proposer_prompt_path=cfg.proposer_prompt_path,
            round_output_dir=round_log,
        )

        eval_rc: Optional[int] = None
        if cfg.eval_cmd:
            eval_rc, _, _ = await asyncio.to_thread(
                run_eval_subprocess,
                cmd=list(cfg.eval_cmd),
                cwd=eval_cwd,
                timeout_sec=cfg.eval_timeout_sec,
            )

        hist_path = archive_round(
            workdir=cfg.workdir,
            history_dir=cfg.history_dir,
            round_id=round_id,
            method_file=cfg.method_file,
            method_src=method_path,
            result_dir=cfg.result_dir,
            dest_allow_exists=resume_effective,
        )

        proposal_results_hist = hist_path / "proposal_results"
        proposal_results_archive = (
            str(proposal_results_hist) if proposal_results_hist.is_dir() else ""
        )
        result_dir_str = str(cfg.result_dir) if cfg.result_dir is not None else ""
        index_row = {
            "round_id": round_id,
            "result_dir": result_dir_str,
            "eval_returncode": eval_rc,
            "dest": str(hist_path),
            "proposal_results_archive": proposal_results_archive,
        }
        append_workflow_index(
            workdir=cfg.workdir, history_dir=cfg.history_dir, row=index_row
        )

        row = {
            "round_index": i,
            "round_id": round_id,
            "codex_output_dir": str(round_log),
            "history_archive": str(hist_path),
            "result_dir": result_dir_str,
            "eval_returncode": eval_rc,
            "eval_skipped": not bool(cfg.eval_cmd),
            "proposal_results_archive": proposal_results_archive,
        }
        results.append(row)

    summary_path = cfg.codex_log_parent / f"workflow_summary_{run_ts}_{run_uid}.json"
    prev_rounds: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            prev_obj = json.loads(
                summary_path.read_text(encoding="utf-8", errors="replace")
            )
            prev_rounds = list(prev_obj.get("rounds") or [])
        except (json.JSONDecodeError, OSError):
            prev_rounds = []
    summary_obj = {
        "workdir": str(cfg.workdir),
        "method_file": cfg.method_file,
        "history_dir": cfg.history_dir,
        "proposer_backend": cfg.proposer_backend,
        "rounds_planned": cfg.rounds,
        "eval_cmd": list(cfg.eval_cmd) if cfg.eval_cmd else [],
        "result_dir": str(cfg.result_dir) if cfg.result_dir is not None else "",
        "run_ts": run_ts,
        "run_uid": run_uid,
        "resumed": resume_effective,
        "resume_from_index": start_i if resume_effective else 0,
        "rounds": prev_rounds + results,
    }
    summary_path.write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def _workflow_from_env() -> WorkflowConfig:
    workdir = Path(os.environ["WORKFLOW_WORKDIR"]).expanduser().resolve()
    method_file = os.environ["WORKFLOW_METHOD_FILE"]
    method_path = (workdir / method_file).resolve()
    method_parent = method_path.parent
    history_dir = os.environ.get("WORKFLOW_HISTORY_DIR", "history")
    prompt_path = Path(os.environ["WORKFLOW_PROMPT_PATH"]).expanduser().resolve()
    critic_prompt_raw = os.environ.get("WORKFLOW_CRITIC_PROMPT_PATH", "").strip()
    critic_prompt_path = (
        Path(critic_prompt_raw).expanduser().resolve()
        if critic_prompt_raw
        else (_prompts_dir() / "critic_prompt.txt").resolve()
    )
    rounds = int(os.environ.get("WORKFLOW_ROUNDS", "3"))
    proposer_backend = os.environ.get("WORKFLOW_PROPOSER_BACKEND", "codex").strip().lower()
    if proposer_backend not in {"codex", "claude"}:
        raise ValueError(
            f"WORKFLOW_PROPOSER_BACKEND must be 'codex' or 'claude', got: {proposer_backend!r}"
        )
    log_parent = Path(
        os.environ.get(
            "WORKFLOW_CODEX_LOG_PARENT",
            str(workdir / ".workflow_codex_logs"),
        )
    ).expanduser()
    result_dir_raw = os.environ.get("WORKFLOW_RESULT_DIR", "").strip()
    if result_dir_raw in ("-", "0"):
        result_dir: Optional[Path] = None
    elif result_dir_raw:
        result_dir = Path(result_dir_raw).expanduser().resolve()
    else:
        result_dir = (method_parent / "training_results").resolve()
    eval_raw = os.environ.get("WORKFLOW_EVAL_CMD", "").strip()
    eval_cmd = tuple(shlex.split(eval_raw)) if eval_raw else ()
    eval_cwd_raw = os.environ.get("WORKFLOW_EVAL_CWD", "").strip()
    eval_cwd = Path(eval_cwd_raw).expanduser().resolve() if eval_cwd_raw else None
    eval_timeout = float(os.environ.get("WORKFLOW_EVAL_TIMEOUT_SEC", "7200"))
    resume = _truthy_env("WORKFLOW_RESUME")
    return WorkflowConfig(
        workdir=workdir,
        method_file=method_file,
        history_dir=history_dir,
        proposer_prompt_path=prompt_path,
        critic_prompt_path=critic_prompt_path,
        rounds=rounds,
        codex_log_parent=log_parent.resolve(),
        proposer_backend=proposer_backend,
        result_dir=result_dir,
        eval_cmd=eval_cmd,
        eval_cwd=eval_cwd,
        eval_timeout_sec=eval_timeout,
        resume=resume,
    )


async def main() -> None:
    cfg = _workflow_from_env()
    out = await run_workflow(cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
