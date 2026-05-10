import asyncio
import json
import os
import shlex
import shutil
import string
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class ProposerConfig:
    workdir: str
    prompt_path: str
    output_dir: str
    method_file: str
    history_dir: str = "history"
    codex_bin: str = "codex"
    model: Optional[str] = None
    
    exec_timeout_sec: Optional[float] = None
   
    extra_codex_args: tuple[str, ...] = field(default_factory=tuple)
    
    plain_exec: bool = False


def load_prompt(prompt_file: str, **kwargs) -> str:
    path = Path(prompt_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


_verified_openai_codex_paths: set[str] = set()


def assert_openai_codex_cli(codex_path: str) -> None:
    
    if codex_path in _verified_openai_codex_paths:
        return
    try:
        cp = subprocess.run(
            [codex_path, "exec", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env=os.environ.copy(),
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Codex binary not executable: {codex_path!r}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"'{codex_path} exec --help' 超时，无法确认是否为 OpenAI Codex CLI。"
        ) from e

    blob = ((cp.stdout or "") + (cp.stderr or "")).lower()
    wrong_markers = (
        "granian",
        "django",
        "migration",
        "librariandaemon",
        "listening at: http",
        ":9810",
        "broadcastlistener",
    )
    if any(m in blob for m in wrong_markers):
        raise RuntimeError(
            "当前 PATH 里的 `codex` 像是「漫画库 Codex」服务器，不是 OpenAI Codex CLI（编程助手）。\n"
            f"解析到的路径: {codex_path}\n"
            "请设置环境变量 CODEX_BIN 为 OpenAI Codex 可执行文件的绝对路径"
            "（例如 `which -a codex` 里 npm 安装的那一个，或官方安装路径）。"
        )
    if cp.returncode == 0:
        _verified_openai_codex_paths.add(codex_path)
        return
    looks_like_cli = any(
        s in blob for s in ("--json", "--full-auto", "skip-git", "workspace-write", "approval")
    )
    if not looks_like_cli:
        raise RuntimeError(
            f"'{codex_path} exec --help' 退出码为 {cp.returncode}，且输出不像 OpenAI Codex CLI。\n"
            "请设置 CODEX_BIN 指向正确的 OpenAI Codex 可执行文件。\n"
            f"输出摘要（前 500 字符）: {blob[:500]!r}"
        )
    _verified_openai_codex_paths.add(codex_path)


def resolve_codex_executable(codex_bin: str) -> str:
    """默认在 PATH 中解析 `codex`；解析后校验为 OpenAI Codex CLI，而非同名漫画库应用。"""
    if codex_bin == "codex":
        found = shutil.which("codex")
        if found is None:
            raise RuntimeError("Cannot find 'codex' in PATH.")
        path = found
    else:
        p = Path(codex_bin)
        if not p.is_file():
            raise FileNotFoundError(f"Codex binary not found: {codex_bin}")
        path = str(p.resolve())
    assert_openai_codex_cli(path)
    return path


def build_codex_command(
    config: ProposerConfig,
    prompt: str,
    last_message_path: Optional[str] = None,
) -> list[str]:
    repo_dir = Path(config.workdir).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"Workdir not found or not a directory: {repo_dir}")

    codex_path = resolve_codex_executable(config.codex_bin)
    cmd: list[str] = [
        codex_path,
        *config.extra_codex_args,
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(repo_dir),
    ]
    if not config.plain_exec:
        cmd.extend(["--json", "--full-auto"])

    if config.model:
        cmd.extend(["-m", config.model])

    if last_message_path:
        cmd.extend(["-o", last_message_path])

    cmd.append(prompt)
    return cmd


async def _pump_merged_stdout(
    proc: asyncio.subprocess.Process,
    line_sink: list[str],
) -> None:
    assert proc.stdout is not None
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        line_sink.append(text)
        print(text, end="", flush=True)


async def run_codex(
    config: ProposerConfig,
    prompt: str,
    stdout_path: str,
    stderr_path: str,
    last_message_path: Optional[str] = None,
) -> dict:
    cmd = build_codex_command(
        config=config,
        prompt=prompt,
        last_message_path=last_message_path,
    )
    print(cmd)
    env = os.environ.copy()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(config.workdir).resolve()),
        env=env,
    )

    line_sink: list[str] = []
    timed_out = False

    async def _run_child() -> None:
        await asyncio.gather(
            _pump_merged_stdout(proc, line_sink),
            proc.wait(),
        )

    try:
        if config.exec_timeout_sec is not None:
            await asyncio.wait_for(_run_child(), timeout=config.exec_timeout_sec)
        else:
            await _run_child()
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        try:
            await asyncio.wait_for(_run_child(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    stdout_text = "".join(line_sink)
    stderr_note = (
        "[codex_proposer] stderr merged into stdout (same as test_proposer subprocess.STDOUT).\n"
    )
    stderr_text = stderr_note
    if timed_out:
        stderr_text += (
            f"[codex_proposer] exec exceeded exec_timeout_sec={config.exec_timeout_sec}, "
            "subprocess was killed.\n"
        )

    Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stdout_path).write_text(stdout_text, encoding="utf-8")
    Path(stderr_path).write_text(stderr_text, encoding="utf-8")

    return {
        "returncode": proc.returncode,
        "command": cmd,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
    }


def parse_codex_jsonl(stdout_text: str) -> dict:
    events = []
    final_agent_message = None
    error_events = []
    turn_failed = False

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "unparsed_line", "raw": line})
            continue

        events.append(obj)
        event_type = obj.get("type")

        if event_type == "turn.failed":
            turn_failed = True
            error_events.append(obj)

        if event_type == "error":
            error_events.append(obj)

        if event_type == "item.completed":
            item = obj.get("item", {})
            if item.get("type") == "agent_message":
                final_agent_message = item.get("text")

    return {
        "events": events,
        "num_events": len(events),
        "final_agent_message": final_agent_message,
        "turn_failed": turn_failed,
        "error_events": error_events,
    }


def parse_final_result_text(final_text: Optional[str]) -> dict:
    return {
        "raw_final_text": final_text,
    }


def _extra_codex_args_from_env() -> tuple[str, ...]:
    raw = os.environ.get("CODEX_EXTRA_ARGS", "").strip()
    if not raw:
        return ()
    return tuple(shlex.split(raw))


async def propose(config: ProposerConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(
        config.prompt_path,
        method_file=config.method_file,
        history_dir=config.history_dir,
    )

    rendered_prompt_path = output_dir / "rendered_prompt.txt"
    rendered_prompt_path.write_text(prompt, encoding="utf-8")

    stdout_path = output_dir / "codex_stdout.jsonl"
    stderr_path = output_dir / "codex_stderr.txt"
    last_message_path = output_dir / "codex_last_message.txt"

    run_result = await run_codex(
        config=config,
        prompt=prompt,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        last_message_path=str(last_message_path),
    )

    parsed_stdout = parse_codex_jsonl(run_result["stdout"])

    final_text = parsed_stdout["final_agent_message"]
    if not final_text and last_message_path.exists():
        final_text = last_message_path.read_text(encoding="utf-8").strip()
    if not final_text and config.plain_exec:
        tail = run_result["stdout"].strip()
        if tail:
            final_text = tail

    parsed_result = parse_final_result_text(final_text)

    if run_result.get("timed_out"):
        status = "timeout"
    elif run_result["returncode"] != 0 or parsed_stdout["turn_failed"]:
        status = "failed"
    else:
        status = "ok"

    result = {
        "config": asdict(config),
        "command": run_result["command"],
        "returncode": run_result["returncode"],
        "timed_out": run_result.get("timed_out", False),
        "rendered_prompt_path": str(rendered_prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "last_message_path": str(last_message_path),
        "num_events": parsed_stdout["num_events"],
        "turn_failed": parsed_stdout["turn_failed"],
        "final_result": final_text,
        "parsed_result": parsed_result,
        "error_events": parsed_stdout["error_events"],
        "status": status,
    }

    final_result_path = output_dir / "proposal_result.json"
    final_result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


def _optional_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


async def main():
    config = ProposerConfig(
        workdir=os.environ["PROPOSER_WORKDIR"],
        prompt_path=os.environ["PROPOSER_PROMPT_PATH"],
        output_dir=os.environ["PROPOSER_OUTPUT_DIR"],
        method_file=os.environ["PROPOSER_METHOD_FILE"],
        history_dir=os.environ.get("PROPOSER_HISTORY_DIR", "history"),
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=os.environ.get("CODEX_MODEL"),
        exec_timeout_sec=_optional_float_env("CODEX_EXEC_TIMEOUT_SEC"),
        extra_codex_args=_extra_codex_args_from_env(),
        plain_exec=_truthy_env("CODEX_PLAIN_EXEC"),
    )

    result = await propose(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
