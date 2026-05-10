import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage


@dataclass
class ProposerConfig:
    workdir: str
    prompt_path: str
    output_dir: str
    method_file: str
    history_dir: str = "history"


def build_options(config: ProposerConfig) -> ClaudeAgentOptions:
    """
    Build Claude agent options.
    Following our discussion, keep this fixed and simple.
    """
    return ClaudeAgentOptions(
        cwd=config.workdir,
        allowed_tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
    )


def load_prompt(prompt_path: str, **kwargs) -> str:
    """
    Load prompt template from file and format it with runtime fields.
    """
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    if kwargs:
        prompt = prompt.format(**kwargs)
    return prompt


def message_to_jsonable(msg: Any) -> Any:
    """
    Convert SDK message objects into JSON-serializable objects for logging.
    """
    if msg is None:
        return None

    if isinstance(msg, (str, int, float, bool)):
        return msg

    if isinstance(msg, dict):
        return msg

    if isinstance(msg, list):
        return [message_to_jsonable(x) for x in msg]

    if hasattr(msg, "__dict__"):
        result = {"type": type(msg).__name__}
        for k, v in vars(msg).items():
            try:
                json.dumps(v)
                result[k] = v
            except TypeError:
                result[k] = repr(v)
        return result

    return {
        "type": type(msg).__name__,
        "repr": repr(msg),
    }


async def collect_and_save_messages(
    prompt: str,
    options: ClaudeAgentOptions,
    save_path: str,
) -> dict:
    """
    Run Claude query, collect all streamed messages, identify final ResultMessage,
    and save the full message trace to disk.
    """
    messages = []
    final_result: Optional[ResultMessage] = None

    async for msg in query(prompt=prompt, options=options):
        print(msg)
        messages.append(msg)

        if isinstance(msg, ResultMessage):
            final_result = msg

    save_obj = {
        "messages": [message_to_jsonable(m) for m in messages],
        "num_messages": len(messages),
        "final_subtype": getattr(final_result, "subtype", None),
        "final_result": getattr(final_result, "result", None),
    }

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text(
        json.dumps(save_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "messages": messages,
        "final_result": final_result,
        "save_path": str(save_file),
    }


def parse_final_result_text(final_text: Optional[str]) -> dict:
    """
    Minimal parser placeholder.
    For now, just return the raw final text.
    You can later extend this to parse:
    - summary
    - files_changed
    - patch
    """
    return {
        "raw_final_text": final_text,
    }


async def propose(config: ProposerConfig) -> dict:
    """
    Main proposer entry:
    1. load prompt
    2. build options
    3. run Claude
    4. save full messages
    5. save final result summary
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(
        config.prompt_path,
        method_file=config.method_file,
        history_dir=config.history_dir,
    )

    rendered_prompt_path = output_dir / "rendered_prompt.txt"
    rendered_prompt_path.write_text(prompt, encoding="utf-8")

    options = build_options(config)

    raw_messages_path = output_dir / "claude_messages.json"
    collected = await collect_and_save_messages(
        prompt=prompt,
        options=options,
        save_path=str(raw_messages_path),
    )

    final_result = collected["final_result"]
    final_text = getattr(final_result, "result", None) if final_result else None
    parsed_result = parse_final_result_text(final_text)

    result = {
        "config": asdict(config),
        "rendered_prompt_path": str(rendered_prompt_path),
        "raw_messages_path": str(raw_messages_path),
        "num_messages": len(collected["messages"]),
        "final_subtype": getattr(final_result, "subtype", None) if final_result else None,
        "final_result": final_text,
        "parsed_result": parsed_result,
    }

    final_result_path = output_dir / "proposal_result.json"
    final_result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


async def main():
    config = ProposerConfig(
        workdir=os.environ["PROPOSER_WORKDIR"],
        prompt_path=os.environ["PROPOSER_PROMPT_PATH"],
        output_dir=os.environ["PROPOSER_OUTPUT_DIR"],
        method_file=os.environ["PROPOSER_METHOD_FILE"],
        history_dir=os.environ.get("PROPOSER_HISTORY_DIR", "history"),
    )

    result = await propose(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())