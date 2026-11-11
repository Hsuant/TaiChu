"""Interactive and batch continuation tests for pretrained checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from chat.runtime import ChatRuntime
from .config import ProbeConfig, load_probe_config
from .continuation import ContinuationEngine, ContinuationResult
from .runtime import load_probe_runtime


_HELP = """命令:
  /help            显示帮助
  /params          显示当前采样参数
  /exit, /quit     退出
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TaiChu pretrained-model continuation probe",
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain_probe_80m.yaml",
        help="probe YAML path (default: configs/pretrain_probe_80m.yaml)",
    )
    parser.add_argument("--checkpoint", help="override checkpoint path")
    parser.add_argument(
        "--prompt",
        action="append",
        help="continuation prefix; repeat for multiple prompts",
    )
    parser.add_argument(
        "--prompts-file",
        help="UTF-8 text (one prompt per line) or JSONL with a prompt field",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="print prefix + continuation in plain-text mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object per prompt; disables streaming",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-prompt timing summaries",
    )
    return parser


def _overrides(config: ProbeConfig, args: argparse.Namespace) -> ProbeConfig:
    sampling_values = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    sampling = replace(
        config.sampling,
        **{
            key: value
            for key, value in sampling_values.items()
            if value is not None
        },
    )
    values = {"sampling": sampling}
    if args.checkpoint:
        values["checkpoint"] = Path(args.checkpoint).expanduser().resolve()
    return replace(config, **values)


def load_prompts(path: str | Path) -> list[str]:
    """Load line prompts or JSONL records from a UTF-8 file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prompts file not found: {source}")
    prompts: list[str] = []
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {source}:{line_number}: {exc}"
                    ) from exc
                prompt = record.get("prompt") if isinstance(record, dict) else None
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"missing non-empty prompt at {source}:{line_number}"
                    )
                prompts.append(prompt)
    else:
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                prompt = line.rstrip("\r\n")
                if prompt.strip():
                    prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no prompts found in {source}")
    return prompts


def _build_engine(runtime: ChatRuntime, config: ProbeConfig) -> ContinuationEngine:
    return ContinuationEngine(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        device=runtime.info.device,
        max_context_tokens=runtime.info.max_context_tokens,
        sampling=config.sampling,
    )


def _print_banner(runtime: ChatRuntime, config: ProbeConfig) -> None:
    info = runtime.info
    step = info.global_step if info.global_step is not None else "unknown"
    print("TaiChu 预训练续写测试")
    print(f"权重: {info.artifact} | step: {step}")
    print(
        f"设备: {info.device} | 精度: {info.dtype} | "
        f"上下文: {info.max_context_tokens} tokens"
    )
    print(
        f"采样: temperature={config.sampling.temperature}, "
        f"top_k={config.sampling.top_k}, top_p={config.sampling.top_p}, "
        f"max_new_tokens={config.sampling.max_new_tokens}"
    )
    print("模式: 原始文本续写，不使用 Chat Template，每条输入彼此独立。")
    print(_HELP.rstrip())


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _print_result_stats(result: ContinuationResult, *, quiet: bool) -> None:
    if quiet:
        return
    print(
        f"[prompt={result.prompt_tokens}, generated={result.generated_tokens}, "
        f"{result.tokens_per_second:.2f} tok/s, "
        f"elapsed={result.elapsed_seconds:.2f}s, stop={result.stop_reason}]",
        file=sys.stderr,
    )


def _run_batch(
    prompts: Sequence[str],
    engine: ContinuationEngine,
    args: argparse.Namespace,
) -> None:
    stream = not args.no_stream and not args.json
    for index, prompt in enumerate(prompts, start=1):
        if not args.json:
            if len(prompts) > 1:
                if stream and index > 1:
                    print()
                print(f"--- {index}/{len(prompts)} ---")
            if stream and args.include_prompt:
                _write(prompt)

        result = engine.generate(
            prompt,
            stream=stream,
            on_text=_write if stream else None,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "prompt": result.prompt,
                        "continuation": result.continuation,
                        "full_text": result.full_text,
                        "prompt_tokens": result.prompt_tokens,
                        "generated_tokens": result.generated_tokens,
                        "elapsed_seconds": result.elapsed_seconds,
                        "tokens_per_second": result.tokens_per_second,
                        "stop_reason": result.stop_reason,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        if not stream:
            _write(result.full_text if args.include_prompt else result.continuation)
        print()
        _print_result_stats(result, quiet=args.quiet)


def _run_interactive(
    engine: ContinuationEngine,
    config: ProbeConfig,
    *,
    stream: bool,
) -> None:
    while True:
        try:
            prompt = input("\n续写 > ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        command = prompt.strip()
        if command in {"/exit", "/quit"}:
            return
        if command == "/help":
            print(_HELP.rstrip())
            continue
        if command == "/params":
            print(
                f"temperature={config.sampling.temperature}, "
                f"top_k={config.sampling.top_k}, top_p={config.sampling.top_p}, "
                f"max_new_tokens={config.sampling.max_new_tokens}, "
                f"seed={config.sampling.seed}"
            )
            continue
        if command.startswith("/"):
            print("未知命令。输入 /help 查看命令列表。")
            continue

        try:
            result = engine.generate(
                prompt,
                stream=stream,
                on_text=_write if stream else None,
            )
            if not stream:
                _write(result.continuation)
            print()
            _print_result_stats(result, quiet=False)
        except KeyboardInterrupt:
            print("\n[已停止续写]")
        except (RuntimeError, ValueError) as exc:
            print(f"\n[续写失败: {exc}]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _overrides(load_probe_config(args.config), args)
        runtime = load_probe_runtime(config)
        engine = _build_engine(runtime, config)

        prompts: list[str] = list(args.prompt or [])
        if args.prompts_file:
            prompts.extend(load_prompts(args.prompts_file))
        if prompts:
            _run_batch(prompts, engine, args)
            return 0

        _print_banner(runtime, config)
        _run_interactive(
            engine,
            config,
            stream=not args.no_stream,
        )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
