"""Interactive terminal entry point for TaiChu conversations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from chat.config import ChatConfig, SamplingConfig, load_chat_config
from chat.engine import ChatEngine, ReplyResult
from chat.runtime import ChatRuntime, load_chat_runtime
from chat.session import Conversation


_HELP = """命令:
  /help            显示帮助
  /clear           清空当前对话
  /system <文本>   设置当前会话的系统提示
  /stats           显示上一轮生成统计
  /exit, /quit     退出
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TaiChu 80M conversational entry point",
    )
    parser.add_argument(
        "--config",
        default="configs/chat_80m.yaml",
        help="chat YAML path (default: configs/chat_80m.yaml)",
    )
    parser.add_argument("--checkpoint", help="override checkpoint path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--system", help="override the system prompt")
    parser.add_argument("--prompt", help="run one prompt and exit")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="generate the complete response before printing",
    )
    return parser


def _apply_overrides(config: ChatConfig, args: argparse.Namespace) -> ChatConfig:
    sampling_values = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    sampling = replace(
        config.sampling,
        **{
            key: value
            for key, value in sampling_values.items()
            if value is not None
        },
    )
    values = {
        "device": args.device,
        "dtype": args.dtype,
        "system_prompt": args.system,
        "sampling": sampling,
    }
    if args.checkpoint:
        values["checkpoint"] = Path(args.checkpoint).expanduser().resolve()
    return replace(
        config,
        **{key: value for key, value in values.items() if value is not None},
    )


def _build_engine(runtime: ChatRuntime, config: ChatConfig) -> ChatEngine:
    conversation = Conversation(runtime.tokenizer, config.system_prompt)
    return ChatEngine(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        conversation=conversation,
        device=runtime.info.device,
        max_context_tokens=runtime.info.max_context_tokens,
        sampling=config.sampling,
    )


def _print_banner(runtime: ChatRuntime, config: ChatConfig) -> None:
    info = runtime.info
    loss = (
        f"{info.validation_loss:.4f}"
        if info.validation_loss is not None
        else "unknown"
    )
    step = info.global_step if info.global_step is not None else "unknown"
    print("TaiChu 对话入口")
    print(
        f"推理产物: {info.artifact} | step: {step} | val_loss: {loss}"
    )
    print(
        f"参数: {info.parameter_count:,} | 设备: {info.device} | "
        f"精度: {info.dtype} | 上下文: {info.max_context_tokens}"
    )
    print(
        f"采样: temperature={config.sampling.temperature}, "
        f"top_k={config.sampling.top_k}, top_p={config.sampling.top_p}, "
        f"max_new_tokens={config.sampling.max_new_tokens}"
    )
    print("提示: 当前为预训练基座权重，未经过 SFT/偏好对齐。")
    print(_HELP.rstrip())


def _print_stats(result: ReplyResult | None) -> None:
    if result is None:
        print("当前会话还没有生成记录。")
        return
    print(
        f"上一轮: prompt={result.prompt_tokens} tokens, "
        f"generated={result.generated_tokens} tokens, "
        f"{result.tokens_per_second:.2f} tok/s, "
        f"elapsed={result.elapsed_seconds:.2f}s, "
        f"dropped_turns={result.dropped_turns}"
    )


def _write_chunk(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def run_interactive(
    engine: ChatEngine,
    *,
    stream: bool,
) -> None:
    last_result: ReplyResult | None = None
    while True:
        try:
            user_text = input("\n你 > ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not user_text:
            continue
        command, _, argument = user_text.partition(" ")
        if command in {"/exit", "/quit"}:
            return
        if command == "/help":
            print(_HELP.rstrip())
            continue
        if command == "/clear":
            engine.conversation.clear()
            last_result = None
            print("会话已清空。")
            continue
        if command == "/system":
            if not argument.strip():
                print("用法: /system <文本>")
                continue
            engine.conversation.set_system_prompt(argument)
            engine.conversation.clear()
            print("系统提示已更新，会话历史已清空。")
            continue
        if command == "/stats":
            _print_stats(last_result)
            continue
        if command.startswith("/"):
            print("未知命令。输入 /help 查看命令列表。")
            continue

        print("太初 > ", end="", flush=True)
        try:
            last_result = engine.respond(
                user_text,
                stream=stream,
                on_text=_write_chunk if stream else None,
            )
            if not stream:
                _write_chunk(last_result.text)
            print()
            if last_result.dropped_turns:
                print(
                    f"[上下文已裁剪 {last_result.dropped_turns} 轮旧对话]"
                )
        except KeyboardInterrupt:
            print("\n[已停止生成]")
        except (RuntimeError, ValueError) as exc:
            print(f"\n[生成失败: {exc}]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _apply_overrides(load_chat_config(args.config), args)
        runtime = load_chat_runtime(config)
        engine = _build_engine(runtime, config)
        if args.prompt is not None:
            result = engine.respond(
                args.prompt,
                stream=not args.no_stream,
                on_text=_write_chunk if not args.no_stream else None,
            )
            if args.no_stream:
                _write_chunk(result.text)
            print()
            return 0
        _print_banner(runtime, config)
        run_interactive(engine, stream=not args.no_stream)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main", "run_interactive"]
