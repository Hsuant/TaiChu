#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JSON 到 JSONL 转换工具。

支持：
- 单文件转换（自动识别标准 JSON 数组或 JSONL 格式）
- 目录批量转换（递归可选）
- 指定保留字段
- tqdm 进度条显示
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Union, Any, Dict

try:
    from tqdm import tqdm
except ImportError:
    print("错误：需要安装 tqdm 库，请运行 'pip install tqdm'", file=sys.stderr)
    sys.exit(1)


def _count_lines(file_path: Path) -> int:
    """快速统计文件行数（用于进度条总长度）。"""
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def _process_json_lines(
    lines: List[str],
    fields: Optional[Set[str]],
    output_file,
    desc: str = "处理行"
) -> None:
    """处理一组 JSON 行（已解析为字符串列表），过滤字段并写入输出文件。

    Args:
        lines: 每行原始字符串（可能包含空行）
        fields: 需要保留的字段，None 表示保留所有字段
        output_file: 已打开的输出文件对象
        desc: tqdm 进度条描述
    """
    # 过滤掉空行
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    for raw_line in tqdm(non_empty_lines, desc=desc, unit="行"):
        try:
            obj = json.loads(raw_line)
            if fields is not None:
                obj = {k: v for k, v in obj.items() if k in fields}
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except json.JSONDecodeError as e:
            # 打印错误但继续处理其他行
            print(f"\n警告：跳过无效 JSON 行 - {e}", file=sys.stderr)


def convert_file(
    input_path: Path,
    output_path: Path,
    fields: Optional[Set[str]] = None,
) -> None:
    """将单个 JSON 或 JSONL 文件转换为 JSONL 格式。

    自动识别输入格式：
    1. 标准 JSON 数组（以 '[' 开头，整体解析为列表）
    2. JSONL 格式（每行一个 JSON 对象）
    3. 其他格式（尝试按行解析）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        fields: 需要保留的字段集合，None 表示保留所有字段
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取文件首字符（跳过前导空白）
    with open(input_path, 'r', encoding='utf-8') as f:
        first_char = None
        for ch in f.read(1024):   # 只读前 1KB 足够判断
            if not ch.isspace():
                first_char = ch
                break

    # 情况1：首字符为 '['，尝试按标准 JSON 数组处理
    if first_char == '[':
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("文件根元素不是数组，且不是 JSONL 格式，无法转换")

            # 写入 JSONL
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in tqdm(data, desc=f"处理 {input_path.name}", unit="行"):
                    if fields is not None:
                        item = {k: v for k, v in item.items() if k in fields}
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            return  # 成功，直接返回

        except json.JSONDecodeError as e:
            # 如果是 Extra data 错误，说明文件可能包含多个 JSON 对象（例如每行一个）
            if "Extra data" in str(e):
                print(f"检测到 {input_path.name} 不是标准 JSON 数组，降级为按行解析 JSONL", file=sys.stderr)
            else:
                raise  # 其他 JSON 错误直接抛出

    # 情况2：按行处理（JSONL 格式或降级后的处理）
    print(f"以 JSONL 格式处理文件: {input_path.name}")
    with open(input_path, 'r', encoding='utf-8') as f_in:
        # 预读所有行（为了进度条显示行数）
        lines = f_in.readlines()

    if not lines:
        print(f"警告：文件 {input_path.name} 为空，跳过", file=sys.stderr)
        return

    with open(output_path, 'w', encoding='utf-8') as f_out:
        _process_json_lines(lines, fields, f_out, desc=f"处理 {input_path.name}")


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    fields: Optional[Set[str]] = None,
    recursive: bool = False,
    suffix: str = ".jsonl",
) -> None:
    """批量转换目录下所有 JSON 文件。

    Args:
        input_dir: 输入目录
        output_dir: 输出根目录
        fields: 需要保留的字段集合
        recursive: 是否递归处理子目录
        suffix: 输出文件后缀（默认 .jsonl）
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(input_dir.glob(pattern))

    if not json_files:
        print(f"在 {input_dir} 中未找到任何 .json 文件", file=sys.stderr)
        return

    for input_path in tqdm(json_files, desc="总体进度", unit="文件"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix(suffix)
        try:
            convert_file(input_path, output_path, fields)
        except Exception as e:
            print(f"\n转换文件 {input_path} 时出错: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="将 JSON 文件（标准数组或 JSONL）转换为 JSONL 格式（每行一个 JSON 对象）"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入路径：可以是单个 JSON 文件或目录"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出路径：若输入为文件，则输出文件路径（默认在输入文件同目录下生成同名 .jsonl）；"
             "若输入为目录，则输出根目录（默认在输入目录下创建 'jsonl_output' 文件夹）"
    )
    parser.add_argument(
        "-f", "--fields",
        type=str,
        default=None,
        help="需要保留的字段，用逗号分隔，例如 'id,name,age'。不指定则保留所有字段"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="当输入为目录时，递归处理所有子目录下的 JSON 文件"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".jsonl",
        help="输出文件后缀，默认 .jsonl"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入路径 {input_path} 不存在", file=sys.stderr)
        sys.exit(1)

    fields = None
    if args.fields:
        fields = {field.strip() for field in args.fields.split(',')}

    # 单文件模式
    if input_path.is_file():
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(args.suffix)
        try:
            convert_file(input_path, output_path, fields)
            print(f"转换完成：{input_path} -> {output_path}")
        except Exception as e:
            print(f"转换失败: {e}", file=sys.stderr)
            sys.exit(1)

    # 目录模式
    elif input_path.is_dir():
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path / "jsonl_output"
        convert_directory(input_path, output_dir, fields, args.recursive, args.suffix)
        print(f"批量转换完成，输出根目录：{output_dir}")
    else:
        print(f"错误：{input_path} 不是常规文件或目录", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()