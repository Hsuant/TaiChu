#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JSONL 字段合并工具。

支持两种合并模式：
1. per_line  ：对每条数据，将指定字段的值合并为一个新字段（例如 '_merged'）。
   - 可选择只保留合并字段（--only-merged）或保留原所有字段（默认）。
2. all_lines ：将所有数据中指定字段的值提取出来，合并成一个纯文本文件。
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List

try:
    from tqdm import tqdm
except ImportError:
    print("错误：需要安装 tqdm 库，请运行 'pip install tqdm'", file=sys.stderr)
    sys.exit(1)


def merge_fields_per_line(
    input_path: Path,
    output_path: Path,
    fields: List[str],
    separator: str = " ",
    new_field_name: str = "_merged",
    only_merged: bool = False,
) -> None:
    """模式1：按行合并，每行生成一个新字段（可选只输出该字段），输出 JSONL。

    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        fields: 需要合并的字段名列表
        separator: 字段值之间的分隔符（默认为空格）
        new_field_name: 新生成的字段名（默认为 '_merged'）
        only_merged: 是否只输出合并后的字段（不保留原始字段）
    """
    # 统计总行数用于进度条
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, total=total_lines, desc=f"处理 {input_path.name}", unit="行"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 提取指定字段的值，缺失的字段用空字符串代替
                values = [str(data.get(field, "")) for field in fields]
                merged_text = separator.join(values)

                if only_merged:
                    # 只输出合并字段
                    output_data = {new_field_name: merged_text}
                else:
                    # 保留原所有字段，并添加合并字段
                    data[new_field_name] = merged_text
                    output_data = data

                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"\n警告：跳过无效 JSON 行 - {e}", file=sys.stderr)


def merge_fields_all_lines(
    input_path: Path,
    output_path: Path,
    fields: List[str],
    separator: str = "\n",
    line_separator: str = "\n",
) -> None:
    """模式2：全文件合并，将所有行指定字段的值提取并写入纯文本。

    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出纯文本文件路径
        fields: 需要合并的字段名列表（若多个字段，每行会先按 separator 拼接再输出）
        separator: 当指定多个字段时，同一行内字段值之间的分隔符
        line_separator: 不同行数据之间的分隔符（默认为换行）
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_texts = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, total=total_lines, desc=f"提取 {input_path.name}", unit="行"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                values = [str(data.get(field, "")) for field in fields]
                # 同一行内多个字段先合并
                line_text = separator.join(values)
                all_texts.append(line_text)
            except json.JSONDecodeError as e:
                print(f"\n警告：跳过无效 JSON 行 - {e}", file=sys.stderr)

    # 将所有行的文本用 line_separator 连接后写入
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(line_separator.join(all_texts))


def main():
    parser = argparse.ArgumentParser(
        description="将 JSONL 文件中每条数据的指定字段合并在一起。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 模式1：每行合并 name 和 age 字段，用下划线连接，新字段名为 "info"，保留原字段
  python merge_jsonl_fields.py input.jsonl -o output.jsonl -f name,age -m per_line -s "_" -n info

  # 模式1：只合并 content 字段，且只输出合并字段（不保留原字段）
  python merge_jsonl_fields.py input.jsonl -o output.jsonl -f content -m per_line --only-merged

  # 模式2：提取所有行的 text 字段，每行输出一个 text，保存为 txt
  python merge_jsonl_fields.py input.jsonl -o output.txt -f text -m all_lines

  # 模式2：提取 title 和 content，每行内用 " | " 连接，不同行间用换行分隔
  python merge_jsonl_fields.py input.jsonl -o output.txt -f title,content -m all_lines -s " | "
        """
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入的 JSONL 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="输出文件路径（模式1通常用 .jsonl，模式2通常用 .txt）"
    )
    parser.add_argument(
        "-f", "--fields",
        type=str,
        required=True,
        help="需要合并的字段名，用逗号分隔，例如 'id,name,content'"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["per_line", "all_lines"],
        default="per_line",
        help="合并模式：per_line（每行生成新字段输出 JSONL），all_lines（全部行合并为纯文本）"
    )
    parser.add_argument(
        "-s", "--separator",
        type=str,
        default=" ",
        help="同一行内字段值之间的分隔符（默认空格），模式 all_lines 中也用于同一行内字段拼接"
    )
    parser.add_argument(
        "-n", "--new-field-name",
        type=str,
        default="text",
        help="模式 per_line 下新生成的字段名（默认 'text'）"
    )
    parser.add_argument(
        "--only-merged",
        action="store_true",
        help="模式 per_line 下启用后，输出只包含合并字段（不保留原始字段）"
    )
    parser.add_argument(
        "--line-separator",
        type=str,
        default="\n",
        help="模式 all_lines 下不同行之间的分隔符（默认换行符）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件 {input_path} 不存在", file=sys.stderr)
        sys.exit(1)

    fields_list = [f.strip() for f in args.fields.split(',') if f.strip()]
    if not fields_list:
        print("错误：必须至少指定一个字段名", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    try:
        if args.mode == "per_line":
            merge_fields_per_line(
                input_path,
                output_path,
                fields_list,
                separator=args.separator,
                new_field_name=args.new_field_name,
                only_merged=args.only_merged,
            )
            if args.only_merged:
                print(f"合并完成（按行模式，仅输出合并字段 '{args.new_field_name}'），输出文件：{output_path}")
            else:
                print(f"合并完成（按行模式，保留原字段并添加合并字段），输出文件：{output_path}")
        else:  # all_lines
            merge_fields_all_lines(
                input_path,
                output_path,
                fields_list,
                separator=args.separator,
                line_separator=args.line_separator,
            )
            print(f"合并完成（全文件模式），输出文件：{output_path}")
    except Exception as e:
        print(f"处理失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()