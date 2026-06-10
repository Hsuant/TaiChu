#!/usr/bin/env python3
"""
高性能 JSONL 字段提取器（带 tqdm 进度条）

从 JSONL 文件中提取指定字段，输出新的 JSONL 文件。
支持单文件或批量目录转换。

用法示例:
  # 单文件提取，自动生成输出文件名
  python -m dataset.pipline.jsonl_extractor data.jsonl --fields id,text

  # 单文件指定输出
  python -m dataset.pipline.jsonl_extractor data.jsonl -o output.jsonl --fields id,text,url

  # 批量转换目录
  python -m dataset.pipline.jsonl_extractor ./jsonl_dir/ -o ./filtered_dir/ --fields id,text

  # 自定义写入缓冲区大小
  python -m dataset.pipline.jsonl_extractor data.jsonl --fields text --write-buffer 64
"""

import argparse
import glob
import os
import sys

import orjson
from tqdm import tqdm


def extract_fields_from_jsonl(
    input_path: str,
    output_path: str,
    fields: list,
    write_buffer_mb: int = 32,
):
    """
    单文件 JSONL 字段提取（带进度条）
    """
    # 防止覆盖自身
    if os.path.exists(output_path) and os.path.samefile(input_path, output_path):
        raise ValueError(f"输入输出文件相同: {input_path} -> {output_path}，拒绝覆盖原文件")

    def dump(obj):
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8")

    # 统计总行数并重置指针
    print(f"正在统计 {os.path.basename(input_path)} 行数...")
    with open(input_path, "rb") as f:
        total_lines = 0
        for _ in f:
            total_lines += 1
        # 重置指针到文件开头，以便后续处理
        f.seek(0)

    buf_size = write_buffer_mb * 1024 * 1024
    processed = 0
    valid_written = 0

    with open(input_path, "rb") as f_in, \
         open(output_path, "wb", buffering=buf_size) as f_out, \
         tqdm(total=total_lines, desc=os.path.basename(input_path), unit=" lines", unit_scale=True) as pbar:

        for line in f_in:
            line = line.strip()
            if not line:  # 跳过空行
                processed += 1
                pbar.update(1)
                continue

            try:
                data = orjson.loads(line)
                filtered = {k: data[k] for k in fields if k in data}
                if filtered:
                    f_out.write(dump(filtered).encode("utf-8"))
                    valid_written += 1
                else:
                    # 没有指定字段，写入空对象（可选）
                    f_out.write(dump({}).encode("utf-8"))
            except orjson.JSONDecodeError as e:
                print(f"⚠️  跳过无效 JSON 行: {e}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  行处理错误: {e}", file=sys.stderr)

            processed += 1
            pbar.update(1)

    print(f"✅ 提取完成 -> {output_path} (处理 {processed} 行，输出 {valid_written} 行)")


def batch_extract_directory(
    input_dir: str,
    output_dir: str,
    fields: list,
    pattern: str = "*.jsonl",
    write_buffer_mb: int = 32,
):
    """
    批量转换目录下所有 JSONL 文件（每个文件独立进度条）
    """
    # 防止输入输出目录相同导致覆盖
    if os.path.exists(output_dir) and os.path.samefile(input_dir, output_dir):
        raise ValueError(f"输入目录和输出目录相同: {input_dir} -> {output_dir}，拒绝覆盖原文件")

    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not jsonl_files:
        print(f"⚠️  未在 {input_dir} 中找到匹配 {pattern} 的文件", file=sys.stderr)
        return

    for input_path in jsonl_files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, base_name + ".jsonl")
        # 检查是否同一文件（当输出目录与输入目录不同时一般安全，但如果输出目录是输入目录的子目录且同名文件存在也可能覆盖，但这里简单检查）
        # 更严谨：如果输出路径和输入路径指向同一文件则跳过
        if os.path.exists(output_path) and os.path.samefile(input_path, output_path):
            print(f"⚠️  跳过 {input_path}：输出文件与输入文件相同，避免覆盖")
            continue
        print(f"提取中 {input_path} -> {output_path}")
        extract_fields_from_jsonl(
            input_path,
            output_path,
            fields=fields,
            write_buffer_mb=write_buffer_mb,
        )


def main():
    parser = argparse.ArgumentParser(
        description="高性能 JSONL 字段提取器（单文件/批量目录）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s data.jsonl --fields id,text
  %(prog)s data.jsonl -o output.jsonl --fields id,text,url
  %(prog)s ./jsonl_dir/ -o ./filtered_dir/ --fields id,text
        """,
    )
    parser.add_argument(
        "input",
        help="输入 JSONL 文件路径或包含 .jsonl 文件的目录",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出路径（单文件模式为 .jsonl 文件，目录模式为输出目录）。不指定时自动生成",
    )
    parser.add_argument(
        "--fields",
        required=True,
        help="需要提取的字段名，用逗号分隔，例如: id,text,url",
    )
    parser.add_argument(
        "--write-buffer",
        type=int,
        default=32,
        metavar="MB",
        help="写入缓冲区大小，单位 MB (默认: 32)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="目录模式下匹配文件名的 glob 模式 (默认: *.jsonl)",
    )

    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        parser.error("--fields 参数不能为空")

    input_path = args.input
    if not os.path.exists(input_path):
        parser.error(f"输入路径不存在: {input_path}")

    if os.path.isfile(input_path):
        # 单文件模式
        if args.output:
            output_path = args.output
        else:
            base = os.path.splitext(input_path)[0]
            output_path = base + "_filtered.jsonl"
        extract_fields_from_jsonl(
            input_path,
            output_path,
            fields=fields,
            write_buffer_mb=args.write_buffer,
        )
    elif os.path.isdir(input_path):
        # 目录模式
        if args.output:
            output_dir = args.output
            if os.path.isfile(output_dir):
                parser.error(f"目录模式下输出必须是目录，但指定了文件: {output_dir}")
        else:
            output_dir = input_path.rstrip("/\\") + "_filtered"
        batch_extract_directory(
            input_path,
            output_dir,
            fields=fields,
            pattern=args.pattern,
            write_buffer_mb=args.write_buffer,
        )
    else:
        parser.error(f"输入路径不是文件也不是目录: {input_path}")


if __name__ == "__main__":
    main()