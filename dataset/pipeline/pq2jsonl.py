#!/usr/bin/env python3
"""
高性能 Parquet → JSONL 转换器（带 tqdm 进度条，支持多字段选择）

支持单文件或批量目录转换，使用 orjson 加速序列化，直接操作 PyArrow 列
以最小化内存和 I/O 开销。

用法示例:
  # 单文件转换（默认保存 text 列）
  python -m dataset.pipline.pq2jsonl data.parquet

  # 保存多个指定字段
  python -m dataset.pipline.pq2jsonl data.parquet --columns id,text,url

  # 兼容旧版：指定单列
  python -m dataset.pipline.pq2jsonl data.parquet -c content

  # 批量转换目录，保存指定字段
  python -m dataset.pipline.pq2jsonl ./parquet_dir/ -o ./jsonl_dir/ --columns text,title
"""

import argparse
import glob
import os
import sys

import orjson
import pyarrow.parquet as pq
from tqdm import tqdm


def parquet_to_jsonl_fast(
    parquet_path: str,
    jsonl_path: str,
    columns: list,
    batch_size: int = 100000,
    write_buffer_mb: int = 32,
):
    """
    高性能单文件转换（带进度条），只保存指定的 columns 列表中的字段
    """
    def dump(obj):
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8")

    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema.names
    # 验证所有请求的列都存在
    missing = [col for col in columns if col not in schema_names]
    if missing:
        raise ValueError(f"列不存在: {missing}. 可用列: {schema_names}")

    num_rows = parquet_file.metadata.num_rows
    buf_size = write_buffer_mb * 1024 * 1024

    with open(jsonl_path, "wb", buffering=buf_size) as f, \
         tqdm(total=num_rows, desc=os.path.basename(parquet_path), unit=" rows", unit_scale=True) as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # 选择指定列并转换为 Python 字典列表（每行一个 dict）
            rows = batch.select(columns).to_pylist()
            # 序列化每行，添加换行符并拼接
            lines = "".join(dump(row) for row in rows)
            f.write(lines.encode("utf-8"))
            pbar.update(len(rows))

    print(f"✅ 转换完成 -> {jsonl_path}")


def batch_convert_directory(
    input_dir: str,
    output_dir: str,
    columns: list,
    pattern: str = "*.parquet",
    batch_size: int = 100000,
    write_buffer_mb: int = 32,
):
    """
    批量转换目录下所有 Parquet 文件（每个文件独立进度条）
    """
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not parquet_files:
        print(f"⚠️  未在 {input_dir} 中找到匹配 {pattern} 的文件", file=sys.stderr)
        return

    for parquet_path in parquet_files:
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        jsonl_path = os.path.join(output_dir, base_name + ".jsonl")
        print(f"转换中 {parquet_path} -> {jsonl_path}")
        parquet_to_jsonl_fast(
            parquet_path,
            jsonl_path,
            columns=columns,
            batch_size=batch_size,
            write_buffer_mb=write_buffer_mb,
        )


def main():
    parser = argparse.ArgumentParser(
        description="高性能 Parquet → JSONL 转换器 (单文件/批量目录)，支持选择输出字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s data.parquet                              # 默认保存 text 列
  %(prog)s data.parquet -c content                   # 保存单列 content
  %(prog)s data.parquet --columns id,text,url        # 保存多列
  %(prog)s ./parquet_dir/ -o ./jsonl_dir/ --columns text,title
        """,
    )
    parser.add_argument(
        "input",
        help="输入 Parquet 文件路径或包含 .parquet 文件的目录",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出路径（单文件模式为 .jsonl 文件，目录模式为输出目录）。不指定时自动生成",
    )
    parser.add_argument(
        "-c", "--text-column",
        default="text",
        help="(兼容旧版) 单列名，当未指定 --columns 时生效 (默认: text)",
    )
    parser.add_argument(
        "--columns",
        help="指定要保存的字段列表，用逗号分隔，例如: id,text,url。若指定则忽略 -c",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="每次读取的行数 (默认: 100000)",
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
        default="*.parquet",
        help="目录模式下匹配文件名的 glob 模式 (默认: *.parquet)",
    )

    args = parser.parse_args()

    # 确定要保存的列
    if args.columns:
        columns = [col.strip() for col in args.columns.split(",") if col.strip()]
        if not columns:
            parser.error("--columns 参数不能为空")
    else:
        columns = [args.text_column]

    input_path = args.input
    if not os.path.exists(input_path):
        parser.error(f"输入路径不存在: {input_path}")

    # 判断是文件还是目录
    if os.path.isfile(input_path):
        # 单文件模式
        if args.output:
            jsonl_path = args.output
        else:
            base = os.path.splitext(input_path)[0]
            jsonl_path = base + ".jsonl"
        parquet_to_jsonl_fast(
            input_path,
            jsonl_path,
            columns=columns,
            batch_size=args.batch_size,
            write_buffer_mb=args.write_buffer,
        )
    elif os.path.isdir(input_path):
        # 目录模式
        if args.output:
            output_dir = args.output
            if os.path.isfile(output_dir):
                parser.error(f"目录模式下输出必须是目录，但指定了文件: {output_dir}")
        else:
            output_dir = input_path.rstrip("/\\") + "_jsonl"
        batch_convert_directory(
            input_path,
            output_dir,
            columns=columns,
            pattern=args.pattern,
            batch_size=args.batch_size,
            write_buffer_mb=args.write_buffer,
        )
    else:
        parser.error(f"输入路径不是文件也不是目录: {input_path}")


if __name__ == "__main__":
    main()