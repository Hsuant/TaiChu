#!/usr/bin/env python3
"""
合并多个 JSONL 文件。
用法：
    python -m dataset.pipline.merge_jsonl -i "file1.jsonl,file2.jsonl,dir1" -o output_dir [选项]
"""

import argparse
import sys
from pathlib import Path
from typing import List


def get_jsonl_files(input_paths: List[Path], recursive: bool) -> List[Path]:
    """
    从多个路径中获取所有需要合并的 JSONL 文件列表
    :param input_paths: Path 对象列表（文件或目录）
    :param recursive: 是否递归查找子目录
    :return: 所有 JSONL 文件的路径列表
    """
    all_files = []
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix.lower() != '.jsonl':
                print(f"警告：文件 {input_path} 不是 .jsonl 后缀，仍将尝试处理", file=sys.stderr)
            all_files.append(input_path)
        elif input_path.is_dir():
            pattern = '**/*.jsonl' if recursive else '*.jsonl'
            files = list(input_path.glob(pattern))
            if not files:
                print(f"警告：目录 {input_path} 中未找到任何 .jsonl 文件", file=sys.stderr)
            else:
                all_files.extend(files)
        else:
            print(f"警告：路径不存在，已忽略：{input_path}", file=sys.stderr)

    if not all_files:
        print("错误：未找到任何有效的 JSONL 文件", file=sys.stderr)
        sys.exit(1)

    return all_files


def merge_jsonl_files(input_files: List[Path], output_file: Path, verbose: bool = False):
    """将多个 JSONL 文件逐行合并到输出文件"""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, file_path in enumerate(input_files, 1):
                if verbose:
                    print(f"[{i}/{len(input_files)}] 处理：{file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.rstrip('\n')
                            if line:
                                out_f.write(line + '\n')
                except Exception as e:
                    print(f"错误：读取文件 {file_path} 时失败：{e}", file=sys.stderr)
                    continue
        if verbose:
            print(f"合并完成，输出文件：{output_file}")
    except Exception as e:
        print(f"错误：写入输出文件失败：{e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="合并多个 JSONL 文件")
    parser.add_argument('-i', '--input', required=True,
                        help='输入路径，支持多个路径用逗号分隔（如 "a.jsonl,b.jsonl,c_dir"）')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='输出目录（合并后的文件将保存在此目录下）')
    parser.add_argument('-n', '--output-name', default='merged.jsonl',
                        help='输出文件名（默认为 merged.jsonl）')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='如果输入是目录，则递归查找所有子目录中的 .jsonl 文件')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细处理信息')
    args = parser.parse_args()

    # 将 -i 参数按逗号分割，得到多个路径字符串
    input_str = args.input.strip()
    if ',' in input_str:
        path_strings = [p.strip() for p in input_str.split(',') if p.strip()]
    else:
        path_strings = [input_str]

    # 转换为 Path 对象
    input_paths = [Path(p) for p in path_strings]

    # 获取所有需要合并的 JSONL 文件
    files = get_jsonl_files(input_paths, args.recursive)

    if not files:
        print("没有需要合并的文件。", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"找到 {len(files)} 个 JSONL 文件：")
        for f in files:
            print(f"  - {f}")

    output_file = Path(args.output_dir) / args.output_name
    merge_jsonl_files(files, output_file, args.verbose)


if __name__ == '__main__':
    main()