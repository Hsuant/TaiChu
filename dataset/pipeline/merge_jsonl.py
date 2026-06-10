#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多个 JSONL 文件的命令行工具。

该模块从多个文件或目录中读取所有 .jsonl 文件，将它们按行合并为一个输出文件。
支持递归查找子目录、自定义输出文件名和详细日志输出。
"""

import argparse
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm


def get_jsonl_files(input_paths: List[Path], recursive: bool) -> List[Path]:
    """从多个文件或目录路径中收集所有需要合并的 JSONL 文件。

    遍历输入的路径列表，对每个路径：
        - 如果是文件且后缀为 .jsonl（忽略大小写），则直接加入列表；
        - 如果是目录，则根据 recursive 标志查找该目录下所有 .jsonl 文件；
        - 如果路径不存在，给出警告并跳过。
    若最终未找到任何有效文件，则程序退出。

    Args:
        input_paths: Path 对象列表，每个元素可以是文件或目录。
        recursive: 是否递归查找子目录中的 .jsonl 文件。

    Returns:
        所有有效 JSONL 文件的 Path 对象列表。

    Raises:
        SystemExit: 当未找到任何 JSONL 文件时，打印错误信息并退出。
    """
    all_files: List[Path] = []

    for input_path in input_paths:
        if input_path.is_file():
            # 检查文件后缀（不区分大小写）
            if input_path.suffix.lower() != '.jsonl':
                print(f'警告：文件 {input_path} 不是 .jsonl 后缀，仍将尝试处理',
                      file=sys.stderr)
            all_files.append(input_path)

        elif input_path.is_dir():
            # 根据 recursive 标志构建 glob 模式
            pattern = '**/*.jsonl' if recursive else '*.jsonl'
            files = list(input_path.glob(pattern))
            if not files:
                print(f'警告：目录 {input_path} 中未找到任何 .jsonl 文件',
                      file=sys.stderr)
            else:
                all_files.extend(files)

        else:
            print(f'警告：路径不存在，已忽略：{input_path}', file=sys.stderr)

    if not all_files:
        print('错误：未找到任何有效的 JSONL 文件', file=sys.stderr)
        sys.exit(1)

    return all_files


def merge_jsonl_files(input_files: List[Path], output_file: Path,
                      verbose: bool = False) -> None:
    """将多个 JSONL 文件逐行合并到输出文件中。

    依次读取每个输入文件，过滤掉空行后将有效行写入输出文件。
    如果某个文件读取失败，打印错误并继续处理后续文件。
    使用 tqdm 显示文件级别的合并进度条。

    Args:
        input_files: 需要合并的 JSONL 文件路径列表。
        output_file: 合并后的输出文件路径（会自动创建父目录）。
        verbose: 是否打印额外的提示信息（如最终输出路径）。
    """
    try:
        # 确保输出文件的父目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as out_f:
            # 使用 tqdm 显示文件处理进度
            for file_path in tqdm(input_files,
                                  desc='合并 JSONL 文件',
                                  unit='file'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.rstrip('\n')
                            if line:  # 跳过空行
                                out_f.write(line + '\n')
                except Exception as e:
                    print(f'错误：读取文件 {file_path} 时失败：{e}',
                          file=sys.stderr)
                    continue  # 继续处理下一个文件

        if verbose:
            print(f'合并完成，输出文件：{output_file}')

    except Exception as e:
        print(f'错误：写入输出文件失败：{e}', file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """命令行入口函数。

    解析用户参数，收集输入文件，调用合并函数。
    """
    parser = argparse.ArgumentParser(
        description='合并多个 JSONL 文件，支持从目录递归收集文件。'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='输入路径，支持多个路径用逗号分隔（如 "a.jsonl,b.jsonl,c_dir"）'
    )
    parser.add_argument(
        '-o', '--output-dir', required=True,
        help='输出目录（合并后的文件将保存在此目录下）'
    )
    parser.add_argument(
        '-n', '--output-name', default='merged.jsonl',
        help='输出文件名（默认为 merged.jsonl）'
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help='如果输入是目录，则递归查找所有子目录中的 .jsonl 文件'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='显示详细处理信息'
    )
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
        print('没有需要合并的文件。', file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f'找到 {len(files)} 个 JSONL 文件：')
        for f in files:
            print(f'  - {f}')

    output_file = Path(args.output_dir) / args.output_name
    merge_jsonl_files(files, output_file, args.verbose)


if __name__ == '__main__':
    main()