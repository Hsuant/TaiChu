#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL 文件抽样脚本

从 JSONL 文件中随机抽取指定比例的数据行，保存到新文件。
支持精确抽样（按比例取整，严格控制行数）和近似抽样（每行独立以概率保留）。

使用方法：
    python -m dataset.pipline.sample_jsonl --input input.jsonl --output output.jsonl --ratio 0.1

可选参数：
    --seed       随机种子（整数），用于复现结果
    --exact      是否严格按比例取整（默认 True，若 False 则每行以概率 ratio 独立抽取）
"""

import argparse
import random
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    print("错误：未安装 tqdm 库，请运行 'pip install tqdm' 安装", file=sys.stderr)
    sys.exit(1)


def count_lines(file_path):
    """快速统计文件总行数。

    Args:
        file_path (str): 文件路径。

    Returns:
        int: 文件总行数。
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count


def sample_exact(input_file, output_file, ratio, seed=None):
    """精确抽样：计算应抽取行数，随机选择行号后写入。

    先统计总行数，按比例计算需要保留的行数，然后随机生成这些行的索引集合，
    第二次遍历文件时只写入索引匹配的行。使用 tqdm 显示处理进度。

    Args:
        input_file (str): 输入 JSONL 文件路径。
        output_file (str): 输出 JSONL 文件路径。
        ratio (float): 抽样比例，取值范围 (0,1)。
        seed (int, optional): 随机种子，用于复现结果。默认为 None。
    """
    total = count_lines(input_file)
    sample_size = int(total * ratio)
    if sample_size <= 0:
        print("警告：抽样数量为 0，将生成空文件。")
    if seed is not None:
        random.seed(seed)

    # 生成随机行号索引集
    indices = set(random.sample(range(total), sample_size))

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        # 使用 tqdm 包装文件对象，显示处理进度（已知总行数 total）
        for line_no, line in enumerate(tqdm(fin, total=total, desc="精确抽样")):
            if line_no in indices:
                fout.write(line)
    print(f"抽样完成：从 {total} 行中抽取 {sample_size} 行 (比例 {ratio:.2%})，保存至 {output_file}")


def sample_approximate(input_file, output_file, ratio, seed=None):
    """近似抽样：每行独立以概率 ratio 保留（结果行数接近 ratio * 总行数）。

    先统计总行数（用于进度条显示），然后逐行生成随机数决定是否保留。
    使用 tqdm 显示处理进度。

    Args:
        input_file (str): 输入 JSONL 文件路径。
        output_file (str): 输出 JSONL 文件路径。
        ratio (float): 抽样比例，取值范围 (0,1)。
        seed (int, optional): 随机种子，用于复现结果。默认为 None。
    """
    total = count_lines(input_file)  # 先统计总行数，用于进度条显示
    if seed is not None:
        random.seed(seed)

    kept = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        # 使用 tqdm 包装文件对象，显示处理进度
        for line in tqdm(fin, total=total, desc="近似抽样"):
            if random.random() < ratio:
                fout.write(line)
                kept += 1
    print(f"抽样完成：从 {total} 行中抽取 {kept} 行 (期望比例 {ratio:.2%})，保存至 {output_file}")


def main():
    """命令行入口函数，解析参数并调用对应的抽样函数。"""
    parser = argparse.ArgumentParser(description="从 JSONL 文件中按比例随机抽样")
    parser.add_argument('--input', '-i', required=True, help="输入 JSONL 文件路径")
    parser.add_argument('--output', '-o', required=True, help="输出 JSONL 文件路径")
    parser.add_argument('--ratio', '-r', type=float, required=True,
                        help="抽样比例，例如 0.1 表示抽取 10%% 的数据")
    parser.add_argument('--seed', type=int, default=None, help="随机种子，用于复现")
    parser.add_argument('--exact', action='store_true', default=True,
                        help="精确按比例取整抽取（默认）。若加上 --no-exact 则使用概率抽样")
    parser.add_argument('--no-exact', dest='exact', action='store_false',
                        help="使用概率抽样，结果行数不严格等于比例")
    args = parser.parse_args()

    # 校验输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误：输入文件 '{args.input}' 不存在", file=sys.stderr)
        sys.exit(1)

    # 校验抽样比例范围
    if not (0 < args.ratio < 1):
        print("错误：抽样比例必须在 (0, 1) 之间", file=sys.stderr)
        sys.exit(1)

    # 根据 exact 标志选择抽样方式
    if args.exact:
        sample_exact(args.input, args.output, args.ratio, args.seed)
    else:
        sample_approximate(args.input, args.output, args.ratio, args.seed)


if __name__ == '__main__':
    main()