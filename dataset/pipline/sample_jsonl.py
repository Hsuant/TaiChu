#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL 文件抽样脚本（先过滤长度，再随机抽样）。

优化点：先按长度区间过滤掉不合规的行，再从符合条件的行中随机抽取目标数量的行。
这样可以避免因过滤导致实际抽取量远低于期望值的问题。

使用方法：
    python -m dataset.pipeline.sample_jsonl --input input.jsonl --output output.jsonl --ratio 0.1 --min-length 50 --max-length 4096

可选参数：
    --seed         随机种子（整数）
    --exact        精确按比例取整抽取（默认）
    --no-exact     使用概率抽样（每行独立以 ratio 概率保留）
    --min-length   文本最小长度（字符数），不设置则不限制
    --max-length   文本最大长度（字符数），不设置则不限制
    --field        JSON 中文本字段名，不指定则按整行文本长度计算
    --log-dir      日志输出目录（默认 ./logs），用于保存运行日志及 TensorBoard 事件文件
"""

import argparse
import json
import random
import sys
import os
import logging
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    print("错误：未安装 tqdm 库，请运行 'pip install tqdm' 安装", file=sys.stderr)
    sys.exit(1)

# 导入日志管理器（需确保 utils.logger 在 PYTHONPATH 中）
from utils.logger import LogManager


def get_text_length(line: str, field: Optional[str] = None) -> int:
    """计算一条 JSONL 数据的文本长度。

    若提供 field 参数，尝试解析 JSON 并返回该字段值的字符长度；
    解析失败或字段不存在时回退到整行长度。
    未提供 field 时直接返回去掉换行符的整行长度。

    Args:
        line: JSONL 文件的一行，含结尾换行符。
        field: JSON 对象中文本字段名。默认为 None。

    Returns:
        文本字符长度（整型）。
    """
    if field is None:
        return len(line.rstrip('\n\r'))
    try:
        data = json.loads(line)
        text = data.get(field, '')
        return len(text)
    except (json.JSONDecodeError, TypeError):
        return len(line.rstrip('\n\r'))


def count_lines(file_path: str) -> int:
    """流式统计文件总行数，适用于大文件。

    Args:
        file_path: 文件路径。

    Returns:
        总行数。
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count


def sample_exact(input_file: str,
                 output_file: str,
                 ratio: float,
                 log_manager: LogManager,
                 seed: Optional[int] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 field: Optional[str] = None) -> None:
    """精确抽样：先过滤长度，再从合格行中随机抽取固定数量。

    工作流程：
        1. 统计总行数 N，计算期望抽取量 target_count = int(N * ratio)。
        2. 第一次遍历：统计满足长度条件的行数 M，并记录每行在文件中的字节偏移量。
        3. 若 M < target_count，发出警告并仅抽取 M 行。
        4. 从 M 个候选行中随机不重复选取 target_count 个索引。
        5. 第二次遍历：仅将选中的行写入输出文件。

    Args:
        input_file: 输入 JSONL 文件路径。
        output_file: 抽样结果输出文件路径。
        ratio: 抽样比例，范围 (0, 1)。
        log_manager: 日志管理器实例，用于记录运行状态。
        seed: 随机种子，用于复现抽样结果。
        min_length: 允许的最小文本长度（含）。
        max_length: 允许的最大文本长度（含）。
        field: JSON 中文本字段名；为 None 则按整行长度计算。
    """
    log_manager.info(f"开始精确抽样：输入文件 {input_file}，目标比例 {ratio:.4f}")
    total = count_lines(input_file)
    target_count = int(total * ratio)
    log_manager.info(f"总行数 {total}，期望抽取 {target_count} 行")

    if target_count <= 0:
        log_manager.warning("抽样数量为 0，将生成空文件。")
        open(output_file, 'w', encoding='utf-8').close()
        return

    if seed is not None:
        random.seed(seed)
        log_manager.info(f"设置随机种子为 {seed}")

    # 第一次遍历：找出所有符合长度条件的行偏移量
    candidates = []          # 记录符合条件的行在文件中的字节偏移量
    log_manager.debug("开始第一遍扫描，筛选符合长度条件的行...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin:
        offset = 0
        for line in tqdm(fin, total=total, desc="筛选合格行"):
            length = get_text_length(line, field)
            if (min_length is not None and length < min_length) or \
               (max_length is not None and length > max_length):
                offset += len(line.encode('utf-8'))
                continue
            candidates.append(offset)
            offset += len(line.encode('utf-8'))

    qualified_count = len(candidates)
    actual_sample = min(target_count, qualified_count)
    log_manager.info(f"长度过滤后保留 {qualified_count} 行，实际将抽取 {actual_sample} 行")

    if qualified_count < target_count:
        log_manager.warning(
            f"合格行数 ({qualified_count}) 少于期望抽取量 ({target_count})，"
            "将抽取全部合格行。"
        )

    # 随机选取要保留的偏移量
    selected_offsets = set(random.sample(candidates, actual_sample))
    log_manager.debug(f"已随机选择 {len(selected_offsets)} 个偏移量")

    # 第二次遍历：写入选中行
    kept = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        offset = 0
        for line in tqdm(fin, total=total, desc="写入选中行"):
            if offset in selected_offsets:
                fout.write(line)
                kept += 1
            offset += len(line.encode('utf-8'))

    # 记录最终统计信息
    length_info = []
    if min_length is not None:
        length_info.append(f"最小长度 {min_length}")
    if max_length is not None:
        length_info.append(f"最大长度 {max_length}")
    filter_str = "，".join(length_info) if length_info else "无"
    summary = (f"精确抽样完成：总行数 {total}，目标抽取 {target_count} 行，"
               f"实际写入 {kept} 行（长度过滤: {filter_str}）")
    log_manager.info(summary)
    log_manager.info(f"结果已保存至 {output_file}")


def sample_approximate(input_file: str,
                       output_file: str,
                       ratio: float,
                       log_manager: LogManager,
                       seed: Optional[int] = None,
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None,
                       field: Optional[str] = None) -> None:
    """近似抽样：先过滤长度，再对合格行以概率 ratio 独立抽取。

    工作流程：
        1. 逐行读取，先判断长度是否符合条件。
        2. 若符合，则以概率 ratio 随机决定是否保留。
        3. 写入选中的行。

    Args:
        input_file: 输入 JSONL 文件路径。
        output_file: 抽样结果输出文件路径。
        ratio: 抽样概率，每行独立以此概率保留。
        log_manager: 日志管理器实例，用于记录运行状态。
        seed: 随机种子，用于复现抽样结果。
        min_length: 允许的最小文本长度（含）。
        max_length: 允许的最大文本长度（含）。
        field: JSON 中文本字段名；为 None 则按整行长度计算。
    """
    log_manager.info(f"开始近似抽样：输入文件 {input_file}，抽样概率 {ratio:.4f}")
    total = count_lines(input_file)
    log_manager.info(f"总行数 {total}")

    if seed is not None:
        random.seed(seed)
        log_manager.info(f"设置随机种子为 {seed}")

    kept = 0
    skipped_by_length = 0

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=total, desc="近似抽样"):
            # 先过滤长度
            length = get_text_length(line, field)
            if (min_length is not None and length < min_length) or \
               (max_length is not None and length > max_length):
                skipped_by_length += 1
                continue
            # 再以概率 ratio 保留
            if random.random() < ratio:
                fout.write(line)
                kept += 1

    # 记录最终统计信息
    length_info = []
    if min_length is not None:
        length_info.append(f"最小长度 {min_length}")
    if max_length is not None:
        length_info.append(f"最大长度 {max_length}")
    filter_str = "，".join(length_info) if length_info else "无"
    summary = (f"近似抽样完成：总行数 {total}，长度过滤跳过 {skipped_by_length} 行，"
               f"实际保留 {kept} 行（期望概率 {ratio:.2%}，长度过滤: {filter_str}）")
    log_manager.info(summary)
    log_manager.info(f"结果已保存至 {output_file}")


def main():
    """解析命令行参数并调用相应的抽样函数。"""
    parser = argparse.ArgumentParser(
        description="从 JSONL 文件中按比例随机抽样（先长度过滤，再抽样）")
    parser.add_argument('--input', '-i', required=True,
                        help="输入 JSONL 文件路径")
    parser.add_argument('--output', '-o', required=True,
                        help="输出 JSONL 文件路径")
    parser.add_argument('--ratio', '-r', type=float, required=True,
                        help="抽样比例，例如 0.1 表示抽取 10%%")
    parser.add_argument('--seed', type=int, default=None,
                        help="随机种子（整数），用于复现")
    parser.add_argument('--exact', action='store_true', default=True,
                        help="精确按比例取整抽取（默认），使用 --no-exact 切换为概率抽样")
    parser.add_argument('--no-exact', dest='exact', action='store_false',
                        help="使用概率抽样（近似）")
    parser.add_argument('--min-length', type=int, default=None,
                        help="文本最小长度（字符数），小于该值的行丢弃")
    parser.add_argument('--max-length', type=int, default=None,
                        help="文本最大长度（字符数），大于该值的行丢弃")
    parser.add_argument('--field', type=str, default=None,
                        help="JSON 中文本字段名；不指定则按整行长度计算")
    parser.add_argument('--log-dir', type=str, default='./experiments/logs',
                        help="日志输出目录，默认 ./experiments/logs")
    args = parser.parse_args()

    # 初始化日志管理器
    log_manager = LogManager(
        log_dir=args.log_dir,
        tensorboard=False,
        log_file="sample_jsonl_for_tokenizer_5GB.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        logger_name="Sampler"
    )
    log_manager.info("===== JSONL 抽样脚本启动 =====")
    log_manager.info(f"命令行参数: {vars(args)}")

    # 参数有效性检查
    if not os.path.isfile(args.input):
        log_manager.error(f"输入文件 '{args.input}' 不存在")
        sys.exit(1)

    if not (0 < args.ratio < 1):
        log_manager.error("抽样比例必须在 (0, 1) 之间")
        sys.exit(1)

    # 根据抽样模式执行对应函数
    if args.exact:
        sample_exact(
            input_file=args.input,
            output_file=args.output,
            ratio=args.ratio,
            log_manager=log_manager,
            seed=args.seed,
            min_length=args.min_length,
            max_length=args.max_length,
            field=args.field
        )
    else:
        sample_approximate(
            input_file=args.input,
            output_file=args.output,
            ratio=args.ratio,
            log_manager=log_manager,
            seed=args.seed,
            min_length=args.min_length,
            max_length=args.max_length,
            field=args.field
        )

    log_manager.info("===== 抽样任务完成 =====")
    log_manager.close()


if __name__ == '__main__':
    main()