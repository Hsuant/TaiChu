import sys
import os
from transformers import AutoTokenizer

print("=== 开始执行脚本 ===")

tokenizer_dir = "./taichu_tokenizer/"

# 检查目录
if not os.path.isdir(tokenizer_dir):
    print(f"错误：目录 {tokenizer_dir} 不存在")
    sys.exit(1)

print(f"正在尝试加载 tokenizer: {tokenizer_dir}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

# 关键：检查返回值是否为 None（理论上不会发生）
if tokenizer is None:
    print("严重错误：from_pretrained 返回了 None！")
    sys.exit(1)

print(f"成功加载 tokenizer，类型: {type(tokenizer)}")

# 所需特殊 token 列表
special_tokens_to_add = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|/think|>",
    "<unk>"
]

# 现有词汇表
vocab = tokenizer.get_vocab()
existing_tokens = set(vocab.keys())

# 找出缺失的
missing = [t for t in special_tokens_to_add if t not in existing_tokens]

if missing:
    print(f"需要添加的缺失特殊 token: {missing}")
    added = tokenizer.add_tokens(missing, special_tokens=True)
    print(f"实际添加数量: {added} (预期 {len(missing)})")
else:
    print("所有特殊 token 已存在，无需添加")

# 设置 eos, pad, unk
if tokenizer.eos_token != "<|im_end|>":
    tokenizer.eos_token = "<|im_end|>"
    print("已将 eos_token 设为 <|im_end|>")
if tokenizer.pad_token != "<|im_end|>":
    tokenizer.pad_token = "<|im_end|>"
    print("已将 pad_token 设为 <|im_end|>")
if tokenizer.unk_token != "<unk>":
    tokenizer.unk_token = "<unk>"
    print("已将 unk_token 设为 <unk>")

# ========== 输出每个特殊 token 的 token_id ==========
print("\n=== 特殊标记的 token_id 映射 ===")
# 先获取最终词汇表（包含刚添加的 token）
final_vocab = tokenizer.get_vocab()
# 针对我们关心的特殊 token 列表，查找并打印 ID
for token in special_tokens_to_add:
    token_id = final_vocab.get(token)
    if token_id is not None:
        print(f"{token:20s} -> {token_id}")
    else:
        print(f"{token:20s} -> 未找到 (添加可能失败)")

# 额外输出由 tokenizer 属性定义的特殊 token ID
print("\n=== Tokenizer 属性指向的 token_id ===")
attrs = ['bos_token', 'eos_token', 'pad_token', 'unk_token']
for attr in attrs:
    token_str = getattr(tokenizer, attr, None)
    if token_str:
        token_id = final_vocab.get(token_str)
        print(f"{attr:12s} = {token_str:15s} -> id {token_id}")
    else:
        print(f"{attr:12s} = None")

# 最终词汇表大小
final_size = len(tokenizer)
print(f"\n最终词汇表大小: {final_size}")

# 保存 tokenizer
tokenizer.save_pretrained(tokenizer_dir)
print(f"已保存 tokenizer 到 {tokenizer_dir}")

print("=== 脚本执行完成 ===")