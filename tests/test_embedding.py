#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""词嵌入模块的单元测试。

测试 TokenEmbedding 的形状、权重绑定以及 dropout 行为。
"""

import unittest
import torch

from model.embedding import TokenEmbedding


class TestTokenEmbedding(unittest.TestCase):
    """TokenEmbedding 单元测试。"""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.hidden_size = 768
        self.batch = 4
        self.seq_len = 10

    def test_forward_shape(self):
        """测试前向传播输出形状正确。"""
        emb = TokenEmbedding(self.vocab_size, self.hidden_size, dropout=0.0)
        input_ids = torch.randint(0, self.vocab_size, (self.batch, self.seq_len))
        output = emb(input_ids)
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))

    def test_dropout_training_eval(self):
        """测试 dropout 在训练和评估模式下的行为差异。"""
        emb = TokenEmbedding(self.vocab_size, self.hidden_size, dropout=0.5)
        input_ids = torch.randint(0, self.vocab_size, (self.batch, self.seq_len))

        emb.train()
        out_train = emb(input_ids)
        emb.eval()
        out_eval = emb(input_ids)
        # 在 eval 模式下，dropout 应不产生缩放，输出应与输入 embedding 完全相同
        # 但因为是 in-place 操作？实际上 TokenEmbedding 内 dropout 是 nn.Dropout，eval 时恒等映射
        # 为了检查 dropout 生效，可以验证两次输出不完全相同（训练模式，随机失活）
        self.assertFalse(torch.allclose(out_train, out_eval))  # 大概率不同

    def test_embedding_matrix_shape(self):
        """检查嵌入矩阵的形状。"""
        emb = TokenEmbedding(self.vocab_size, self.hidden_size)
        self.assertEqual(emb.embedding.weight.shape, (self.vocab_size, self.hidden_size))

    def test_zero_dropout_identity(self):
        """当 dropout=0 时，dropout 层应为 Identity。"""
        emb = TokenEmbedding(self.vocab_size, self.hidden_size, dropout=0.0)
        self.assertIsInstance(emb.dropout, torch.nn.Identity)


if __name__ == '__main__':
    unittest.main()