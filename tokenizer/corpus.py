"""
数据集迭代器模块，用于从磁盘高效地流式读取文本数据。
这避免了将整个数据集加载到内存中，适合CPU训练环境。
"""


class CorpusIterator:
    """
    一个可迭代的数据集类，用于逐批生成文本。

    Args:
        files (list): 包含文本文件的路径列表。
        epoch (int): 遍历数据集的次数。
        text_key (str): jsonl格式数据存储字段
    """

    def __init__(self, files, epoch=1, text_key="text"):
        self.files = files
        self.epoch = epoch
        self.text_key = text_key  # JSON 中存储文本的字段名

    def __iter__(self):
        import json
        for _ in range(self.epoch):
            for file_path in self.files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = obj.get(self.text_key)
                            if text and isinstance(text, str):
                                yield text
                        except json.JSONDecodeError:
                            # 兼容普通纯文本行，直接返回原行
                            yield line