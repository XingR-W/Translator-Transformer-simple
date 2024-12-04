import re
import jieba
import torch
from torch.utils.data import Dataset
import json  # 新增：导入json库

# 文件路径
src_file = './data/ch_en_final.txt'
ch_vocab_file = './data/ch_vocab.txt'
en_vocab_file = './data/en_vocab.txt'
output_tensor_file = './data/train_tensor.json' 

# 常量
MAX_LEN = 25  # 假设最大句子长度为25

# 英文分词正则表达式
en_tokenizer = re.compile(r"[\w']+|[.,!?;()\"]")

# 读取词汇表并建立映射
def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # 添加行号以便调试
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                word, idx = line.split('\t')
                vocab[word] = int(idx)
            except ValueError:
                print(f"Skipping line {line_num}: '{line}' - incorrect format")
                continue
    return vocab

# 加载中英文词汇表
ch_vocab = load_vocab(ch_vocab_file)
en_vocab = load_vocab(en_vocab_file)

# 特殊符号索引
PAD_IDX = ch_vocab.get('<pad>', 0)
SOS_IDX = ch_vocab.get('<sos>', 1)
EOS_IDX = ch_vocab.get('<eos>', 2)
UNK_IDX = ch_vocab.get('<unk>', 3)

class TextToTensorDataset(Dataset):
    def __init__(self, src_file, ch_vocab, en_vocab, max_len):
        # 初始化数据
        self.src_lines = []
        self.trg_lines = []
        self.ch_vocab = ch_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len

        # 读取文件并提取中英文token
        with open(src_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 分离中文和英文句子
                ch_sentence, en_sentence = line.strip().split('\t')
                
                # 中文分词
                ch_tokens = jieba.lcut(ch_sentence)
                # 英文分词
                en_tokens = en_tokenizer.findall(en_sentence)

                self.src_lines.append(ch_tokens)
                self.trg_lines.append(en_tokens)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        # 获取分词后的中文和英文句子
        src_tokens = self.src_lines[idx]
        trg_tokens = self.trg_lines[idx]

        # 转换中文tokens到词汇索引
        src_ids = [self.ch_vocab.get(token, UNK_IDX) for token in src_tokens]  # 未知词映射为<unk>
        # 加上开始和结束标记
        src_ids = [SOS_IDX] + src_ids + [EOS_IDX]
        # 填充到固定长度
        src_ids = src_ids[:self.max_len] + [PAD_IDX] * (self.max_len - len(src_ids))

        # 转换英文tokens到词汇索引
        trg_ids = [self.en_vocab.get(token, UNK_IDX) for token in trg_tokens]
        # 加上开始和结束标记
        trg_ids = [SOS_IDX] + trg_ids + [EOS_IDX]
        # 填充到固定长度
        trg_ids = trg_ids[:self.max_len] + [PAD_IDX] * (self.max_len - len(trg_ids))
        
        for idx in src_ids:
            if idx >= len(self.ch_vocab) or idx < 0:
                print(f"Invalid index in src_ids: {idx}")

        for idx in trg_ids:
            if idx >= len(self.en_vocab) or idx < 0:
                print(f"Invalid index in trg_ids: {idx}")

        
        # 将 src_ids 和 trg_ids 转换为张量
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        trg_tensor = torch.tensor(trg_ids, dtype=torch.long)
        return src_tensor, trg_tensor

# 创建数据集
dataset = TextToTensorDataset(src_file, ch_vocab, en_vocab, MAX_LEN)

# 保存至JSON格式张量文件
data = []
for i in range(len(dataset)):
    src_tensor, trg_tensor = dataset[i]
    # 将张量转换为列表再保存
    data.append([src_tensor.tolist(), trg_tensor.tolist()])

# 写入 JSON 文件
with open(output_tensor_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)  # 格式化输出使文件更易读

print(f"张量文件已保存至 {output_tensor_file}")
print(f"Chinese vocab size: {len(ch_vocab)}, English vocab size: {len(en_vocab)}")