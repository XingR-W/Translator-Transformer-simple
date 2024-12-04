import itertools
from collections import Counter
import jieba  # 引入jieba库进行中文分词
import re  # 引入re模块处理英文标点符号
import itertools 
from collections import Counter
import jieba  # 引入jieba库进行中文分词
import re  # 引入re模块处理英文标点符号

# 初始化一些参数
MIN_FREQ = 1  # 最小词频阈值
src_file1 = './data/ch_en_final.txt'  # 第一个输入文件路径
src_file2 = './data/ch_en_test_final.txt'  # 第二个输入文件路径
ch_vocab_path = './data/ch_vocab.txt'  # 中文词汇表输出路径
en_vocab_path = './data/en_vocab.txt'  # 英文词汇表输出路径

# 定义词汇表初始状态（包括特殊token）
ch_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
en_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
ch_tot, en_tot = 4, 4  # 初始索引号，从4开始

# 初始化两个列表来存储所有的中文和英文token
ch_tokens = []
en_tokens = []

# 正则表达式，用于匹配英文单词和标点符号
en_tokenizer = re.compile(r"[\w']+|[.,!?;()\"]")

# 读取第一个文件并提取中英文token
for src_file in [src_file1]:
    with open(src_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 分离中文和英文句子
            ch_sentence, en_sentence = line.strip().split('\t')
            
            # 中文分词：使用jieba进行中文分词
            ch_tokens.extend(jieba.lcut(ch_sentence))  # 使用 jieba 进行中文分词
            # 英文分词：使用正则表达式分词，处理标点符号
            en_tokens.extend(en_tokenizer.findall(en_sentence))  # 英文token，处理标点符号

# 统计词频
ch_vocab_counter = Counter(ch_tokens)
en_vocab_counter = Counter(en_tokens)

# 按照词频降序排列，频率大的排在前面
sorted_ch_vocab = sorted(ch_vocab_counter.items(), key=lambda x: x[1], reverse=True)
sorted_en_vocab = sorted(en_vocab_counter.items(), key=lambda x: x[1], reverse=True)

# 构建中文词汇表，添加频率大于或等于 MIN_FREQ 的词
for token, freq in sorted_ch_vocab:
    if token.strip() and freq >= MIN_FREQ:  # 检查 token 是否为空或仅包含空白字符
        ch_vocab[token] = ch_tot
        ch_tot += 1

# 构建英文词汇表，添加频率大于或等于 MIN_FREQ 的词
for token, freq in sorted_en_vocab:
    if token.strip() and freq >= MIN_FREQ:  # 检查 token 是否为空或仅包含空白字符
        en_vocab[token] = en_tot
        en_tot += 1

# 将中文词汇表写入文件
with open(ch_vocab_path, 'w', encoding='utf-8') as ch_outfile:
    for token, idx in ch_vocab.items():
        ch_outfile.write(f"{token}\t{idx}\n")

# 将英文词汇表写入文件
with open(en_vocab_path, 'w', encoding='utf-8') as en_outfile:
    for token, idx in en_vocab.items():
        en_outfile.write(f"{token}\t{idx}\n")

print("中文词汇表已保存到", ch_vocab_path)
print("英文词汇表已保存到", en_vocab_path)
