import torch
from model import Transformer  # 导入手动实现的 Transformer 模型
from dataset import TextToTensorDataset  # 导入数据集类，用于处理输入文本
import jieba
from nltk.translate.bleu_score import corpus_bleu  # 计算BLEU分数
import re
import torch.nn.functional as F  # 用于 softmax 计算
from collections import defaultdict


# 加载词汇表
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

# 加载模型
def load_model(model_path, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device):
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
        embed_size=512, num_layers=6, forward_expansion=4, heads=8,
        dropout=0.1, device=device, max_length=25
    ).to(device)
    model.load_state_dict(torch.load(model_path,weights_only=False))
    model.eval()
    return model

# 将文本转换为模型输入的张量
def text_to_tensor(text, vocab, device):
    # 使用jieba进行中文分词
    tokens = list(jieba.cut(text))  # jieba分词，返回一个生成器，转换为list
    #indices = [vocab.get(token, vocab.get('<unk>')) for token in tokens]
    sos_index = vocab.get('<sos>', None)
    indices = [sos_index] + [vocab.get(token, vocab.get('<unk>')) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)  

# 将模型输出的索引序列转换为文本
def tensor_to_text(tensor, vocab):
    index_to_token = {index: token for token, index in vocab.items()}
    return ' '.join([index_to_token.get(idx.item(), '<unk>') for idx in tensor if index_to_token.get(idx.item()) != '<eos>'])


# 推理函数
def translate_ch_to_en(model, input_text, ch_vocab, en_vocab, device, max_length=25,confidence_threshold=0.9):
    """
    翻译函数：当置信度低于阈值时返回 "我不知道"
    """
    src_tensor = text_to_tensor(input_text, ch_vocab, device)
    #print("src_tensor:", src_tensor) 
    output_indices = [en_vocab.get('<sos>')]
    confidences = []
    for _ in range(max_length):
        trg_tensor = torch.tensor([output_indices], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        
        #next_token = output.argmax(dim=-1)[:, -1].item()
        #output_indices.append(next_token)
        # 获取当前时间步的输出分布
        logits_last_step = output[:, -1, :]  # 取最后一个时间步的 logits
        probabilities = F.softmax(logits_last_step, dim=-1)  # 转化为概率分布
        next_token = probabilities.argmax(dim=-1).item()  # 获取概率最大的索引
        
        # 记录置信度（最大概率值）
        confidence = probabilities[0, next_token].item()
        confidences.append(confidence)
        
        output_indices.append(next_token)
        
        if next_token == en_vocab.get('<eos>'):
            break
    # 平均置信度
    avg_confidence = sum(confidences) / len(confidences)
    #置信度判断
    '''
    print(f"平均置信度: {avg_confidence:.2f}")
    #如果置信度低于阈值，返回 "我不知道"
    if avg_confidence < confidence_threshold:
        return "我不知道"
    '''
    result_tensor = torch.tensor(output_indices[1:], dtype=torch.long)
    print("Final Output Indices Tensor {temp}:", result_tensor)  # 打印最终输出张量
    return tensor_to_text(result_tensor, en_vocab)

# 计算BLEU分数
def calculate_bleu(model, test_file_path, ch_vocab, en_vocab, device):
    references = []  # 参考翻译（真实标签）
    hypotheses = []  # 模型生成的翻译

    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 读取中英文句子对
            ch_sentence, en_sentence = line.strip().split('\t')

            # 中文分词，使用 jieba 分词
            ch_tokens = list(jieba.cut(ch_sentence))  # 中文分词

            # 英文分词，使用正则表达式分词
            en_tokens = re.findall(r'\w+|[^\w\s]', en_sentence)  # 英文分词

            references.append([en_tokens])  # 真实翻译，按空格拆分为词语列表
            
            output_text = translate_ch_to_en(model, ch_sentence, ch_vocab, en_vocab, device)
            
            # 对模型生成的英文翻译进行分词
            hypothesis_tokens = re.findall(r'\w+|[^\w\s]', output_text)  # 英文分词
            
            hypotheses.append(hypothesis_tokens)  # 模型输出翻译，按空格拆分为词语列表

    # 计算 BLEU 分数
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU Score: {bleu_score * 100:.2f}")


# 主程序
if __name__ == "__main__":
    ch_vocab_file = './data/ch_vocab.txt'
    en_vocab_file = './data/en_vocab.txt'
    model_path = './model/model_epoch_60.pth'
    test_file_path = './data/ch_en_test_final.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # 加载词汇表和模型
    ch_vocab = load_vocab(ch_vocab_file)
    en_vocab = load_vocab(en_vocab_file) 
    src_pad_idx = ch_vocab.get('<pad>', 0)
    trg_pad_idx = en_vocab.get('<pad>', 0)
    model = load_model(model_path, len(ch_vocab), len(en_vocab), src_pad_idx, trg_pad_idx, device)

    # 计算 BLEU 分数
    #calculate_bleu(model, test_file_path, ch_vocab, en_vocab, device)

    print("输入中文句子，输入'quit'退出：")
    while True:
        input_text = input("中文: ")
        if input_text.strip().lower() == 'quit':
            break

        # 进行翻译并输出结果
        output_text = translate_ch_to_en(model, input_text, ch_vocab,en_vocab, device)
        print("转换为英文: ", output_text)
