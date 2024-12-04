import torch
import torch.nn as nn

# 多头自注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

         # 确保嵌入的大小可以被注意力头数整除，保证可以均匀分配到每个
        assert (self.head_dim * heads == embed_size ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        #映射回原始大小
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        #print('query:',query.shape)
        N = query.shape[0]
        #print('N：',N)#批次
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #print('value_len, key_len, query_len:',value_len, key_len, query_len)
        
         # 将embedding（QKV）切分为 self.heads 个不同的部分（在后两个维度做了reshape）
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        #线性变换，得到新维度
        values = self.values(values)# (N, value_len, heads, head_dim)
        #print("values after rshape:",values.shape)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        #矩阵求和
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        #print('energy:',energy.shape)
        if mask is not None:
            #将mask为0的位置替换成一个非常小的接近负无穷的数，这样做玩softmax归一化后这些值会近乎0
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        #在 energy 张量的第 4 个维度上（key_len 维度）进行 softmax(e的负无穷) 操作,0123s
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        #print('out before:',out.shape)
        #调用该线性层将其映射回原来的大小
        out = self.fc_out(out)
        return out

# 编码器块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 编码器
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, dropout, forward_expansion, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layer = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape # 完整的encoder输入只有一个向量x表示要翻译的句子
        #print(x.shape)
        
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        #print('positions:',positions.shape)
        
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        #print('out:',out.shape)
        
        for layer in self.layer:
            out = layer(out, out, out, mask)
        '''
        for i, layer in enumerate(self.layer):
            # 每经过一层记录显存变化
            allocated_memory_before = torch.cuda.memory_allocated(self.device)
            out = layer(out, out, out, mask)
            allocated_memory_after = torch.cuda.memory_allocated(self.device)
            activation_memory = (allocated_memory_after - allocated_memory_before) / 1024 ** 2  # 转换为 MB
            print(f"Encoder Layer {i+1} - Memory before: {allocated_memory_before / 1024 ** 2:.2f} MB, "
                  f"Memory after: {allocated_memory_after / 1024 ** 2:.2f} MB,"
                  f"Activation Memory: {activation_memory:.2f} MB")
        '''
        return out

# 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

# 解码器
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        '''
        for j, layer in enumerate(self.layers):
            # 每经过一层记录显存变化
            allocated_memory_before = torch.cuda.memory_allocated(self.device)
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            allocated_memory_after = torch.cuda.memory_allocated(self.device)
            activation_memory = (allocated_memory_after - allocated_memory_before) / 1024 ** 2  # 转换为 MB
            
            print(f"Decoder Layer {j+1} - Memory before: {allocated_memory_before / 1024 ** 2:.2f} MB, "
                  f"Memory after: {allocated_memory_after / 1024 ** 2:.2f} MB, "
                  f"Activation Memory: {activation_memory:.2f} MB")
        '''	
        out = self.fc_out(x)
        return out

# Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, dropout, forward_expansion, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src): #把句子中填充的<pad>符号mask掉
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):# 把label中每个单词后面的词mask掉，也就是一个下三角矩阵
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
		)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        #源序列掩码
        src_mask = self.make_src_mask(src)
        #目标序列掩码
        trg_mask = self.make_trg_mask(trg)
        #源序列编码
        enc_src = self.encoder(src, src_mask)
        #源序列编码，掩码，目标序列掩码传入解码器
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

'''    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    embed_size = 512
    num_layers =6 
    forward_expansion =4 
    heads = 8 
    dropout = 0
    max_length = 100
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size, num_layers, forward_expansion, heads, dropout, device,max_length).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
'''