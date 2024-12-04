# 手动搭建6层Transformer模型的情况记录与其性能、资源利用分析

## 一、环境配置

### 1. 服务器情况

```bash
服务器IP：172.19.202.70
GPU情况：NVIDIA GeForce RTX 3090 24GB
支持CUDA版本：12.2以及12.2之前的版本
conda虚拟环境情况：python3.8+cuda toolkit 12.1+cudnn9.9.5
```

### 2. Python环境

```bash
torch=2.4.0 
torchvision=0.19.0
nltk=3.9.1
numpy=1.24.3
matplotlib=3.7.5
jieba=0.42.1
opencc-python-reimplemented=0.1.7
torchinfo=1.8.0
```

## 二、数据集来源

| **数据集名称**   | **类型**     | **来源**                                                     |
| ---------------- | ------------ | ------------------------------------------------------------ |
| **en_ch.txt**    | 中英短句子对 | *https://github.com/cuicaihao/Annotated-Transformer-English-to-Chinese-Translator* |
| **en_vocab.txt** | 英文词汇表   | 根据数据集手动构建                                           |
| **ch_vocab.txt** | 中文词汇表   | 根据数据集手动构建                                           |

## 三、 代码结构

```bash
Translator-Transformer-simple/
│
├── data                              # 存放数据集文件
│   ├── ch_en_final.txt               # 中文-英文对齐的训练集（简体中文）
│   ├── ch_en_test_final.txt          # 中文-英文对齐的测试集（简体中文）
│   ├── en_ch_test.txt                # 英文-中文对齐的测试集（繁体中文）
│   ├── en_ch.txt                     # 英文-中文对齐的训练数据集（繁体中文）
│   ├── ch_vocab.txt                  # 生成的中文词汇表
│   ├── en_vocab.txt                  # 生成的英文词汇表
│   └── train_tensor.json             # 根据词汇表映射得到的张量训练集（JSON 格式）
│
├── dataset.py                        # 数据集处理脚本，生成TextToTensorDataset类和张量文件
├── document
│   ├── transformer-simple-translator_report.docx  # 本翻译器的搭建、性能分析的报告
│   └── 训练参数资源占用时间线分析.docx                # 对训练过程的显存和参数资源进行时序化、层级化分析
├── generate_vocab.py                 # 分词、生成词汇表的脚本
├── model                             # 存放模型文件，第60轮已具备较好性能，BLEU分数为25.48
│   ├── model_epoch_60.pth            # 训练后的模型权重（第 60 轮）
│
├── model2                            # 备用模型文件，供测试验证用
│   ├── model_epoch_1.pth             # 训练后的模型权重（第 1 轮）
│
├── model.py                          # 手动搭建6层Transformer 模型的代码
├── picture                           # 存放训练过程中的图像文件
│   ├── loss_curve_128.png            # 训练过程中 batch_size=128 的损失曲线图
│   ├── loss_curve_256.png            # 训练过程中 batch_size=256 的损失曲线图
│   ├── loss_curve_32.png             # 训练过程中 batch_size=32 的损失曲线图
│   ├── loss_curve_512.png            # 训练过程中 batch_size=512 的损失曲线图
│   ├── loss_curve_64.png             # 训练过程中 batch_size=64 的损失曲线图
│   └── loss_curve.png                # 训练过程中默认的损失曲线图
│
├── preprocess.py                     # 数据预处理脚本,实现繁体转简体中文和中文-英文对齐
├── __pycache__                       # 存放 Python 编译的字节码文件
│   ├── dataset.cpython-38.pyc        # 编译后的 dataset.py 文件
│   └── model.cpython-38.pyc          # 编译后的 model.py 文件
│
├── train.py                          # 训练模型的脚本
├── translator.py                     # 翻译器的脚本，加载模型并进行推理
├── visual_mem_2024_11_29_07_15_49.html   # 训练过程的profiler可视化内存日志（HTML 格式）
├── visual_mem_2024_11_29_07_15_49.json   # 训练过程的profiler可视化内存日志（JSON 格式）
├── visual_mem_2024_11_29_07_16_29.html   # 训练过程的profiler可视化内存日志（HTML 格式）
├── visual_mem_2024_11_29_07_16_29.json   # 训练过程的profiler可视化内存日志（JSON 格式）
├── visual_mem_2024_11_29_07_41_30.html   # 训练过程的profiler可视化内存日志（HTML 格式）
├── visual_mem_2024_11_29_07_41_30.json   # 训练过程的profiler可视化内存日志（JSON 格式）
├── visual_mem_2024_11_29_07_42_05.html   # 训练过程的profiler可视化内存日志（HTML 格式）
├── visual_mem_2024_11_29_07_42_05.json   # 训练过程的profiler可视化内存日志（JSON 格式）
└── visual_mem_2024_11_28_14_49_48.pickle # 训练过程的snapshot可视化内存日志（Pickle格式）
```

## 四、训练操作

### 1.参数说明

```python
# 主程序
if __name__ == "__main__":
    ch_vocab_file = './data/ch_vocab.txt' #中文词汇表路径
    en_vocab_file = './data/en_vocab.txt' #英文词汇表路径
    ch_en_path = './data/ch_en_final.txt' #训练集路径

    # 加载中英文词汇表
    ch_vocab = load_vocab(ch_vocab_file)
    en_vocab = load_vocab(en_vocab_file) 
    src_vocab_size = len(ch_vocab)  # embedding操作的参数
    trg_vocab_size = len(en_vocab)  # embedding操作的参数
    src_pad_idx = ch_vocab.get('<pad>', 0)  
    trg_pad_idx = en_vocab.get('<pad>', 0)
    embed_size = 512  # 常用256、512...
    num_layers = 6  # transformer模型层数
    forward_expansion = 4  # 扩展隐藏层维度，前馈全连接网络的参数
    heads = 8  # 512可以被8整除，512/8=64
    dropout = 0.1  # 随机化，防止过拟合
    max_length = 25  # 最大序列长度
    num_epochs = 60  # 训练轮次
    learning_rate = 3e-4  # 学习率

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length).to(device)

    # 优化器Adam 优化算法来更新模型的参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器，监测loss在无明显变化时减少学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 损失函数，交叉熵损失
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    dataset = TextToTensorDataset(ch_en_path, ch_vocab, en_vocab, max_len=25)
    # 每批次样本数量为128
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 指定模型保存路径
    save_path = './model'
    # 训练模型并保存
    train_model(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
```



### 2.训练步骤

```bash
cd ./Translator-Transformer-simple
python train.py
```



## 五、运行翻译器

### 1.调用模型

修改translator.py中以下代码，以修改调用的模型，如下面的代码调用model路径下第60轮的模型

```python
model_path = './model/model_epoch_60.pth'
```

### 2.运行翻译器

```bash
cd ./Translator-Transformer-simple
python translator.py
```

### 3.计算BLEU分数

去掉translator.py中下面代码的注释，即可计算模型对测试集的BLEU分数，测试集文件路径位于test_file_path，测试集文件需要预处理为中-英对齐数据集(简体中文)。

```python
#计算 BLEU 分数
calculate_bleu(model, test_file_path, ch_vocab, en_vocab, device)
```

### 4.置信度判断

由于transformer模型会选择token中概率最大的作为输出，因此我们取其概率作为置信度，累加后除以步长，即可得到平均置信度。当平均置信度小于某阙值，则表示模型比较没有信心，则输出“我不知道”。

置信度阙值设置可以在translator.py中下面函数定义修改参数confidence_threshold来设置，默认设置为0.9，具体如下，

```python
# 推理函数
def translate_ch_to_en(model, input_text, ch_vocab, en_vocab, device, max_length=25,confidence_threshold=0.9):
```

是否启用置信度判断可以选择是否注释translator.py中以下代码：

```python
    #置信度判断
    '''
    print(f"平均置信度: {avg_confidence:.2f}")
    #如果置信度低于阈值，返回 "我不知道"
    if avg_confidence < confidence_threshold:
        return "我不知道"
    '''
```

## 六、训练资源利用分析

(具体结果分析见./document/训练参数资源占用时间线分析.docx)

**建议执行下述操作将train.py的训练轮次设置为1，减少运行时间并且不影响资源分析**

```python
num_epochs = 1  # 训练轮次
```

进行好下述相关参数/显存设置后，运行下面命令

```bash
python train.py
```

### 1.参数

#### （1）使用torchinfo

使用时取消train.py中下面代码的注释即可：

```python
   print("模型参数信息：")
   summary(model, verbose=2)
```

注意：这里verbose参数决定输出信息的详尽度，不同的数值代表不同的信息深度。此外，verbose must be either 0 (quiet), 1 (default), or 2 (verbose)，即0代表未启用summary无输出，1代表默认，2代表更详细的输出，如encoder的attention层的参数数目等。

#### （2）使用自定义函数print_layer_params

使用时取消train.py中下面的代码的注释即可：

```python
    #打印模型参数信息
    print_layer_params(model)
```

### 2.显存

#### （1）6层decoder和6层encoder分别消耗的显存大小

取消model.py中Encoder类中forward()下面代码的注释，并注释掉前两行代码

```python
        #for layer in self.layer:
            #out = layer(out, out, out, mask)

        for i, layer in enumerate(self.layer):
            # 每经过一层记录显存变化
            allocated_memory_before = torch.cuda.memory_allocated(self.device)
            out = layer(out, out, out, mask)
            allocated_memory_after = torch.cuda.memory_allocated(self.device)
            activation_memory = (allocated_memory_after - allocated_memory_before) / 1024 ** 2  # 转换为 MB
            print(f"Encoder Layer {i+1} - Memory before: {allocated_memory_before / 1024 ** 2:.2f} MB, "
                  f"Memory after: {allocated_memory_after / 1024 ** 2:.2f} MB,"
                  f"Activation Memory: {activation_memory:.2f} MB")

        return out
```

取消model.py中Decoder类中forward()下面代码的注释，并注释掉前两行代码

```python
        #for layer in self.layers:
           # x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        for j, layer in enumerate(self.layers):
            # 每经过一层记录显存变化
            allocated_memory_before = torch.cuda.memory_allocated(self.device)
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            allocated_memory_after = torch.cuda.memory_allocated(self.device)
            activation_memory = (allocated_memory_after - allocated_memory_before) / 1024 ** 2  # 转换为 MB
            
            print(f"Decoder Layer {j+1} - Memory before: {allocated_memory_before / 1024 ** 2:.2f} MB, "
                  f"Memory after: {allocated_memory_after / 1024 ** 2:.2f} MB, "
                  f"Activation Memory: {activation_memory:.2f} MB")
        
        out = self.fc_out(x)
        return out
```

#### （2）显存与时序分析

##### ①SnapshotAPI可视化显存

去掉下述代码的注释，并调用train_model()函数：

```python
    # Snapshot部分，Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)
    
    # 训练模型并保存
    train_model(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
    # 训练模型并保存，启用Profiler显存可视化
    #train_model_profiler(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
    
    #Snapshot部分
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)

```

可视化结果分析：

将上述代码生成的pickle格式文件拖拽到浏览器（如chrome等）的下面网址中:*https://pytorch.org/memory_viz* （需要挂梯子），进行详细的显存数据分析：

![](./1.png)

##### ②Profiler中显存可视化使用

train.py中注释t掉rain_model的调用，而使用train_model_profiler

```python
    # 训练模型并保存
    #train_model(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
    # 训练模型并保存，启用Profiler显存可视化
    train_model_profiler(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
```

查看生成的Html文件即可，json文件则跟踪和记录了程序在 GPU 和 CPU 上的计算过程，例如一些堆栈信息。

