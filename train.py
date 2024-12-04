import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from model import Transformer
from dataset import TextToTensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd.profiler import record_function
from torchinfo import summary



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

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"[{hours:02}:{minutes:02}:{seconds:02}]"

def print_layer_params(model):
    """
    打印模型每一层的参数数量。
    """
    print("\n模型每层的参数数量：")
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {num_params}个参数")
    print(f"总参数数量: {total_params}个\n")

# 打印 GPU 的显存和利用率信息
def print_gpu_utilization():
    allocated_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved() / 1024**2    # 转换为 MB
    max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"当前显存占用: {allocated_memory:.2f} MB, 保留显存: {reserved_memory:.2f} MB, 历史最大显存使用: {max_allocated_memory:.2f} MB")

# 训练函数
def train_model(model, data_loader, num_epochs, device, optimizer, criterion, save_path, val_loader=None):
    model.train()
    losses = []
    start_time = time.time()  # 总计时器的开始时间
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        # 每个 epoch 开始时打印显存使用情况
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        print_gpu_utilization()  # 打印 GPU 显存和利用率信息
        
        for batch_idx, batch in enumerate(data_loader):
            src, trg = batch  # 解包元组
            input = src.to(device)
            target = trg.to(device)

            output = model(input, target[:, :-1])  # 模型输出
            output = output.reshape(-1, output.shape[2])  # 调整输出形状
            target = target[:, 1:].reshape(-1)  # 目标不包含<sos>标记
            
            optimizer.zero_grad()
            loss = criterion(output, target)
            losses.append(loss.item())  # 将每个batch的损失值添加到losses列表
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()
            
            # 每10个batch打印一次loss
            if batch_idx % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"{format_time(elapsed_time)} Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        
        # 打印当前周期的平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(data_loader)}")
        print(f"--- Epoch {epoch + 1} End ---")
        print_gpu_utilization()

        # 保存模型参数
        torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

    total_training_time = time.time() - start_time
    print(f"Training completed in {format_time(total_training_time)}")
    
    # 在训练结束后绘制损失曲线并保存到./picture目录
    plot_loss_curve(losses)


# 训练函数--启动Profiler显存可视化 
def train_model_profiler(model, data_loader, num_epochs, device, optimizer, criterion, save_path, val_loader=None):
    model.train()
    losses = []
    start_time = time.time()  # 总计时器的开始时间

    # 启动 PyTorch Profiler 来收集性能数据
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=2),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,  # 训练完成后处理并导出 trace 数据
    ) as prof:
        for epoch in range(num_epochs):
            epoch_loss = 0
            # 每个 epoch 开始时打印显存使用情况
            print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
            print_gpu_utilization()  # 打印 GPU 显存和利用率信息
            
            for batch_idx, batch in enumerate(data_loader):
                prof.step()
                print(f"---- Batch {batch_idx} ----")
                src, trg = batch  # 解包元组
                input = src.to(device)
                target = trg.to(device)

                # 标记前向传播操作
                with record_function("## forward ##"):
                    output = model(input, target[:, :-1])  # 模型输出
                    output = output.reshape(-1, output.shape[2])  # 调整输出形状
                    target = target[:, 1:].reshape(-1)  # 目标不包含<sos>标记

                optimizer.zero_grad()

                # 标记损失计算和反向传播
                with record_function("## backward ##"):
                    loss = criterion(output, target)
                    losses.append(loss.item())  # 将每个batch的损失值添加到losses列表
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                # 标记优化器步骤
                with record_function("## optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()

                # 每10个batch打印一次loss
                if batch_idx % 50 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"{format_time(elapsed_time)} Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            # 打印当前周期的平均损失
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(data_loader)}")
            print(f"--- Epoch {epoch + 1} End ---")
            print_gpu_utilization()

            # 保存模型参数
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

        total_training_time = time.time() - start_time
        print(f"Training completed in {format_time(total_training_time)}")

        # 在训练结束后绘制损失曲线并保存到./picture目录
        plot_loss_curve(losses)


# 绘制损失曲线并保存
def plot_loss_curve(losses):
    # 创建存储图片的目录
    if not os.path.exists('./picture'):
        os.makedirs('./picture')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 保存图像到./picture目录
    plt.savefig('./picture/loss_curve.png')
    print("Loss curve saved as ./picture/loss_curve.png")
    plt.close()  # 关闭图像以释放内存


def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")

# 主程序
if __name__ == "__main__":
    ch_vocab_file = './data/ch_vocab.txt'
    en_vocab_file = './data/en_vocab.txt'
    ch_en_path = './data/ch_en_final.txt'

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
    num_epochs =1  # 训练轮次
    learning_rate = 3e-4  # 学习率

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length).to(device)
    
    #打印模型参数信息
    #print_layer_params(model)
    
    # 输出模型信息
    #print("模型参数信息：")
    #summary(model, verbose=2)

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
    save_path = './model2'
    
    
    # Snapshot部分，Start recording memory snapshot history
    #torch.cuda.memory._record_memory_history(max_entries=100000)
    
    # 训练模型并保存
    train_model(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
    # 训练模型并保存，启用Profiler显存可视化
    #train_model_profiler(model, data_loader, num_epochs, device, optimizer, criterion, save_path)
    
    #Snapshot部分
    '''
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)
    '''