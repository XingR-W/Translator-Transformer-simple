import opencc

# 创建OpenCC对象，用于繁体到简体的转换
cc = opencc.OpenCC('t2s')  # 't2s'表示繁体转简体

# 输入文件路径
input_file = './data/cmn.txt'  # 假设您的输入文件名是en_ch.txt
output_file = './data/cmn_final.txt'  # 输出文件名

# 读取并处理文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 使用制表符分隔英文和中文
        english, traditional_chinese = line.strip().split('\t')
        # 将繁体中文转换为简体中文
        simplified_chinese = cc.convert(traditional_chinese)
        # 将转换后的内容写入输出文件，交换英文和简体中文的位置，并加上引号
        outfile.write(f'{simplified_chinese}\t{english}\n')

print("繁体中文已成功转换为简体中文，且已交换位置，结果保存到新文件中。")
