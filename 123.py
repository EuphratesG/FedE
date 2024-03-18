# 读取映射文件并创建映射字典
mapping = {}
with open('FB12k237/relationDescription.txt', 'r') as f:
    for line_num, line in enumerate(f):
        parts = line.split(", ")
        parts = [part.rstrip() for part in parts]
        mapping[line_num] = parts[1]

# 打开包含要替换数字的文件
with open('FB12k237/result.txt', 'r') as f:
    lines = f.readlines()

# 遍历文件中的每一行，将数字替换为对应的字符串
with open('FB12k237/output_file.txt', 'w') as f:
    for line in lines:
        parts = line.split(", ")
        name = parts[0]
        number = int(parts[1].strip())
        profession = parts[2].strip()
        
        # 如果数字在映射中存在，则替换为对应的字符串，否则保持原样
        if number in mapping:
            number = mapping[number]

        # 将替换后的内容写入到输出文件中
        f.write(f"{name}, {number}, {profession}\n")
