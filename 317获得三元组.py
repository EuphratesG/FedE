# 打开映射文件并读取映射关系
mapping = {}
with open('FB12k237/entityDescription.txt', 'r') as f:
    for line_num, line in enumerate(f):
        parts = line.split(", ")
        parts = [part.rstrip() for part in parts]
        mapping[line_num] = parts[1]
print(mapping.get(3))
# 打开输出文件
with open('FB12k237/triplesid.txt', 'r') as f:
    lines = f.readlines()

# 替换每行中的数字id为对应的编码
for i in range(len(lines)):
    elements = lines[i].split()
    for j in range(len(elements)):
        # print(elements[j])
        # print(elements[j].isdigit())
        if(j==1) :
            continue
        if elements[j].isdigit():
            elements[j] = int(elements[j])
            elements[j] = mapping.get(elements[j], elements[j])  # 如果找不到对应的编码，则保持原始id不变
            #print(elements[j])
    lines[i] = ', '.join(elements)  # 重新组合成一行
#print(lines)
# 写入结果到新文件
with open('FB12k237/result.txt', 'w') as f:
        # 遍历每个元素
    for item in lines:
        f.writelines(item+'\n')
