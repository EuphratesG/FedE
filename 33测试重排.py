import pandas as pd
import numpy as np

# # 从pkl文件加载DataFrame
# loaded_df = pd.read_pickle('data/FB15k237-Fed3.pkl')
# a=np.sort(np.unique(loaded_df[0]['train']['edge_type']))
# b=np.sort(np.unique(loaded_df[0]['train']['edge_type_ori']))
# # 显示加载的DataFrame
# print(a)
# print(b)
# c=np.sort(np.unique(loaded_df[0]['train']['edge_index']))
# d=np.sort(np.unique(loaded_df[0]['train']['edge_index_ori']))
# print(c)
# print(d)

# 读取entityid
# 打开文本文件
file_path = 'FB12k237/entity2id.txt'  # 请替换为你的文件路径
with open(file_path, 'r') as file:
    # 读取每一行并分割字符串和整数
    data = [line.strip().split('\t')[0] for line in file]


# 显示读取的数据
print(len(data))


# 读取entity2text
# 打开文本文件
file_path = 'FB12k237/entity2textlong.txt'  # 请替换为你的文件路径
with open(file_path, 'r') as file:
    # 读取每一行并分割字符串和整数
    data2 = [line.strip().split('\t') for line in file]


# 显示读取的数据
print(len(data2))

# # 转换为集合并计算差集
# difference_set = set(data2) - set(data)


# # 将差集转换为列表
# difference_list = list(difference_set)

# # 打印多出的元素
# print("多出的元素:", len(difference_list))


# 去除原数组中多余的元素
data2 = [item for item in data2  if item[0] in data]
print(len(data2))


# 创建映射字典，将元素映射到它在第一个数组中的索引位置
index_mapping = {element: index for index, element in enumerate(data)}

# 根据映射对第二个数组进行排序
sorted_data2 = sorted(data2, key=lambda x: data.index(x[0]))
#print(sorted_data2)
# 打开一个文本文件以写入模式
with open("FB12k237/entityDescriptionLong.txt", "w") as file:
    # 将列表中的每个元素写入文件
    for item in sorted_data2:
        file.write("%s\n" % ('\t'.join(item)))


# 检查数组长度是否相同
if len(sorted_data2) == len(data):
    # 逐个元素比较
    for element1, element2 in zip(sorted_data2, data):
        if element1[0] != element2:
            print("数组不完全一致。")
            break
    else:
        print("数组完全一致。")
else:
    print("数组不完全一致（长度不同）。")


# array1 = [["apple", "new1"], ["banana", "new2"], ["orange", "new3"], ["grape", "new4"], ["kiwi", "new5"]]
# array2 = ["kiwi", "banana", "apple", "grape", "pear", "melon"]

# # 去除原数组中多余的元素
# result_array = [item for item in array1 if item[0] in array2]

# # 打印
# # 去除多余元素后的数组
# print("去除多余元素后的数组:", result_array)