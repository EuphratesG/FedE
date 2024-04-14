import random
from collections import Counter
import numpy as np
import pickle
import math

# 从总的文本文件中读取三元组数据
def read_triples_from_file(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            head, relation, tail = map(int, line.strip().split())
            triples.append((head, relation, tail))
    return triples

# 生成数据集
def generate_dataset_with_clients(total_file):
    # 读取总的三元组数据
    triples = read_triples_from_file(total_file)


    # 统计每种关系的三元组数量
    relation_counts = Counter([triple[1] for triple in triples])
    # ？？？这里还是6529个三元组，看来是后面导致的三元组个数丢失   也是46个关系没有丢失
    print(len(relation_counts))
    # 定义每个客户端应该选择的每种关系的数量比例
    client_ratios = [0.3, 0.3, 0.4]  # 可以根据需要调整比例
    
    # 计算每个客户端应该选择的每种关系的数量（全部下取整）
    client_relation_counts = [{relation: int(count * ratio) for relation, count in relation_counts.items()} for ratio in client_ratios]

    # 计算每个关系的差值
    remainder_per_relation = {}
    for relation, count in relation_counts.items():
        total_count_per_relation = sum(counts.get(relation, 0) for counts in client_relation_counts)
        remainder_per_relation[relation] = count - total_count_per_relation

    # 将差值分配给最后一个客户端对应的关系
    last_client_counts = client_relation_counts[-1].copy()
    for relation, remainder in remainder_per_relation.items():
        if remainder > 0:
            last_client_counts[relation] += remainder
        elif remainder < 0:
            last_client_counts[relation] = max(0, last_client_counts[relation] + remainder)

    client_relation_counts[-1] = last_client_counts

    # 检查每个客户端应选择的每种关系的数量
    print("Client relation counts:")
    for i, counts in enumerate(client_relation_counts):
        print(f"Client {i+1}: {counts}")

    print(sum(client_relation_counts[0].values()))
    print(sum(client_relation_counts[1].values()))
    print(sum(client_relation_counts[2].values()))
    # 初始化三个客户端的数据集
    client_data = [[] for _ in range(3)]

    # 随机选择每种关系并分配给客户端
    for relation, count in relation_counts.items():
        selected_counts = [client_relation_counts[i][relation] for i in range(3) if client_relation_counts[i][relation] > 0]  # 每个客户端应该选择的数量
        if len(selected_counts) == 0:
            continue
        selected_counts_sum = sum(selected_counts)
        probabilities = [count / selected_counts_sum for count in selected_counts]  # 每个客户端选择该关系的概率
        #print(probabilities)
        # 累计概率，确保最后一个客户端能够获得所有的剩余概率
        cum_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        print(cum_probabilities)
        cum_probabilities[-1] = 1.0  # 确保最后一个值为1
        selected_clients = []
        #print(count)
        for _ in range(count):
            rand_num = random.random()
            for i, prob in enumerate(cum_probabilities):
                if rand_num <= prob:
                    selected_clients.append(i)
                    break
        #print(selected_clients)# 319
        triples_with_relation = [triple for triple in triples if triple[1] == relation]
        #print(triples_with_relation)
        #print(triples_with_relation)
        #这里gpt写错了一不改按范围取值，二不该用extend而是要用apped
        for i, client_idx in enumerate(selected_clients):
            #print(triples_with_relation[i])
            client_data[client_idx].append(triples_with_relation[i])
            #print(client_data[client_idx])
            #triples_with_relation = triples_with_relation[i:]

    # 将每个客户端的数据集分成train、valid和test, 因为这里data已经是随机了所以是随机取的
    dataset = []
    for data in client_data:
        #print(data)
        # 打乱数据顺序
        random.shuffle(data)
        # 划分数据集
        total_size = len(data)
        train_size = math.ceil(total_size * 0.8)
        valid_size = math.ceil(total_size * 0.1)
        #test_size = total_size - train_size - valid_size
        train_data = data[:train_size]
        valid_data = data[train_size:train_size+valid_size]
        test_data = data[train_size+valid_size:]
        # 重新编号边索引和关系类型
        edge_index_dict = {}
        edge_type_dict = {}
        new_edge_index = []
        new_edge_type = []
        new_edge_index_ori = []
        new_edge_type_ori = []


        #这里没有从0开始,把+1去掉
        for triple in data:
            head, relation, tail = triple
            if head not in edge_index_dict:
                edge_index_dict[head] = len(edge_index_dict)
            if tail not in edge_index_dict:
                edge_index_dict[tail] = len(edge_index_dict)
            if relation not in edge_type_dict:
                edge_type_dict[relation] = len(edge_type_dict)
            new_edge_index.append([edge_index_dict[head], edge_index_dict[tail]])
            new_edge_type.append(edge_type_dict[relation])
            new_edge_index_ori.append([head, tail])
            new_edge_type_ori.append(relation)
        # 整理成字典形式
        client_dataset = {
            'train': {
                'edge_index': np.array(new_edge_index[:train_size], dtype=np.int32),
                'edge_type': np.array(new_edge_type[:train_size], dtype=np.int32),
                'edge_index_ori': np.array(new_edge_index_ori[:train_size], dtype=np.int32),
                'edge_type_ori': np.array(new_edge_type_ori[:train_size], dtype=np.int32)
            },
            'valid': {
                'edge_index': np.array(new_edge_index[train_size:train_size+valid_size], dtype=np.int32),
                'edge_type': np.array(new_edge_type[train_size:train_size+valid_size], dtype=np.int32),
                'edge_index_ori': np.array(new_edge_index_ori[train_size:train_size+valid_size], dtype=np.int32),
                'edge_type_ori': np.array(new_edge_type_ori[train_size:train_size+valid_size], dtype=np.int32)
            },
            'test': {
                'edge_index': np.array(new_edge_index[train_size+valid_size:], dtype=np.int32),
                'edge_type': np.array(new_edge_type[train_size+valid_size:], dtype=np.int32),
                'edge_index_ori': np.array(new_edge_index_ori[train_size+valid_size:], dtype=np.int32),
                'edge_type_ori': np.array(new_edge_type_ori[train_size+valid_size:], dtype=np.int32)
            }
        }
        dataset.append(client_dataset)
    
    return dataset

# 示例总的三元组文件路径
total_file = '/home/yvhe/FedE/umls/get_neighbor/all2id.txt'

# 生成数据集
dataset = generate_dataset_with_clients(total_file)
# 将数据集转置
for client_dataset in dataset:
    for data_type in ['train', 'valid', 'test']:
        client_dataset[data_type]['edge_index'] = client_dataset[data_type]['edge_index'].T
        client_dataset[data_type]['edge_index_ori'] = client_dataset[data_type]['edge_index_ori'].T
        client_dataset[data_type]['edge_type'] = client_dataset[data_type]['edge_type'].T
        client_dataset[data_type]['edge_type_ori'] = client_dataset[data_type]['edge_type_ori'].T
# 将数据集保存为.pkl文件
with open('/home/yvhe/FedE/data/umls-Fed3.pkl', 'wb') as f:
    pickle.dump(dataset, f)
# 打印数据集
for i, client_dataset in enumerate(dataset):
    print(f"Client {i+1} Data:")
    #print(client_dataset['test']['edge_index_ori'][0])
    print(client_dataset['test']['edge_type_ori'][0])
    #print(len(client_dataset))
