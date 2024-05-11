import random

# 读取数据文件，并将每行解析为三元组
def read_data(file_path):
    with open(file_path, 'r') as file:
        triples = [line.strip().split() for line in file]
    return triples

# 统计每种关系的出现次数
def count_relations(triples):
    relation_counts = {}
    for triple in triples:
        relation = triple[1]
        if relation in relation_counts:
            relation_counts[relation].append(triple)
        else:
            relation_counts[relation] = [triple]
    return relation_counts

# 抽样
def sample_data(relation_counts, sample_ratio):
    sampled_triples = []
    for relation, triples in relation_counts.items():
        num_samples = int(len(triples) * sample_ratio)
        sampled_triples.extend(random.sample(triples, max(num_samples, 1)))
    return sampled_triples

# 保存抽样结果
def save_sampled_data(sampled_triples, output_file):
    with open(output_file, 'w') as file:
        for triple in sampled_triples:
            file.write(' '.join(triple) + '\n')

# 主函数
def main(input_file, output_file, sample_ratio):
    # 读取数据文件
    triples = read_data(input_file)
    # 统计关系
    relation_counts = count_relations(triples)
    # 抽样
    #print(len(relation_counts))
    sampled_triples = sample_data(relation_counts, sample_ratio)
    # 保存抽样结果
    save_sampled_data(sampled_triples, output_file)

if __name__ == "__main__":
    input_file = "/home/yvhe/FedE/fb15k-237/get_neighbor/all2id.txt"
    output_file = "/home/yvhe/FedE/fb15k-237/sampled_output_file.txt"
    sample_ratio = 0.01  # 抽样比例为1%
    main(input_file, output_file, sample_ratio)
