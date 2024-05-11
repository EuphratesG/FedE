# 读取第一个和最后一个数字的映射关系文件
first_last_mapping = {}
with open("/home/yvhe/FedE/umls/get_neighbor/entity2id.txt", "r") as f:
    for line in f:
        value,key = line.strip().split("\t")
        first_last_mapping[key] = value

# 读取中间数字的映射关系文件
middle_mapping = {}
with open("/home/yvhe/FedE/umls/get_neighbor/relation2id.txt", "r") as f:
    for line in f:
        value,key = line.strip().split("\t")
        middle_mapping[key] = value

# 读取三元组文件，替换第一个和最后一个数字为对应文本，并替换中间数字
with open("/home/yvhe/FedE/FB12k237/triplesid.txt", "r") as f:
    with open("umls三元组.txt", "w") as output:
        for line in f:
            triplets = line.strip().split()
            text_triplets = [first_last_mapping[triplets[0]]]  # 替换第一个数字
            for id_number in triplets[1:-1]:  # 替换中间数字
                text_triplets.append(middle_mapping[id_number])
            text_triplets.append(first_last_mapping[triplets[-1]])  # 替换最后一个数字
            output.write("\t".join(text_triplets) + "\n")
