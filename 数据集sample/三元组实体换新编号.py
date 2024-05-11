def read_mapping_file(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as file:
        for line in file:
            old_id, new_id = line.strip().split(' : ')
            mapping[old_id] = int(new_id)
    return mapping
def read_triplets_from_file(filename):
    triplets = []
    with open(filename, 'r') as file:
        for line in file:
            triplet = line.strip().split()
            triplets.append(triplet)
    return triplets

def replace_entities(triplets, entity_to_id):
    replaced_triplets = []
    for triplet in triplets:
        replaced_triplet = [entity_to_id[triplet[0]], triplet[1], entity_to_id[triplet[2]]]
        replaced_triplets.append(replaced_triplet)
    return replaced_triplets

def write_triplets_to_file(triplets, output_file):
    with open(output_file, 'w') as file:
        for triplet in triplets:
            file.write(' '.join(map(str, triplet)) + '\n')


def main(input_file, output_file, entity_to_id):
    triplets = read_triplets_from_file(input_file)
    replaced_triplets = replace_entities(triplets, entity_to_id)
    write_triplets_to_file(replaced_triplets, output_file)

if __name__ == "__main__":
    mapping_file = "/home/yvhe/FedE/fb15k-237/新旧编号.txt"  # 修改为你的映射文件的路径
    entity_to_id = read_mapping_file(mapping_file)
    #print(entity_to_id)
    main("/home/yvhe/FedE/fb15k-237/sampled_output_file.txt", "output.txt", entity_to_id)