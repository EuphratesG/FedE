def read_mapping_file1(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' : ')
            mapping[parts[0]] = int(parts[1])
    return mapping
def read_mapping_file2(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            mapping[parts[0]] = int(parts[1])
    return mapping
def generate_new_mapping(old_to_new_mapping, string_to_old_mapping):
    new_mapping = {}
    for string, old_id in string_to_old_mapping.items():
        if str(old_id) in old_to_new_mapping:
            new_id = old_to_new_mapping[str(old_id)]
            new_mapping[string] = new_id
    return new_mapping


def write_new_mapping(new_mapping, output_file):
    sorted_mapping = sorted(new_mapping.items(), key=lambda x: int(x[1]))
    with open(output_file, 'w') as file:
        for string, new_id in sorted_mapping:
            file.write(f"{string}\t{new_id}\n")


def main(old_to_new_mapping_file, string_to_old_mapping_file, output_file):
    old_to_new_mapping = read_mapping_file1(old_to_new_mapping_file)
    string_to_old_mapping = read_mapping_file2(string_to_old_mapping_file)
    #print(string_to_old_mapping.items())
    new_mapping = generate_new_mapping(old_to_new_mapping, string_to_old_mapping)
    write_new_mapping(new_mapping, output_file)

if __name__ == "__main__":
    main("/home/yvhe/FedE/fb15k-237/新旧编号.txt", "/home/yvhe/FedE/fb15k-237/get_neighbor/entity2id.txt", "string_to_new_mapping.txt")
