def extract_entities(triplets):
    entities = []
    for triplet in triplets:
        entities.append(triplet[0])
        entities.append(triplet[2])
    return entities

def unique_count(entities):
    unique_entities = list(set(entities))
    unique_entities.sort()
    return unique_entities

def assign_new_ids(entities):
    entity_to_id = {}
    for i, entity in enumerate(entities):
        entity_to_id[entity] = i
    return entity_to_id

def write_new_ids_to_file(entity_to_id, output_file):
    with open(output_file, 'w') as file:
        for entity, new_id in entity_to_id.items():
            file.write(f"{entity} : {new_id}\n")

def main(input_file, output_file):
    triplets = []
    with open(input_file, 'r') as file:
        for line in file:
            triplet = line.strip().split()
            triplets.append(triplet)
    
    entities = extract_entities(triplets)
    unique_entities = unique_count(entities)
    entity_to_id = assign_new_ids(unique_entities)
    write_new_ids_to_file(entity_to_id, output_file)

if __name__ == "__main__":
    main("/home/yvhe/FedE/fb15k-237/sampled_output_file.txt", "/home/yvhe/FedE/fb15k-237/新旧编号.txt")
