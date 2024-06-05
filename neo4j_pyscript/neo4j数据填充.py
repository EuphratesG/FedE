from py2neo import Graph, Node, Relationship
import os
# Neo4j数据库连接参数
uri = "bolt://localhost:7687"
username = "neo4j"
password = os.environ.get("neo4j_PASSWORD")  # 请替换为你的实际密码

# 连接到Neo4j数据库
graph = Graph(uri, auth=(username, password))

# 读取实体ID和实体名字对应的文本文件
entity_names = {}
with open('/home/yvhe/511FedE/umls/get_neighbor/entity2id.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        entity_names[int(parts[-1])] = ' '.join(parts[:-1])

# 读取关系ID和关系名字对应的文本文件
relationship_names = {}
with open('/home/yvhe/511FedE/umls/get_neighbor/relation2id.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        relationship_names[int(parts[-1])] = ' '.join(parts[:-1])

# 读取三元组文件
triples = []
with open('/home/yvhe/511FedE/umls/get_neighbor/all2id.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        triples.append((int(parts[0]), int(parts[1]), int(parts[2])))

# 创建实体和关系
for entity_id, entity_name in entity_names.items():
    entity_node = Node(entity_name, id=entity_id, name=entity_name)
    graph.merge(entity_node, entity_name, "id")

for start, rel, end in triples:
    start_node = graph.nodes.match(id=start).first()
    end_node = graph.nodes.match(id=end).first()
    if start_node and end_node:
        rel_name = relationship_names.get(rel, "UNKNOWN_RELATION")
        relationship = Relationship(start_node, rel_name, end_node, id=rel)
        graph.merge(relationship)
