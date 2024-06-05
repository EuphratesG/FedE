from py2neo import Graph, Node, Relationship
import numpy as np
import os
# Neo4j数据库连接参数
uri = "bolt://localhost:7687"
username = "neo4j"
password = os.environ.get("neo4j_PASSWORD")  # 请替换为你的实际密码

# 连接到Neo4j数据库
graph = Graph(uri, auth=(username, password))

def check_triple_exists(graph, start_node_id, relationship_id, end_node_id):
    query = """
    MATCH (a)-[r]->(b)
    WHERE a.id = $start_node_id AND r.id = $relationship_id AND b.id = $end_node_id
    RETURN a, r, b
    """
    result = graph.run(query, start_node_id=start_node_id, relationship_id=relationship_id, end_node_id=end_node_id).data()
    return len(result) > 0

def get_neighbors_triples(graph, start_node_id, relationship_id, end_node_id):
    query = """
    MATCH (a)-[r]->(b)
    WHERE a.id = $start_node_id AND r.id = $relationship_id AND b.id = $end_node_id
    WITH a, b
    MATCH (a)-[r1]->(n)
    RETURN a AS start_node, r1 AS relationship, n AS end_node
    UNION
    MATCH (n)-[r2]->(a)
    RETURN n AS start_node, r2 AS relationship, a AS end_node
    UNION
    MATCH (b)-[r3]->(n)
    RETURN b AS start_node, r3 AS relationship, n AS end_node
    UNION
    MATCH (n)-[r4]->(b)
    RETURN n AS start_node, r4 AS relationship, b AS end_node
    """
    return graph.run(query, start_node_id=start_node_id, relationship_id=relationship_id, end_node_id=end_node_id).data()

def add_triples_to_db(graph, triples, entity_names, relationship_names):
    for triple in triples:
        start_node_id, relationship_id, end_node_id = triple
        start_node_name = entity_names.get(start_node_id, "Unknown")
        end_node_name = entity_names.get(end_node_id, "Unknown")
        relationship_name = relationship_names.get(relationship_id, "UNKNOWN_RELATION")

        start_node = Node(start_node_name, id=start_node_id, name=start_node_name)
        end_node = Node(end_node_name, id=end_node_id, name=end_node_name)
        relationship = Relationship(start_node, relationship_name, end_node, id=relationship_id)

        # 检查是否存在重复的三元组
        if not check_triple_exists(graph, start_node_id, relationship_id, end_node_id):
            graph.merge(start_node, start_node_name, "id")
            graph.merge(end_node, end_node_name, "id")
            graph.merge(relationship)

def process_triple(graph, client_data, start_node_id, relationship_id, end_node_id, entity_names, relationship_names):
    if check_triple_exists(graph, start_node_id, relationship_id, end_node_id):
        triples = get_neighbors_triples(graph, start_node_id, relationship_id, end_node_id)
        return [(triple['start_node']['id'], triple['relationship']['id'], triple['end_node']['id']) for triple in triples]
    else:
        for triple in client_data:
            if (triple[0] == start_node_id) and (triple[1] == relationship_id) and (triple[2] == end_node_id):
                neighbors = [t for t in client_data if t[0] == start_node_id or t[2] == start_node_id or t[0] == end_node_id or t[2] == end_node_id]
                add_triples_to_db(graph, neighbors, entity_names, relationship_names)
                return neighbors
    return []

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

# 示例客户端数据，替换为实际数据
all_client_data = {
    0: np.array([
        [0, 1, 1],
        [1, 1, 2],
        [2, 1, 3]
    ], dtype=object)
}

# 示例实体ID和关系类型
start_node_id = 0
relationship_id = 1
end_node_id = 1

# 处理三元组
client_idx = 0  # 替换为实际的客户端索引
related_triples = process_triple(graph, all_client_data[client_idx], start_node_id, relationship_id, end_node_id, entity_names, relationship_names)

# 输出格式化的相关三元组
for triple in related_triples:
    print(f"{triple[0]} {triple[1]} {triple[2]}")
