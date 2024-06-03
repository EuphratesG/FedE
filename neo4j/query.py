from py2neo import Graph, NodeMatcher

# Neo4j数据库连接参数
uri = "bolt://137.132.92.43:7687"
username = "neo4j"
password = "Xiao001112"  # 请替换为你的实际密码

# 连接到Neo4j数据库
graph = Graph(uri, auth=(username, password))

def get_related_triples(graph, entity_id):
    query = """
    MATCH (a)-[r]->(b)
    WHERE a.id = $entity_id
    RETURN a AS start_node, r AS relationship, b AS end_node
    UNION
    MATCH (c)-[r]->(a)
    WHERE a.id = $entity_id
    RETURN c AS start_node, r AS relationship, a AS end_node
    """
    return graph.run(query, entity_id=entity_id).data()

# 示例实体ID
entity_id = 0  # 请替换为实际的实体ID

# 获取与实体相关的所有三元组
related_triples = get_related_triples(graph, entity_id)

for triple in related_triples:
    print(triple)
