from neo4j import GraphDatabase

# Neo4j数据库连接参数
uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j"  # 请替换为你的实际密码
triples = [
    (49, 28, 50),
    (81, 5, 21),
    (115, 43, 107),
    (89, 27, 50),
    (103, 26, 81),
    (22, 13, 128),
    (42, 19, 116),
    (111, 28, 63),
    (20, 40, 81)
]

entity_names = {
    49: "EntityA",
    28: "EntityB",
    50: "EntityC",
    81: "EntityD",
    5: "EntityE",
    21: "EntityF",
    115: "EntityG",
    43: "EntityH",
    107: "EntityI",
    89: "EntityJ",
    27: "EntityK",
    103: "EntityL",
    26: "EntityM",
    22: "EntityN",
    13: "EntityO",
    128: "EntityP",
    42: "EntityQ",
    19: "EntityR",
    116: "EntityS",
    111: "EntityT",
    63: "EntityU",
    20: "EntityV",
    40: "EntityW"
}

# 连接到Neo4j数据库
driver = GraphDatabase.driver(uri, auth=(username, password))

def create_entity(tx, entity_id, entity_name):
    tx.run("MERGE (e:Entity {id: $id, name: $name})", id=entity_id, name=entity_name)

def create_relationship(tx, start_id, relationship_id, end_id):
    tx.run("""
    MATCH (a:Entity {id: $start_id})
    MATCH (b:Entity {id: $end_id})
    MERGE (a)-[:RELATED_TO {id: $rel_id}]->(b)
    """, start_id=start_id, end_id=end_id, rel_id=relationship_id)

# 创建实体和关系
with driver.session() as session:
    for entity_id, entity_name in entity_names.items():
        session.write_transaction(create_entity, entity_id, entity_name)
    
    for start, rel, end in triples:
        session.write_transaction(create_relationship, start, rel, end)

# 关闭数据库连接
driver.close()
