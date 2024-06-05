from neo4j import GraphDatabase
import os
# Neo4j数据库连接参数
uri = "bolt://localhost:7687"
username = "neo4j"
print(os.environ.get("neo4j_PASSWORD"))
password = os.environ.get("neo4j_PASSWORD")  # 请替换为你的实际密码

# 连接到Neo4j数据库
driver = GraphDatabase.driver(uri, auth=(username, password))

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# 删除所有节点和关系
with driver.session() as session:
    session.write_transaction(clear_database)

# 关闭数据库连接
driver.close()
