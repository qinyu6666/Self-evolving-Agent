from neo4j import GraphDatabase
from typing import Optional


class KnowledgeGraph:
    """
    知识图谱轻量封装
    默认标签 :Concept，属性 {name, key}
    关系类型可自定义，默认 HAS_EXAMPLE
    """

    def __init__(
        self,
        uri: str,
        user: str = "neo4j",
        password: str = "password",
        node_label: str = "Concept",
        rel_type: str = "HAS_EXAMPLE",
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_label = node_label
        self.rel_type = rel_type
        # 创建唯一约束（仅第一次生效）
        self._init_constraints()

    # -------------------- 内部工具 --------------------
    def _init_constraints(self):
        """保证 name 唯一"""
        cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{self.node_label}) REQUIRE n.name IS UNIQUE"
        with self._driver.session() as session:
            session.run(cypher)

    def close(self):
        self._driver.close()

    # -------------------- 写 --------------------
    def add_triplet(self, subj: str, pred: str, obj: str):
        """极简三元组： (subj)-[pred]->(obj)  """
        cypher = f"""
        MERGE (s:{self.node_label} {{name: $subj}})
        MERGE (o {{key: $obj}})        // 对象节点不指定标签，可通用
        MERGE (s)-[r:{pred}]->(o)
        RETURN id(r)
        """
        with self._driver.session() as session:
            session.run(cypher, subj=subj, obj=obj)

    def add_node(self, name: str, properties: Optional[dict] = None):
        """新建/更新节点"""
        props = properties or {}
        props["name"] = name
        cypher = f"""
        MERGE (n:{self.node_label} {{name: $name}})
        SET n += $props
        """
        with self._driver.session() as session:
            session.run(cypher, name=name, props=props)

    # -------------------- 读 --------------------
    def get_node(self, name: str):
        cypher = f"MATCH (n:{self.node_label} {{name: $name}}) RETURN n"
        with self._driver.session() as session:
            rec = session.run(cypher, name=name).single()
            return dict(rec["n"]) if rec else None

    def node_count(self) -> int:
        cypher = f"MATCH (n:{self.node_label}) RETURN count(n) as c"
        with self._driver.session() as session:
            return session.run(cypher).single()["c"]

    # -------------------- 上下文管理器（可选） --------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
