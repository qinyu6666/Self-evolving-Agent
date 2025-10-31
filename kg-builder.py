import re, torch
from neo4j import GraphDatabase
from transformers import pipeline

class KGBuilder:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", pwd="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        # 用小型 LLM 做三元组抽取
        self.triplet_extractor = pipeline("text2text-generation",
                                          model="Babelscape/rebel-large",
                                          device=0 if torch.cuda.is_available() else -1)

    # ---- 抽取 ----
    def extract_triples(self, text, max_new=8):
        out = self.triplet_extractor(text, max_new_tokens=max_new, return_full_text=False)
        triples = []
        for o in out:
            pieces = o["generated_text"].split("<obj>")
            if len(pieces) == 3:
                subj, pred, obj = pieces
                triples.append((subj.strip(), pred.strip(), obj.strip()))
        if not triples:
            triples.append(("it", "is", "object"))
        return triples

    # ---- 写入 Neo4j ----
    def write_to_neo4j(self, name, category, triples):
        with self.driver.session() as session:
            session.run("MERGE (n:Object {name:$name, category:$cat})", name=name, cat=category)
            for s, p, o in triples:
                session.run("""
                    MERGE (sub:Entity {name:$sub})
                    MERGE (obj:Entity {name:$obj})
                    MERGE (sub)-[r:REL {type:$pred}]->(obj)
                """, sub=s, pred=p, obj=o)
        print(f"[KG] 写入 {len(triples)} 条三元组到 Neo4j")