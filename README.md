## Self-evolving Agent: A framework for autonomous evolving AR interaction based on context confidence

### [Paper](https://arxiv.org) | [Project Page](https://qinyu.github.io/RAP-Project/) | 
| ![Self-evolving Agent](./images/1.png) |
|:--:|
| The difference between the time it takes for a self-evolving agent to reply  |

Visit our [Project Page](https://github.io/RAP-Project/) for more demostrations.

## 📋 Contents
Self-evolving Agent: YOLO Perception → Vector Memory → Knowledge Graph → LoRA Incremental Learning

## One-click Installation（Linux / macOS）
```
git clone https://github.com/yourname/Self-evolving-Agent.git
cd Self-evolving-Agent

#Automatically create a virtual environment, install dependencies, download weights, and start Neo4j:
bash scripts/run.sh
```
## DEMO
```
from yolo_concept import ConceptLearner

agent = ConceptLearner(
    yolo_weights="yolo_concept_sdk/yolo_concept/data/weights/yolov11.pt",
    neo4j_uri="bolt://localhost:7687",
    neo4j_auth=("neo4j", "password")
)
# Incremental learning from a single image
agent.learn_from_image("desk.jpg", concepts=["keyboard", "mouse"])

# Query learned concepts
print(agent.query_concept("keyboard"))
# → {'definition': 'electronic device ...', 'images': [...], 'relations': [{'type': 'PartOf', 'target': 'desk'}]}

# Generate LoRA training prompt from natural language
prompt = agent.generate_prompt("A cat sitting on a keyboard under RGB lighting")
print(prompt)
```
Open http://localhost:8000/docs in your browser to view the interactive API.

## Autonomous Training (Incremental LoRA)
```
# Put 1 new concept images into a folder
agent.continual_lora_train(
    image_dir="new_concept/rgb_keyboard",
    concept="rgb_keyboard",
    epochs=5,
    output_dir="lora_weights/rgb_keyboard"
)
# he weights are automatically saved and merged on next load
```
## Offline Evaluation
```
cd eval
python eval.py \
  --weights ../Self-evolving Agent/concept/data/weights/yolov8n.pt \
  --gt ./ground_truth.json \
  --topk 5
```

## BibTeX
```
@InProceedings{

}
```



