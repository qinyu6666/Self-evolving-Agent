import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "data/lora"

class Learner:
    def __init__(self):
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_cfg, device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, peft_config)
        if os.path.exists(LORA_PATH):
            self.model.load_adapter(LORA_PATH, adapter_name="default")

    # ---- 用图文对微调 1 epoch ----
    def update(self, name, desc):
        prompt = f"Describe {name} in one sentence:"
        target = desc
        full = prompt + " " + target
        ds = Dataset.from_dict({"text": [full]})
        def tokenize(x):
            return self.tokenizer(x["text"], truncation=True, max_length=128)
        ds = ds.map(tokenize, batched=True, remove_columns=["text"])
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        for batch in torch.utils.data.DataLoader(ds, batch_size=1):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            out = self.model(**batch, labels=batch["input_ids"])
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.model.save_pretrained(LORA_PATH)
        print("[Learner] LoRA 权重已更新")
