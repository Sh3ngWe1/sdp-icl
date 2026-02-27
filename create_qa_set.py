import json
import random
import os
from datasets import load_dataset

OUTPUT_PATH = "data/qa_validation_set.json"
NUM_QA_SAMPLES = 500
random.seed(42)

print("正在下載/載入 SQuAD 2.0 Validation Set...")
dataset = load_dataset("squad_v2", split="validation")
# 只挑有答案的問題
answerable_data = [ex for ex in dataset if len(ex['answers']['text']) > 0]

print(f"隨機抽取 {NUM_QA_SAMPLES} 題作為效能測試...")
sampled_data = random.sample(answerable_data, NUM_QA_SAMPLES)
qa_dataset = []

for example in sampled_data:
    qa_dataset.append({
        "id": example['id'],
        "context": example['context'],
        "question": example['question'],
        "ground_truth_answers": example['answers']['text'] # SQuAD 通常有多個正確答案寫法
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
print(f"✅ 成功產出 QA 測試卷至 {OUTPUT_PATH}")