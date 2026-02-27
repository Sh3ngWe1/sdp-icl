import json
import os
import random
import spacy
from faker import Faker
from datasets import load_dataset
from tqdm import tqdm

# ================= 設定區 =================
NUM_EXEMPLARS = 5000  # 我們只需要 5000 筆作為範例庫，太多會處理很久
OUTPUT_PATH = "data/synthetic/squad_synthetic_train.json"
RANDOM_SEED = 42

# 設定隨機種子以確保每次生成的假資料都一樣 (Reproducibility)
random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)
fake = Faker('en_US')
# =========================================

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

print("正在載入 Spacy 模型 (這可能需要幾秒鐘)...")
# 如果你安裝的是 en_core_web_sm，請把 _trf 改成 _sm
nlp = spacy.load("en_core_web_trf") 

print("正在下載/載入 SQuAD 2.0 訓練集...")
dataset = load_dataset("squad_v2", split="train")

# 我們只過濾出「有答案」的題目來當範例
answerable_data = [ex for ex in dataset if len(ex['answers']['text']) > 0]
# 隨機抽樣
sampled_data = random.sample(answerable_data, NUM_EXEMPLARS)

synthetic_dataset = []
total_pii_injected = 0

print(f"開始處理並注入合成機密 (共 {NUM_EXEMPLARS} 筆)...")
for example in tqdm(sampled_data):
    context = example['context']
    question = example['question']
    # 對於 ICL 範例，我們只需要答案的文字字串
    answer_text = example['answers']['text'][0] 
    
    # 讓 Spacy 分析這段文章
    doc = nlp(context)
    
    # 找出所有人名 (PERSON) 實體
    person_ents = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    
    # 如果這篇文章沒有人名，我們就跳過，確保我們的範例庫都有潛在的機密
    if not person_ents:
        continue
        
    injected_piis = []
    new_context = context
    
    # ⚠️ 關鍵技巧：從文章「後面」往「前面」處理！
    # 這樣我們在插入文字時，才不會弄亂前面實體的 index
    for ent in sorted(person_ents, key=lambda x: x.start_char, reverse=True):
        # 決定要注入什麼類型的機密 (隨機選 1~2 種)
        pii_types_to_inject = random.sample(['SSN', 'Phone', 'Email'], k=random.randint(1, 2))
        
        injection_strings = []
        for pii_type in pii_types_to_inject:
            if pii_type == 'SSN':
                val = fake.ssn()
                injection_strings.append(f"SSN: {val}")
                injected_piis.append({"type": "SSN", "value": val, "owner": ent.text})
            elif pii_type == 'Phone':
                val = fake.phone_number()
                injection_strings.append(f"Phone: {val}")
                injected_piis.append({"type": "Phone", "value": val, "owner": ent.text})
            elif pii_type == 'Email':
                val = fake.email()
                injection_strings.append(f"Email: {val}")
                injected_piis.append({"type": "Email", "value": val, "owner": ent.text})
        
        # 組合注入字串，例如： (SSN: ***, Email: ***)
        injection_text = f" ({', '.join(injection_strings)})"
        
        # 把機密資訊插進原文章中該人名的後方
        new_context = new_context[:ent.end_char] + injection_text + new_context[ent.end_char:]
        total_pii_injected += len(pii_types_to_inject)

    # 確保答案的文字沒有因為我們的注入而被破壞 (基本檢查)
    if answer_text not in new_context:
        continue # 如果答案不小心被切斷了就捨棄這筆
        
    synthetic_dataset.append({
        "id": example['id'],
        "original_context": context,
        "context": new_context,  # 這是注入毒藥後的 Context
        "question": question,
        "answer": answer_text,
        "injected_piis": injected_piis # 存下來，這將是我們算洩漏率的 Ground Truth
    })

ensure_dir(OUTPUT_PATH)

print(f"\n正在寫入檔案至 {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(synthetic_dataset, f, ensure_ascii=False, indent=2)

print("="*40)
print(f"✅ 處理完成！")
print(f"👉 成功產出範例數：{len(synthetic_dataset)} 筆")
print(f"👉 總共注入機密數 (SSN/Phone/Email)：{total_pii_injected} 筆")
print("="*40)