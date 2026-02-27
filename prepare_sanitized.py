import json
import os
import spacy
from collections import defaultdict
from tqdm import tqdm
import re

# ================= 設定區 =================
INPUT_PATH = "data/synthetic/squad_synthetic_train.json"
OUTPUT_PATH = "data/sanitized/squad_sanitized_train.json"
# 允許被替換的一般實體標籤 (可依需求增減)
ALLOWED_NER_LABELS = ["PERSON", "ORG", "GPE", "LOC", "DATE"]
# =========================================

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

print("正在載入 Spacy 模型 (這可能需要幾秒鐘)...")
# 如果上一步是用 sm 版本，請記得改成 en_core_web_sm
nlp = spacy.load("en_core_web_trf") 

print(f"正在讀取合成資料集: {INPUT_PATH} ...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

sanitized_dataset = []

print(f"開始進行單向實體淨化 (共 {len(dataset)} 筆)...")
for example in tqdm(dataset):
    context = example['context']
    question = example['question']
    answer = example['answer']
    injected_piis = example.get('injected_piis', [])
    
    mapping_table = {}
    counters = defaultdict(int)
    
    # ---------------------------------------------------------
    # 步驟 1: 優先處理「極度敏感」的合成機密 (SSN, Phone, Email)
    # ---------------------------------------------------------
    for pii in injected_piis:
        pii_value = pii['value']
        pii_type = pii['type'] # 'SSN', 'Phone', 或 'Email'
        
        if pii_value not in mapping_table:
            counters[pii_type] += 1
            mapping_table[pii_value] = f"[{pii_type}_{counters[pii_type]}]"
            
    # ---------------------------------------------------------
    # 步驟 2: 處理「一般」的命名實體 (透過 Spacy NER)
    # ---------------------------------------------------------
    doc = nlp(context)
    for ent in doc.ents:
        if ent.label_ in ALLOWED_NER_LABELS:
            ent_value = ent.text
            # 如果這個字串還沒被替換過 (避免蓋掉剛剛的 PII)
            if ent_value not in mapping_table:
                counters[ent.label_] += 1
                mapping_table[ent_value] = f"[{ent.label_}_{counters[ent.label_]}]"
                
    # ---------------------------------------------------------
    # 步驟 3: 執行字串替換 (Discretization 輸出降維)
    # ---------------------------------------------------------
    # ⚠️ 關鍵技巧：依據字串長度「由長到短」排序替換
    # 這樣可以避免 "New York" 裡的 "York" 先被換掉導致出錯
    sorted_keys = sorted(mapping_table.keys(), key=len, reverse=True)
    
    def sanitize_text(text):
        for k in sorted_keys:
            # 確保不會發生大小寫造成的漏網之魚 (但此處為求準確，使用精確匹配)
            text = text.replace(k, mapping_table[k])
        return text
        
    s_context = sanitize_text(context)
    s_question = sanitize_text(question)
    s_answer = sanitize_text(answer)
    
    # ---------------------------------------------------------
    # 步驟 4: 封裝結果 (故意「丟棄」 mapping_table 以實現單向淨化)
    # ---------------------------------------------------------
    sanitized_dataset.append({
        "id": example['id'],
        "sanitized_context": s_context,
        "sanitized_question": s_question,
        "sanitized_answer": s_answer, # 如果答案原本是地名，這裡也會變成 [GPE_1]，這就是我們要的！
        "original_answer": answer,    # 留著作參考
        "injected_piis": injected_piis # 留著給 Attack 實驗當 Ground Truth 算分用
    })

ensure_dir(OUTPUT_PATH)

print(f"\n正在寫入淨化後的檔案至 {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sanitized_dataset, f, ensure_ascii=False, indent=2)

print("="*50)
print(f"✅ 單向實體淨化完成！ (SDP-ICL 第一步解鎖)")
print(f"👉 成功產出淨化範例數：{len(sanitized_dataset)} 筆")
print("="*50)