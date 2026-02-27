import json
from leakage_matcher import calculate_leakage_rate, calculate_leakage_score

# 讀取結果檔
with open('results/baseline_standard_attack_qwen_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print("="*60)

# 手動驗證前 3 題
for i in range(3):
    item = data[i]
    llm_answer = item['ensemble_raw_answers'][0]
    prompted_piis = item['prompted_piis']
    
    leakage_score = calculate_leakage_score(llm_answer, prompted_piis)
    leakage_rate = calculate_leakage_rate(llm_answer, prompted_piis)
    
    print(f"\n[Sample {i+1}] attack_id: {item['attack_id']}")
    print(f"Total PIIs in prompt: {len(prompted_piis)}")
    print(f"Leakage Score: {leakage_score:.2f} / {len(prompted_piis)}")
    print(f"Leakage Rate: {leakage_rate:.2%}")
    print(f"\nLLM Output (first 400 chars):")
    print(llm_answer[:400])
    print("\nPrompted PIIs (first 5):")
    for pii in prompted_piis[:5]:
        print(f"  - {pii['type']}: {pii['value']} ({pii['owner']})")
    print("-"*60)

# 計算整體平均
print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
total_leakage_sum = 0.0
for item in data:
    rate = calculate_leakage_rate(item['ensemble_raw_answers'][0], item['prompted_piis'])
    total_leakage_sum += rate

avg_leakage = (total_leakage_sum / len(data)) * 100
print(f"Average Leakage Rate: {avg_leakage:.2f}%")
print(f"Total samples: {len(data)}")
