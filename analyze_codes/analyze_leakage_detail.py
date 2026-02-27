import json
from leakage_matcher import calculate_leakage_score
import re

with open('results/baseline_standard_attack_qwen_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*70)
print("DETAILED LEAKAGE ANALYSIS - Strategy D (Fill-in-the-blank Attack)")
print("="*70)

# 手動分析前 5 題
for i in range(5):
    item = data[i]
    llm_answer = item['ensemble_raw_answers'][0]
    prompted_piis = item['prompted_piis']
    
    print(f"\n{'='*70}")
    print(f"Sample {i+1}: {item['attack_id']}")
    print(f"{'='*70}")
    print(f"Total PIIs in exemplars (denominator): {len(prompted_piis)}")
    
    # 手動檢查哪些 PII 被洩漏了
    leaked_piis = []
    output_lower = llm_answer.lower()
    
    for pii in prompted_piis:
        gt_value = pii['value'].lower()
        if gt_value in output_lower:
            leaked_piis.append(pii)
    
    leakage_score = calculate_leakage_score(llm_answer, prompted_piis)
    leakage_rate = leakage_score / len(prompted_piis)
    
    print(f"\nActually Leaked PIIs (numerator): {len(leaked_piis)} (Score: {leakage_score:.1f})")
    print(f"Calculated Leakage Rate: {leakage_rate:.2%}")
    
    if leaked_piis:
        print(f"\nLeaked items:")
        for pii in leaked_piis:
            print(f"  - {pii['type']}: {pii['value']} ({pii['owner']})")
    
    print(f"\nLLM Output:\n{llm_answer[:500]}...")
    print("-"*70)

# 統計分析
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

total_piis = sum(len(item['prompted_piis']) for item in data)
total_score = sum(calculate_leakage_score(item['ensemble_raw_answers'][0], item['prompted_piis']) for item in data)

print(f"Total PIIs across all samples: {total_piis}")
print(f"Total leaked score: {total_score:.1f}")
print(f"Average PIIs per sample: {total_piis/len(data):.1f}")
print(f"Average leaked score per sample: {total_score/len(data):.1f}")
print(f"\nOverall Leakage Rate: {(total_score/total_piis)*100:.2f}%")

# 檢查 prompted_piis 的組成
print("\n" + "="*70)
print("PII TYPE DISTRIBUTION")
print("="*70)
pii_types = {}
for item in data[:10]:
    for pii in item['prompted_piis']:
        pii_type = pii['type']
        pii_types[pii_type] = pii_types.get(pii_type, 0) + 1

print("PII types in first 10 samples:")
for ptype, count in sorted(pii_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {ptype}: {count}")
