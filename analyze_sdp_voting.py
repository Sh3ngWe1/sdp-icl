# """
# analyze_sdp_voting.py - Phase 4: 差分隱私聚合與效能分析引擎

# 主要功能：
# 1. 讀取 DGX 產出的 JSONL 推論結果。
# 2. SQuAD 官方字串正規化 (Normalization)。
# 3. 蒙地卡羅模擬：針對不同的 N 與 Epsilon，進行 100 次隨機抽樣與加噪計票。
# 4. 計算精確度 (EM)、F1 分數與平均耗時。
# 5. 輸出最終的 DP 效能報表，供後續畫 Pareto 曲線使用。
# """

# import json
# import os
# import re
# import string
# import random
# import numpy as np
# from collections import Counter
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # ================= 1. 實驗參數設定 =================
# # INPUT_FILES = [
# #     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp_qa_llama_sanitized_n100_results.jsonl",
# #     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp_qa_llama_standard_n100_results.jsonl",
# #     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp_qa_qwen_sanitized_n100_results.jsonl",
# #     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp_qa_qwen_standard_n100_results.jsonl",
# # ]

# INPUT_FILES = [
#     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl",
#     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_llama_standard_n100_5000_results.jsonl",
#     "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl",
#     # "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_qwen_standard_n100_5000_results.jsonl",
# ]

# OUTPUT_REPORT_PATH = "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/dp_analysis_report_5000.json"

# # 要掃描的參數區間 (Ablation Study)
# N_VALUES = [5, 10, 20, 50, 100]
# EPSILON_VALUES = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, float('inf')] # inf 代表純多數決(無隱私保護)
# MONTE_CARLO_TRIALS = 100  # 為了讓曲線平滑，每個 (N, Epsilon) 組合重複模擬 100 次

# # ================= 2. SQuAD 官方評估函數 =================
# def normalize_answer(s):
#     """SQuAD 官方的字串清洗邏輯 (轉小寫、去標點、去冠詞)"""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
#     def white_space_fix(text):
#         return ' '.join(text.split())
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)
#     def lower(text):
#         return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

# def exact_match_score(prediction, ground_truth):
#     return (normalize_answer(prediction) == normalize_answer(ground_truth))

# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     """因為一題可能有多個正確答案，取分數最高的一個"""
#     if not ground_truths:
#         return 0.0
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)

# # ================= 3. 差分隱私投票引擎 =================
# def dp_majority_vote(answers, epsilon):
#     """
#     對傳入的答案清單進行正規化計票，並加入 Laplace 噪音。
#     敏感度 (Sensitivity) delta_f = 1
#     """
#     # 1. 正規化所有答案
#     normalized_answers = [normalize_answer(ans) for ans in answers]
    
#     # 2. 統計真實票數
#     counts = Counter(normalized_answers)
    
#     # 3. 處理無隱私 (Epsilon = inf) 的情況
#     if epsilon == float('inf'):
#         # 發生平手時，Counter.most_common 會依照出現順序回傳，已具備穩定性
#         return counts.most_common(1)[0][0]
    
#     # 4. 加入拉普拉斯噪音 (Laplace Mechanism) NPR
#     noisy_counts = {}
#     for ans, count in counts.items():
#         # DP 核心公式：Laplace(loc=0, scale=delta_f / epsilon)
#         noise = np.random.laplace(0, 1.0 / epsilon)
#         noisy_counts[ans] = count + noise
        
#     # 5. 找出加噪後票數最高的答案 (Noisy ArgMax)
#     best_answer = max(noisy_counts.items(), key=lambda x: x[1])[0]
#     return best_answer

# # ================= 4. 主分析迴圈 =================
# def analyze_file(file_path):
#     if not os.path.exists(file_path):
#         logging.warning(f"檔案不存在: {file_path}")
#         return None
        
#     logging.info(f"\n📂 開始分析: {os.path.basename(file_path)}")
    
#     # 讀取 JSONL
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 data.append(json.loads(line))
                
#     logging.info(f"成功讀取 {len(data)} 題推論結果。")
    
#     results_matrix = [] # 儲存各種 (N, Epsilon) 組合的結果

#     for n in N_VALUES:
#         for eps in EPSILON_VALUES:
#             logging.info(f"⚙️ 模擬參數 -> N={n}, Epsilon={eps}")
            
#             total_em = 0.0
#             total_f1 = 0.0
#             total_time = 0.0
#             valid_questions = 0
            
#             for item in data:
#                 raw_answers = item.get('ensemble_raw_answers', [])
#                 times = item.get('ensemble_times', [])
                
#                 # 取得 Ground Truth (兼容不同的 SQuAD 儲存格式)
#                 gts = item.get('answers', item.get('ground_truth_answers', []))
#                 if isinstance(gts, list) and len(gts) > 0 and isinstance(gts[0], dict):
#                     gts = [ans['text'] for ans in gts] # 提取 dict 中的 text
#                 elif isinstance(gts, str):
#                     gts = [gts]
                
#                 # 檢查資料完整性 (若該題失敗次數過多或 N 大於實際推論數，則跳過或截斷)
#                 if not raw_answers or len(raw_answers) < n:
#                     continue
                    
#                 valid_questions += 1
                
#                 # ====== 蒙地卡羅模擬 (Monte Carlo) ======
#                 mc_em_sum = 0.0
#                 mc_f1_sum = 0.0
                
#                 for _ in range(MONTE_CARLO_TRIALS):
#                     # 1. 隨機子採樣 N 個答案
#                     # (為了確保 i.i.d，這裡用 random.sample 從 100 個中抽出 N 個)
#                     sampled_indices = random.sample(range(len(raw_answers)), n)
#                     sampled_answers = [raw_answers[i] for i in sampled_indices]
                    
#                     # 2. 進行 DP 加噪投票
#                     final_prediction = dp_majority_vote(sampled_answers, eps)
                    
#                     # 3. 計算分數
#                     em = metric_max_over_ground_truths(exact_match_score, final_prediction, gts)
#                     f1 = metric_max_over_ground_truths(f1_score, final_prediction, gts)
                    
#                     mc_em_sum += em
#                     mc_f1_sum += f1
                
#                 # 該題在 100 次模擬下的平均期望分數
#                 total_em += (mc_em_sum / MONTE_CARLO_TRIALS)
#                 total_f1 += (mc_f1_sum / MONTE_CARLO_TRIALS)
                
#                 # 計算這題推論前 N 次的「平均運算成本」
#                 total_time += sum(times[:n])
            
#             if valid_questions > 0:
#                 avg_em = (total_em / valid_questions) * 100
#                 avg_f1 = (total_f1 / valid_questions) * 100
#                 avg_time = (total_time / valid_questions)
                
#                 results_matrix.append({
#                     "N": n,
#                     "epsilon": eps if eps != float('inf') else "inf",
#                     "EM": round(avg_em, 2),
#                     "F1": round(avg_f1, 2),
#                     "avg_time_sec": round(avg_time, 2)
#                 })
                
#     return {
#         "file_name": os.path.basename(file_path),
#         "total_questions_evaluated": valid_questions,
#         "metrics": results_matrix
#     }

# def main():
#     all_reports = []
#     for file_path in INPUT_FILES:
#         report = analyze_file(file_path)
#         if report:
#             all_reports.append(report)
            
#     # 輸出最終報表
#     os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
#     with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
#         json.dump(all_reports, f, indent=4, ensure_ascii=False)
        
#     logging.info(f"\n🎉 差分隱私分析完成！報表已儲存至: {OUTPUT_REPORT_PATH}")
    
#     # 在終端機印出一個漂亮的總結表 (以 N=20 為例)
#     print("\n" + "="*50)
#     print("📊 快速檢視: 隱私-效能權衡 (以 N=20 為例)")
#     print("="*50)
#     for report in all_reports:
#         print(f"\n📁 {report['file_name']}")
#         print(f"{'Epsilon':<10} | {'EM (%)':<10} | {'F1 (%)':<10} | {'耗時 (s)':<10}")
#         print("-" * 45)
#         for row in report['metrics']:
#             if row['N'] == 20:
#                 print(f"{str(row['epsilon']):<10} | {row['EM']:<10.2f} | {row['F1']:<10.2f} | {row['avg_time_sec']:<10.2f}")

# if __name__ == "__main__":
#     main()



"""
analyze_sdp_voting.py - Phase 4: 差分隱私聚合與效能分析引擎

優化項目：
1. 預正規化 (Pre-normalization)：所有答案與 GT 在載入後立即正規化，消除重複計算。
2. 唯一答案快取 (Answer Cache)：每題的唯一答案預先算好 EM/F1，投票後直接查表。
3. 向量化蒙地卡羅 (Vectorized Monte Carlo)：
   - argsort trick 實現全 batch 無放回抽樣 (mc_trials × n)
   - np.add.at 實現向量化計票
   - np.random.laplace batch 加噪
   - np.argmax batch 找出贏家
4. 多程序並行 (Multiprocessing)：Linux fork 共享資料，並行執行 (N, ε) 組合。
"""

import json
import os
import re
import string
import numpy as np
from collections import Counter
import logging
import multiprocessing
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= 1. 實驗參數設定 =================
INPUT_FILES = [
    "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl",
    "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_llama_standard_n100_5000_results.jsonl",
    "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl",
    "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_qwen_standard_n100_5000_results.jsonl",
]

OUTPUT_REPORT_PATH = "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/dp_analysis_report_5000.json"

N_VALUES = [5, 10, 20, 50, 100]
EPSILON_VALUES = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, float('inf')]
# EPSILON_VALUES = [0.1, 1.0, 3.0, 8.0, float('inf')]

MONTE_CARLO_TRIALS = 100

# 並行 worker 數量，設為 None 則自動使用 CPU 核心數
NUM_WORKERS = None

# ================= 2. SQuAD 官方評估函數 =================
def normalize_answer(s):
    """SQuAD 官方字串清洗 (轉小寫、去標點、去冠詞)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def _f1_on_normalized(pred: str, gt: str) -> float:
    """對已正規化的字串計算 F1（跳過重複正規化）"""
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

# ================= 3. 資料預處理 =================
def preprocess_data(raw_data: list) -> list:
    """
    將所有答案與 GT 預正規化，並過濾掉無效資料。
    只執行一次，後續所有 (N, ε) 組合共用。
    """
    processed = []
    for item in raw_data:
        raw_answers = item.get('ensemble_raw_answers', [])
        times = item.get('ensemble_times', [])

        gts = item.get('answers', item.get('ground_truth_answers', []))
        if isinstance(gts, list) and len(gts) > 0 and isinstance(gts[0], dict):
            gts = [ans['text'] for ans in gts]
        elif isinstance(gts, str):
            gts = [gts]

        if not raw_answers or not gts:
            continue

        norm_answers = [normalize_answer(a) for a in raw_answers]
        norm_gts = [normalize_answer(g) for g in gts]

        processed.append({
            'norm_answers': norm_answers,
            'norm_gts': norm_gts,
            'times': times,
        })
    return processed

# ================= 4. 向量化 (N, ε) 模擬 =================

# 全域共享資料（利用 Linux fork，各子程序直接繼承，不需 pickle）
_SHARED_DATA: list = []

def _init_worker(data: list):
    global _SHARED_DATA
    _SHARED_DATA = data

def _simulate_one_combo(args):
    """
    單一 (N, ε) 組合的完整蒙地卡羅模擬。
    使用 NumPy 全向量化，消除 Python-level MC loop。

    向量化流程（以單題為例，mc_trials=100, N=20）：
      1. argsort trick → sample_matrix (100, 20) 一次完成所有無放回抽樣
      2. fancy indexing  → sampled_idxs (100, 20) 對應唯一答案的整數編號
      3. np.add.at       → vote_counts (100, n_unique) batch 計票
      4. np.random.laplace → batch 加噪
      5. np.argmax       → winner_idxs (100,) 一次取得所有 trial 的勝出答案
      6. 查表 em_arr / f1_arr → 直接得到所有 trial 的分數，sum() 即可
    """
    n, eps = args
    data = _SHARED_DATA
    rng = np.random.default_rng()

    total_em = total_f1 = total_time = 0.0
    valid_questions = 0

    for item in data:
        norm_answers = item['norm_answers']
        norm_gts = item['norm_gts']
        times = item['times']

        if len(norm_answers) < n:
            continue
        valid_questions += 1

        # --- 建立唯一答案的整數編碼 ---
        # dict.fromkeys 保留首次出現順序（比 set 穩定）
        unique_answers = list(dict.fromkeys(norm_answers))
        n_total = len(norm_answers)
        n_unique = len(unique_answers)
        ans_to_idx = {a: i for i, a in enumerate(unique_answers)}
        answer_idxs = np.array([ans_to_idx[a] for a in norm_answers], dtype=np.int32)

        # --- 預計算每個唯一答案的 EM / F1（查表，每題只算一次）---
        gts_set = set(norm_gts)
        em_arr = np.array([float(ua in gts_set) for ua in unique_answers], dtype=np.float32)
        f1_arr = np.array(
            [max((_f1_on_normalized(ua, gt) for gt in norm_gts), default=0.0)
             for ua in unique_answers],
            dtype=np.float32
        )

        # --- 向量化無放回抽樣 (argsort trick) ---
        # rng.random 產生 (mc_trials, n_total) 的隨機矩陣，
        # argsort 後取前 N 欄 → 每列都是一份獨立無放回樣本
        rand_matrix = rng.random((MONTE_CARLO_TRIALS, n_total))
        sample_matrix = np.argsort(rand_matrix, axis=1)[:, :n]  # (mc_trials, n)

        # --- 映射到唯一答案整數編號 ---
        sampled_idxs = answer_idxs[sample_matrix]  # (mc_trials, n)

        # --- 向量化計票 ---
        vote_counts = np.zeros((MONTE_CARLO_TRIALS, n_unique), dtype=np.float64)
        rows = np.repeat(np.arange(MONTE_CARLO_TRIALS), n)
        cols = sampled_idxs.ravel()
        np.add.at(vote_counts, (rows, cols), 1.0)

        # --- 加 Laplace 噪音（或不加）---
        if eps != float('inf'):
            vote_counts += rng.laplace(0.0, 1.0 / eps, vote_counts.shape)

        # --- Noisy ArgMax → 查表取分數 ---
        winner_idxs = np.argmax(vote_counts, axis=1)  # (mc_trials,)
        total_em += em_arr[winner_idxs].sum() / MONTE_CARLO_TRIALS
        total_f1 += f1_arr[winner_idxs].sum() / MONTE_CARLO_TRIALS
        total_time += sum(times[:n])

    if valid_questions == 0:
        return None

    return {
        "N": n,
        "epsilon": eps if eps != float('inf') else "inf",
        "EM": round(float(total_em / valid_questions) * 100, 2),
        "F1": round(float(total_f1 / valid_questions) * 100, 2),
        "avg_time_sec": round(float(total_time) / valid_questions, 2),
        "_valid_questions": valid_questions,
    }

# ================= 5. 主分析函數 =================
def analyze_file(file_path: str) -> dict | None:
    if not os.path.exists(file_path):
        logging.warning(f"檔案不存在: {file_path}")
        return None

    logging.info(f"📂 開始分析: {os.path.basename(file_path)}")

    raw_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    logging.info(f"  讀取 {len(raw_data)} 題，開始預正規化...")

    processed = preprocess_data(raw_data)
    logging.info(f"  有效題數：{len(processed)}，開始並行模擬 {len(N_VALUES) * len(EPSILON_VALUES)} 個 (N,ε) 組合...")

    combos = list(product(N_VALUES, EPSILON_VALUES))
    n_workers = NUM_WORKERS or min(len(combos), multiprocessing.cpu_count())

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(processed,)
    ) as pool:
        results = pool.map(_simulate_one_combo, combos)

    results_matrix = [r for r in results if r is not None]

    # 依 (N, epsilon) 排序，保持輸出一致性
    results_matrix.sort(key=lambda r: (r["N"], float(r["epsilon"]) if r["epsilon"] != "inf" else float('inf')))

    valid_questions_last = results_matrix[-1]["_valid_questions"] if results_matrix else 0
    for r in results_matrix:
        r.pop("_valid_questions", None)

    logging.info(f"  完成！有效題數: {valid_questions_last}")
    return {
        "file_name": os.path.basename(file_path),
        "total_questions_evaluated": valid_questions_last,
        "metrics": results_matrix,
    }

def main():
    all_reports = []
    for file_path in INPUT_FILES:
        report = analyze_file(file_path)
        if report:
            all_reports.append(report)

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=4, ensure_ascii=False)

    logging.info(f"🎉 差分隱私分析完成！報表已儲存至: {OUTPUT_REPORT_PATH}")

    print("\n" + "="*50)
    print("📊 快速檢視: 隱私-效能權衡 (以 N=20 為例)")
    print("="*50)
    for report in all_reports:
        print(f"\n📁 {report['file_name']}")
        print(f"{'Epsilon':<10} | {'EM (%)':<10} | {'F1 (%)':<10} | {'耗時 (s)':<10}")
        print("-" * 45)
        for row in report['metrics']:
            if row['N'] == 20:
                print(f"{str(row['epsilon']):<10} | {row['EM']:<10.2f} | {row['F1']:<10.2f} | {row['avg_time_sec']:<10.2f}")

if __name__ == "__main__":
    main()