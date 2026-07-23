"""
find_qualitative_cases.py
=========================
從已儲存的 SDP-ICL 實驗結果中，自動找出適合放入論文 Chapter 4
qualitative comparison 小節的代表性案例。

支援的結果檔類型：
  - SDP JSONL (results/sdp/val_5000/*.jsonl)
      欄位：id, context, question, ground_truth_answers,
             ensemble_raw_answers, ensemble_times, success_count, fail_count
  - Baseline multiseed JSONL (results/baselines/multiseed/*.jsonl)
      QA 欄位：id, context, question, ground_truth_answers,
               answer_prediction, prompted_piis, time_seconds
      Attack 欄位：attack_id, original_id, context, malicious_question,
                   ground_truth_secrets, ...

主要輸出：
  outputs/qualitative_candidates.json
  outputs/qualitative_candidates.csv
  outputs/qualitative_candidates.md

執行範例：
  python scripts/find_qualitative_cases.py
  python scripts/find_qualitative_cases.py --top-k 5 --model qwen --epsilon inf
  python scripts/find_qualitative_cases.py \\
      --input results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl \\
      --baseline results/baselines/multiseed/baseline_standard_qa_qwen_seed_42_results.jsonl \\
      --output-dir outputs --top-k 5
"""

import argparse
import collections
import csv
import json
import re
import string
import sys
from pathlib import Path
from typing import Optional

# ==================== 常數設定 ====================

# SDP 主要結果目錄（相對於本 script 所在資料夾的上一層，即專案根目錄）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SDP_DIR = PROJECT_ROOT / "results" / "sdp" / "val_5000"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "baselines" / "multiseed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_results"

# 案例分類閾值
F1_SUCCESS_THRESH = 0.8       # F1 >= 此值 → success
F1_PARTIAL_LOWER = 0.4        # F1 介於此值與 F1_SUCCESS_THRESH → partial
F1_FAILURE_THRESH = 0.2       # F1 <= 此值 → failure

# ==================== SQuAD 標準評估函數 ====================
# （與 evaluate_baselines.py 保持一致）

def normalize_answer(s: str) -> str:
    """SQuAD 官方字串正規化：轉小寫、去標點、去冠詞、壓縮空格"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def max_over_ground_truths(metric_fn, prediction: str, ground_truths: list) -> float:
    """因為 SQuAD 一題可能有多個參考答案，取最高分"""
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ==================== 工具函數 ====================

def majority_vote(answers: list) -> str:
    """對答案清單做多數決（純 argmax，無 DP 噪音），回傳正規化後的最多票答案"""
    if not answers:
        return ""
    normalized = [normalize_answer(a) for a in answers]
    counter = collections.Counter(a for a in normalized if a)
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def parse_filename_metadata(path: Path) -> dict:
    """
    從檔名推測 model、exemplar_type、N 等 metadata。
    支援格式：
      sdp_qa_{model}_{exemplar_type}_n{N}_5000_results.jsonl
      baseline_{exemplar_type}_{task}_{model}_seed_{seed}_results.jsonl
    """
    name = path.stem  # 去掉副檔名
    meta = {
        "model": None,
        "exemplar_type": None,
        "N": None,
        "seed": None,
        "task_type": None,
        "method": None,
    }

    # SDP 檔：sdp_qa_qwen_sanitized_n100_5000
    sdp_m = re.match(
        r"sdp_(?P<task>qa|attack)_(?P<model>qwen|llama)_(?P<etype>sanitized|standard)_n(?P<N>\d+)",
        name,
    )
    if sdp_m:
        meta["model"] = sdp_m.group("model")
        meta["exemplar_type"] = sdp_m.group("etype")
        meta["N"] = int(sdp_m.group("N"))
        meta["task_type"] = sdp_m.group("task")
        meta["method"] = "SDP-ICL"
        return meta

    # Baseline 檔：baseline_standard_qa_qwen_seed_42
    bl_m = re.match(
        r"baseline_(?P<etype>sanitized|standard)_(?P<task>qa|attack)_(?P<model>qwen|llama)_seed_(?P<seed>\d+)",
        name,
    )
    if bl_m:
        meta["model"] = bl_m.group("model")
        meta["exemplar_type"] = bl_m.group("etype")
        meta["seed"] = int(bl_m.group("seed"))
        meta["task_type"] = bl_m.group("task")
        meta["method"] = (
            "Baseline-Standard" if bl_m.group("etype") == "standard" else "Baseline-Sanitized"
        )
        return meta

    # Singleseed JSON：baseline_standard_qa_qwen
    bl2_m = re.match(
        r"baseline_(?P<etype>sanitized|standard)_(?P<task>qa|attack)_(?P<model>qwen|llama)",
        name,
    )
    if bl2_m:
        meta["model"] = bl2_m.group("model")
        meta["exemplar_type"] = bl2_m.group("etype")
        meta["task_type"] = bl2_m.group("task")
        meta["method"] = (
            "Baseline-Standard" if bl2_m.group("etype") == "standard" else "Baseline-Sanitized"
        )
        return meta

    return meta


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def load_csv(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_result_file(path: Path) -> list:
    """根據副檔名選擇讀取方式"""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"不支援的檔案格式：{path.suffix}")


def normalize_record(raw: dict, meta: dict) -> Optional[dict]:
    """
    將原始 record 正規化成統一的 internal schema。
    回傳 None 代表該筆不是 QA 任務（例如 attack），直接跳過。
    """
    task_type = meta.get("task_type", "")

    # ---- SDP JSONL ----
    if meta.get("method") == "SDP-ICL":
        qid = raw.get("id", "")
        question = raw.get("question", "")
        context = raw.get("context", "")
        gt_answers = raw.get("ground_truth_answers", [])
        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]

        raw_answers = raw.get("ensemble_raw_answers", [])
        times = raw.get("ensemble_times", [])
        success_count = raw.get("success_count", len(raw_answers))
        fail_count = raw.get("fail_count", 0)

        # final answer = majority vote over all raw answers
        final_answer = majority_vote(raw_answers)

        # avg latency
        avg_latency = round(sum(times) / len(times), 4) if times else None
        total_latency = round(sum(times), 2) if times else None

        # answer diversity（unique 答案數 / N，越低代表答案越一致）
        n_total = len(raw_answers)
        n_unique = len(set(normalize_answer(a) for a in raw_answers if a))
        diversity = round(n_unique / n_total, 4) if n_total else None

        return {
            "id": qid,
            "question": question,
            "context": context,
            "ground_truth_answers": gt_answers,
            "method": meta["method"],
            "model": meta.get("model"),
            "exemplar_type": meta.get("exemplar_type"),
            "N": meta.get("N"),
            "seed": None,
            "final_answer": final_answer,
            "ensemble_raw_answers": raw_answers,
            "avg_latency_sec": avg_latency,
            "total_latency_sec": total_latency,
            "success_count": success_count,
            "fail_count": fail_count,
            "answer_diversity": diversity,
            # 下面兩個欄位會在後面填入
            "f1": None,
            "em": None,
        }

    # ---- Baseline multiseed JSONL (QA) ----
    if raw.get("answer_prediction") is not None:
        # 可能是 QA 或 attack，只保留 QA
        if task_type == "attack":
            return None

        qid = raw.get("id", "")
        question = raw.get("question", "")
        context = raw.get("context", "")
        gt_answers = raw.get("ground_truth_answers", [])
        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]

        pred = raw.get("answer_prediction", "")
        try:
            latency = float(raw.get("time_seconds", 0) or 0)
        except (ValueError, TypeError):
            latency = None

        return {
            "id": qid,
            "question": question,
            "context": context,
            "ground_truth_answers": gt_answers,
            "method": meta["method"],
            "model": meta.get("model"),
            "exemplar_type": meta.get("exemplar_type"),
            "N": 1,
            "seed": meta.get("seed"),
            "final_answer": pred,
            "ensemble_raw_answers": [pred],
            "avg_latency_sec": latency,
            "total_latency_sec": latency,
            "success_count": 1,
            "fail_count": 0,
            "answer_diversity": None,
            "f1": None,
            "em": None,
        }

    # ---- Baseline singleseed JSON ----
    if raw.get("ensemble_raw_answers") is not None and meta.get("method") != "SDP-ICL":
        if task_type == "attack":
            return None
        qid = raw.get("id", "")
        question = raw.get("question", "")
        context = raw.get("context", "")
        gt_answers = raw.get("ground_truth_answers", [])
        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]

        raw_answers = raw.get("ensemble_raw_answers", [])
        final_answer = majority_vote(raw_answers) if raw_answers else ""

        return {
            "id": qid,
            "question": question,
            "context": context,
            "ground_truth_answers": gt_answers,
            "method": meta["method"],
            "model": meta.get("model"),
            "exemplar_type": meta.get("exemplar_type"),
            "N": len(raw_answers),
            "seed": None,
            "final_answer": final_answer,
            "ensemble_raw_answers": raw_answers,
            "avg_latency_sec": None,
            "total_latency_sec": None,
            "success_count": len(raw_answers),
            "fail_count": 0,
            "answer_diversity": None,
            "f1": None,
            "em": None,
        }

    return None


def compute_scores(record: dict) -> dict:
    """計算 F1 / EM，直接修改並回傳"""
    pred = record.get("final_answer", "") or ""
    gts = record.get("ground_truth_answers", []) or []
    record["f1"] = round(max_over_ground_truths(compute_f1, pred, gts), 4)
    record["em"] = int(max_over_ground_truths(compute_em, pred, gts))
    return record


# ==================== 讀取與建立索引 ====================

def load_and_index(paths: list) -> dict:
    """
    讀取多個結果檔，正規化後依 question_id 建立索引。
    回傳：{qid: [record1, record2, ...]}
    """
    index = collections.defaultdict(list)
    total_loaded = 0
    total_skipped = 0

    for path in paths:
        meta = parse_filename_metadata(path)
        print(f"  [READ] {path.name}")
        print(f"     schema -> method={meta['method']}, model={meta['model']}, "
              f"exemplar_type={meta['exemplar_type']}, N={meta['N']}, seed={meta['seed']}")

        try:
            raw_records = load_result_file(path)
        except Exception as e:
            print(f"     [WARN] 讀取失敗：{e}，跳過")
            continue

        loaded = 0
        skipped = 0
        for raw in raw_records:
            rec = normalize_record(raw, meta)
            if rec is None:
                skipped += 1
                continue
            rec = compute_scores(rec)
            qid = rec["id"]
            if qid:
                index[qid].append(rec)
                loaded += 1
            else:
                skipped += 1

        print(f"     [OK] 正規化成功：{loaded} 筆，跳過：{skipped} 筆")
        total_loaded += loaded
        total_skipped += skipped

    print(f"\n  合計讀取：{total_loaded} 筆，跳過：{total_skipped} 筆，"
          f"涵蓋 {len(index)} 個不重複問題 ID\n")
    return index


# ==================== 案例篩選邏輯 ====================

def classify_case(record: dict) -> str:
    """根據 F1/EM 分類為 success / partial / failure"""
    f1 = record.get("f1") or 0.0
    em = record.get("em") or 0
    if em == 1 or f1 >= F1_SUCCESS_THRESH:
        return "success"
    if f1 >= F1_PARTIAL_LOWER:
        return "partial"
    if f1 <= F1_FAILURE_THRESH:
        return "failure"
    return "partial"  # 0.2 < F1 < 0.4，也算 partial boundary


def select_cases(
    index: dict,
    top_k: int = 5,
    model_filter: Optional[str] = None,
    epsilon_filter: Optional[float] = None,
    n_filter: Optional[int] = None,
    sdp_method_filter: str = "SDP-ICL",
    require_baseline: bool = False,
) -> dict:
    """
    從索引中找出三類代表案例：success / partial / failure。

    優先策略：
    - 每個 qid 至少要有一筆 SDP-ICL 記錄，才考慮放入候選。
    - 成功案例：F1 >= 0.8 或 EM=1，優先選 F1 最高者
    - 部分成功：0.4 <= F1 < 0.8，優先選 F1 差異大的（說明 metric gap）
    - 失敗案例：F1 <= 0.2，優先選「ensemble 答案分散」的，以說明限制

    回傳：{"success": [...], "partial": [...], "failure": [...]}
    """

    success_candidates = []
    partial_candidates = []
    failure_candidates = []

    for qid, records in index.items():
        # 只考慮含有 SDP-ICL 記錄的問題
        sdp_records = [r for r in records if r["method"] == sdp_method_filter]
        if not sdp_records:
            continue

        # 模型過濾
        if model_filter:
            sdp_records = [r for r in sdp_records if r["model"] == model_filter]
        if not sdp_records:
            continue

        # --require-baseline：確保同時有 baseline 資料（支援三方比較）
        if require_baseline:
            has_std = any(r["method"] == "Baseline-Standard"
                          and (model_filter is None or r["model"] == model_filter)
                          for r in records)
            has_san = any(r["method"] == "Baseline-Sanitized"
                          and (model_filter is None or r["model"] == model_filter)
                          for r in records)
            if not (has_std and has_san):
                continue

        # 取第一筆 SDP 記錄（通常每個 qid 對一個模型只有一筆 SDP）
        sdp_rec = sdp_records[0]
        case_type = classify_case(sdp_rec)

        # 找對應的 baseline 記錄（若有）
        baseline_std = next(
            (r for r in records
             if r["method"] == "Baseline-Standard"
             and (model_filter is None or r["model"] == model_filter)),
            None,
        )
        baseline_san = next(
            (r for r in records
             if r["method"] == "Baseline-Sanitized"
             and (model_filter is None or r["model"] == model_filter)),
            None,
        )

        candidate = {
            "case_type": case_type,
            "sdp_record": sdp_rec,
            "baseline_standard": baseline_std,
            "baseline_sanitized": baseline_san,
            "selection_reason": "",
        }

        # ---- 判斷「被選中原因」----
        f1 = sdp_rec.get("f1", 0.0) or 0.0
        em = sdp_rec.get("em", 0) or 0
        diversity = sdp_rec.get("answer_diversity") or 0.0
        std_f1 = (baseline_std or {}).get("f1") or 0.0
        san_f1 = (baseline_san or {}).get("f1") or 0.0

        reasons = []
        if case_type == "success":
            reasons.append(f"SDP-ICL F1={f1:.3f}" + (" (EM=1)" if em else ""))
            if std_f1 > 0 and f1 >= std_f1:
                reasons.append(f"優於或持平 Baseline-Standard (F1={std_f1:.3f})")
            if san_f1 > 0 and f1 >= san_f1:
                reasons.append(f"優於 Baseline-Sanitized (F1={san_f1:.3f})")
            candidate["selection_reason"] = "; ".join(reasons)
            success_candidates.append(candidate)

        elif case_type == "partial":
            reasons.append(f"SDP-ICL F1={f1:.3f} (介於 {F1_PARTIAL_LOWER}~{F1_SUCCESS_THRESH})")
            if std_f1 > f1 + 0.1:
                reasons.append(f"Baseline-Standard 明顯較高 (F1={std_f1:.3f})，顯示隱私代價")
            reasons.append("EM=0 但語意可能部分正確，適合說明 metric gap")
            candidate["selection_reason"] = "; ".join(reasons)
            partial_candidates.append(candidate)

        elif case_type == "failure":
            reasons.append(f"SDP-ICL F1={f1:.3f} (<=  {F1_FAILURE_THRESH})")
            if diversity and diversity > 0.5:
                reasons.append(
                    f"ensemble highly scattered (diversity={diversity:.3f}),"
                    f" suggests DP aggregation difficulty"
                )
            elif diversity is not None and diversity < 0.1:
                reasons.append(
                    f"ensemble concentrated (diversity={diversity:.3f}) but still wrong,"
                    f" suggests first-stage error"
                )
            if std_f1 > 0.6:
                reasons.append(
                    f"Baseline-Standard answered correctly (F1={std_f1:.3f}),"
                    f" SDP introduced error -- good privacy-utility tradeoff example"
                )
            if san_f1 < 0.3 and std_f1 > 0.6:
                reasons.append(
                    f"Both SDP and Baseline-Sanitized fail while Standard succeeds,"
                    f" suggests sanitization itself causes the accuracy drop"
                )
            candidate["selection_reason"] = "; ".join(reasons)
            # Attach a sorting key to prioritize high-contrast cases
            candidate["_sort_key"] = (
                std_f1 - f1,           # higher std_f1 gap = more interesting
                diversity or 0,        # higher diversity = more interesting
                -f1,                   # lower SDP f1 = clearer failure
            )
            failure_candidates.append(candidate)

    # ---- 排序並取 top-k ----
    success_candidates.sort(key=lambda x: x["sdp_record"]["f1"] or 0, reverse=True)
    partial_candidates.sort(
        key=lambda x: abs((x["sdp_record"]["f1"] or 0) - 0.6), reverse=False
    )
    failure_candidates.sort(
        key=lambda x: x.get("_sort_key", (0, 0, 0)),
        reverse=True,
    )

    return {
        "success": success_candidates[:top_k],
        "partial": partial_candidates[:top_k],
        "failure": failure_candidates[:top_k],
    }


# ==================== 輸出格式化 ====================

def format_answer_list(answers: list, max_show: int = 5) -> str:
    """縮短列印 ensemble 答案（只顯示 top-N 個不重複答案與票數）"""
    if not answers:
        return "(none)"
    counter = collections.Counter(normalize_answer(a) for a in answers if a)
    top = counter.most_common(max_show)
    parts = [f'"{ans}" × {cnt}' for ans, cnt in top]
    remaining = len(counter) - max_show
    if remaining > 0:
        parts.append(f"... 另有 {remaining} 種不同答案")
    return " | ".join(parts)


def flatten_for_csv(candidate: dict) -> dict:
    """將巢狀 candidate 展平成一層 dict，方便寫入 CSV"""
    sdp = candidate["sdp_record"]
    std = candidate.get("baseline_standard") or {}
    san = candidate.get("baseline_sanitized") or {}

    return {
        "case_type": candidate["case_type"],
        "selection_reason": candidate["selection_reason"],
        "question_id": sdp.get("id"),
        "model": sdp.get("model"),
        "exemplar_type": sdp.get("exemplar_type"),
        "N": sdp.get("N"),
        "question": sdp.get("question"),
        "context_snippet": (sdp.get("context") or "")[:200],
        "ground_truth": " | ".join(sdp.get("ground_truth_answers") or []),
        "sdp_final_answer": sdp.get("final_answer"),
        "sdp_f1": sdp.get("f1"),
        "sdp_em": sdp.get("em"),
        "sdp_avg_latency_sec": sdp.get("avg_latency_sec"),
        "sdp_answer_diversity": sdp.get("answer_diversity"),
        "sdp_top_ensemble_answers": format_answer_list(sdp.get("ensemble_raw_answers") or []),
        "baseline_standard_answer": std.get("final_answer"),
        "baseline_standard_f1": std.get("f1"),
        "baseline_standard_em": std.get("em"),
        "baseline_sanitized_answer": san.get("final_answer"),
        "baseline_sanitized_f1": san.get("f1"),
        "baseline_sanitized_em": san.get("em"),
    }


def render_markdown(cases: dict) -> str:
    """產生 Markdown 格式的 qualitative report"""
    lines = ["# SDP-ICL Qualitative Candidates Report\n"]
    lines.append(
        "> 本報告由 `scripts/find_qualitative_cases.py` 自動產生。\n"
        "> 所有案例均來自真實實驗結果，請人工審查後決定是否放入論文。\n"
    )

    section_titles = {
        "success": "## 1. Success Cases（成功案例，F1 ≥ 0.8 或 EM = 1）",
        "partial": "## 2. Partial Success Cases（部分成功案例，0.4 ≤ F1 < 0.8）",
        "failure": "## 3. Failure Cases（失敗案例，F1 ≤ 0.2）",
    }

    for case_type, title in section_titles.items():
        candidates = cases.get(case_type, [])
        lines.append(f"\n{title}\n")
        if not candidates:
            lines.append("_（未找到符合條件的案例）_\n")
            continue

        for idx, cand in enumerate(candidates, 1):
            sdp = cand["sdp_record"]
            std = cand.get("baseline_standard") or {}
            san = cand.get("baseline_sanitized") or {}

            lines.append(f"### Case {idx} ({case_type.upper()})\n")
            lines.append(f"**自動判斷原因**：{cand['selection_reason']}\n")
            lines.append("")

            # Metadata table
            lines.append("| 屬性 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| Question ID | `{sdp.get('id', 'N/A')}` |")
            lines.append(f"| Model | `{sdp.get('model', 'N/A')}` |")
            lines.append(f"| Exemplar Type | `{sdp.get('exemplar_type', 'N/A')}` |")
            lines.append(f"| N (ensemble size) | `{sdp.get('N', 'N/A')}` |")
            lines.append(f"| SDP Avg Latency | `{sdp.get('avg_latency_sec', 'N/A')} sec/query` |")
            lines.append(f"| Answer Diversity | `{sdp.get('answer_diversity', 'N/A')}` |")
            lines.append("")

            lines.append(f"**Question**: {sdp.get('question', 'N/A')}\n")
            lines.append(f"**Ground Truth**: `{' | '.join(sdp.get('ground_truth_answers') or [])}`\n")

            context_snip = (sdp.get("context") or "")[:300]
            lines.append(f"**Context Snippet**:\n> {context_snip}...\n")

            lines.append("**Answers**:\n")
            lines.append("| Method | Answer | F1 | EM |")
            lines.append("|--------|--------|----|----|")

            def row(method_name, rec: dict):
                if not rec:
                    return f"| {method_name} | _N/A_ | _N/A_ | _N/A_ |"
                ans = rec.get("final_answer", "") or ""
                f1 = rec.get("f1")
                em = rec.get("em")
                f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
                em_str = str(em) if em is not None else "N/A"
                return f"| {method_name} | {ans} | {f1_str} | {em_str} |"

            lines.append(row("Baseline-Standard (No Privacy)", std))
            lines.append(row("Baseline-Sanitized (DP-ICL)", san))
            lines.append(row("**SDP-ICL (Proposed)**", sdp))
            lines.append("")

            # Ensemble 答案分布
            top_ensemble = format_answer_list(sdp.get("ensemble_raw_answers") or [])
            lines.append(f"**SDP Ensemble Top Answers** (N={sdp.get('N', '?')} total): {top_ensemble}\n")
            lines.append("---\n")

    return "\n".join(lines)


# ==================== 主程式 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="從 SDP-ICL 實驗結果中自動找出 qualitative 代表案例"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="指定要讀取的結果檔路徑（可多個）。若未指定，自動掃描 results/sdp/val_5000/ 下所有 JSONL",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=None,
        help="指定 baseline 結果檔路徑（可多個）。若未指定，自動掃描 results/baselines/multiseed/ 下所有 JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="輸出目錄（預設：outputs/）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每類案例最多輸出幾筆（預設：5）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["qwen", "llama"],
        help="只看指定模型的結果（不指定則合併所有模型）",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        default=None,
        help="（保留參數，目前結果檔不含 epsilon 欄位，需透過 analyze_sdp_voting.py 設定）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="（保留參數，目前用 N=100 全量結果做 majority vote）",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        default=False,
        help="只挑選同時有 Baseline-Standard 和 Baseline-Sanitized 資料的問題（確保三方比較完整）",
    )
    return parser.parse_args()


def discover_files(directory: Path, pattern: str = "*.jsonl") -> list[Path]:
    """掃描目錄下符合 pattern 的檔案"""
    return sorted(directory.glob(pattern))


def main():
    args = parse_args()

    print("=" * 60)
    print("  SDP-ICL Qualitative Case Finder")
    print("=" * 60)

    output_dir = Path(args.output_dir)

    # ---- 決定要讀取的 SDP 檔案 ----
    if args.input:
        sdp_files = [Path(p) for p in args.input]
    else:
        sdp_files = discover_files(DEFAULT_SDP_DIR, "*.jsonl")
        print(f"\n[Auto-discover] SDP result files ({DEFAULT_SDP_DIR}):")
        for f in sdp_files:
            print(f"  {f.name}")

    # ---- 決定要讀取的 Baseline 檔案（只取 QA，且只取每個 seed 一個，避免重複） ----
    if args.baseline:
        baseline_files = [Path(p) for p in args.baseline]
    else:
        all_bl = discover_files(DEFAULT_BASELINE_DIR, "*.jsonl")
        # 只取 qa 任務、seed_42（避免 5 個 seed 的資料壓垮記憶體，可自行修改）
        baseline_files = [f for f in all_bl if "qa" in f.name and "seed_42" in f.name]
        print(f"\n[Auto-discover] Baseline QA files (seed_42 only):")
        for f in baseline_files:
            print(f"  {f.name}")

    all_files = sdp_files + baseline_files

    if not all_files:
        print("\n[ERROR] 找不到任何結果檔，請確認路徑正確。")
        sys.exit(1)

    # ---- 讀取並正規化 ----
    print("\n[Step 1] 讀取並正規化結果檔...")
    index = load_and_index(all_files)

    if not index:
        print("[ERROR] 正規化後沒有任何資料，請檢查檔案格式。")
        sys.exit(1)

    # ---- 篩選案例 ----
    print("[Step 2] 篩選代表性案例...")
    cases = select_cases(
        index,
        top_k=args.top_k,
        model_filter=args.model,
        require_baseline=args.require_baseline,
    )
    n_success = len(cases["success"])
    n_partial = len(cases["partial"])
    n_failure = len(cases["failure"])
    print(f"  成功案例：{n_success} 筆")
    print(f"  部分成功：{n_partial} 筆")
    print(f"  失敗案例：{n_failure} 筆")

    # ---- 輸出 ----
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    json_path = output_dir / "qualitative_candidates.json"
    # 將 Path 轉成 str 以便 JSON 序列化
    def path_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    output_data = {
        "meta": {
            "sdp_files": [str(f) for f in sdp_files],
            "baseline_files": [str(f) for f in baseline_files],
            "model_filter": args.model,
            "top_k": args.top_k,
        },
        "cases": {
            ct: [
                {
                    "case_type": c["case_type"],
                    "selection_reason": c["selection_reason"],
                    "sdp_record": {
                        k: v for k, v in c["sdp_record"].items()
                        if k != "ensemble_raw_answers"
                    },
                    "sdp_top_ensemble_answers": format_answer_list(
                        c["sdp_record"].get("ensemble_raw_answers") or []
                    ),
                    "baseline_standard": {
                        k: v for k, v in (c.get("baseline_standard") or {}).items()
                        if k != "ensemble_raw_answers"
                    } if c.get("baseline_standard") else None,
                    "baseline_sanitized": {
                        k: v for k, v in (c.get("baseline_sanitized") or {}).items()
                        if k != "ensemble_raw_answers"
                    } if c.get("baseline_sanitized") else None,
                }
                for c in candidates
            ]
            for ct, candidates in cases.items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] JSON  -> {json_path}")

    # 2. CSV
    csv_path = output_dir / "qualitative_candidates.csv"
    all_flat = []
    for ct in ["success", "partial", "failure"]:
        for cand in cases[ct]:
            all_flat.append(flatten_for_csv(cand))

    if all_flat:
        fieldnames = list(all_flat[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flat)
        print(f"  [SAVED] CSV   -> {csv_path}")

    # 3. Markdown
    md_path = output_dir / "qualitative_candidates.md"
    md_content = render_markdown(cases)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [SAVED] MD    -> {md_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  SDP files loaded       : {len(sdp_files)}")
    print(f"  Baseline files loaded  : {len(baseline_files)}")
    print(f"  Unique question IDs    : {len(index)}")
    print(f"  Success cases (F1>=0.8/EM=1) : {n_success}")
    print(f"  Partial cases (0.4<=F1<0.8)  : {n_partial}")
    print(f"  Failure cases (F1<=0.2)      : {n_failure}")
    print(f"  Output dir             : {output_dir}")
    print("=" * 60)

    if args.epsilon:
        print(
            "\n[NOTE] --epsilon was passed, but the SDP JSONL files store raw"
            " ensemble_raw_answers (N=100). DP noise via Laplace mechanism is"
            " applied in analyze_sdp_voting.py (dp_majority_vote)."
            "\n   This script uses epsilon=inf (pure majority vote) as final answer."
            "\n   To use a specific epsilon, first run analyze_sdp_voting.py to"
            " produce per-question noisy answers, then pass that file here."
        )


if __name__ == "__main__":
    main()
