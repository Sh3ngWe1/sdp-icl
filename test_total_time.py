"""
test_total_time.py - 三種 QA 方法的端到端時間比較測試

方法 1: Baseline RAG (Standard KB)
  - 用戶輸入 → 從 synthetic KB 檢索 → LLM 直接回答
  - 計量：第一個 Token 抵達時間 (TTFT, Time To First Token)

方法 2: DP-ICL (Gaussian 差分隱私)
  - 用戶輸入 → 直接丟給模型跑 N=100 次 → Gaussian 噪音 → 投票輸出
  - 計量：整體時間（從 query 送出到最終答案）

方法 3: SDP-ICL (Laplace 差分隱私 + 去匿名化 + LLM 重組)
  - 用戶輸入 → LLM_1 retrieve sanitized KB → N=5 次推論
  - → Laplace 噪音 → 投票 → [XXX] 映射回原始答案
  - → LLM_2 重組自然語言句子
  - 計量：整體時間（從 query 送出到最終重組句子）
"""

import json
import os
import re
import random
import time
import logging
import asyncio
import numpy as np
import requests
from collections import Counter
from openai import AsyncOpenAI

# ===================================================
# 1. 實驗設定
# ===================================================
MODEL_TYPE = "qwen"                        # 預設測試模型
N_TEST_QUESTIONS = 5                       # 測試題目數量（從 validation set 隨機抽）
K_SHOTS = 3                                # k-shot 範例數

N_DP_ICL = 100                             # 方法 2：DP-ICL 推論次數
N_SDP_ICL = 5                              # 方法 3：SDP-ICL 推論次數
MAX_CONCURRENT = 4                         # 最大並行請求數

# DP 噪音參數
GAUSSIAN_SIGMA = 1.0                       # 方法 2 Gaussian 噪音標準差
LAPLACE_SCALE = 1.0                        # 方法 3 Laplace 噪音 scale

MAX_TOKENS_ANSWER = 50                     # 抽取型答案最大 token
MAX_TOKENS_RECOMPOSE = 150                 # 方法 3 LLM_2 重組句子最大 token

# 資料路徑
SANITIZED_DB_PATH = "data/sanitized/squad_sanitized_train.json"
SYNTHETIC_DB_PATH = "data/synthetic/squad_synthetic_train.json"
TEST_SET_PATH = "data/qa_validation_set_5000.json"
OUTPUT_DIR = "results/timing"

# API 設定
API_BASE_URL = "http://127.0.0.1:1234"
API_V1_URL = f"{API_BASE_URL}/v1"
API_KEY = "lm-studio"
MODEL_NAME_QWEN = "qwen2.5-7b-instruct"
MODEL_NAME_LLAMA = "meta-llama-3.1-8b-instruct"
TARGET_CONTEXT_LENGTH = 8192

random.seed(42)
np.random.seed(42)

# ===================================================
# 2. 初始化
# ===================================================
os.makedirs("logs", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/test_total_time.log", encoding="utf-8"),
    ],
)

client = AsyncOpenAI(base_url=API_V1_URL, api_key=API_KEY)


def get_model_name(model_type: str) -> str:
    if model_type == "qwen":
        return MODEL_NAME_QWEN
    elif model_type == "llama":
        return MODEL_NAME_LLAMA
    raise ValueError(f"不支援的模型類型: {model_type}")


# ===================================================
# 3. 模型載入 / 卸載
# ===================================================
def load_model(model_type: str) -> str | None:
    """載入模型，回傳 instance_id；失敗則回傳 None"""
    model_name = get_model_name(model_type)
    logging.info(f"\n{'='*60}")
    logging.info(f"📥 載入模型: {model_name} (Context={TARGET_CONTEXT_LENGTH})")
    logging.info(f"{'='*60}")
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/models/load",
            json={
                "model": model_name,
                "context_length": TARGET_CONTEXT_LENGTH,
                "flash_attention": True,
                "echo_load_config": True,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            result = resp.json()
            instance_id = result.get("instance_id")
            logging.info(f"✅ 載入成功  instance_id={instance_id}  "
                         f"耗時={result.get('load_time_seconds', 0):.2f}s")
            return instance_id
        logging.error(f"❌ 載入失敗 {resp.status_code}: {resp.text}")
        return None
    except Exception as exc:
        logging.error(f"❌ 載入例外: {exc}")
        return None


def unload_model(instance_id: str | None) -> bool:
    """卸載模型，釋放 GPU 資源"""
    if not instance_id:
        logging.warning("⚠️  無 instance_id，跳過卸載")
        return False
    logging.info(f"\n{'='*60}")
    logging.info("📤 卸載模型，釋放 GPU 資源")
    logging.info(f"{'='*60}")
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/models/unload",
            json={"instance_id": instance_id},
            timeout=30,
        )
        if resp.status_code == 200:
            logging.info("✅ 卸載成功，GPU 資源已釋放")
            return True
        logging.warning(f"⚠️  卸載回應 {resp.status_code}: {resp.text}")
        return False
    except Exception as exc:
        logging.error(f"❌ 卸載例外: {exc}")
        return False


# ===================================================
# 4. 文字工具 & 檢索 (Keyword BM25-lite)
# ===================================================
def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def score_doc(query: str, doc: dict) -> float:
    q_tokens = Counter(tokenize(query))
    if not q_tokens:
        return 0.0
    doc_text = " ".join([
        str(doc.get("question", "")),
        str(doc.get("sanitized_question", "")),
        str(doc.get("context", "")),
        str(doc.get("sanitized_context", "")),
        str(doc.get("answer", "")),
        str(doc.get("sanitized_answer", "")),
    ])
    d_tokens = Counter(tokenize(doc_text))
    return float(sum(min(q_tokens[t], d_tokens[t]) for t in q_tokens))


def retrieve(user_input: str, db: list[dict], top_k: int = K_SHOTS) -> list[dict]:
    """從 db 中取回與 user_input 最相關的 top_k 筆文件"""
    scored = sorted(
        ((score_doc(user_input, doc), doc) for doc in db),
        key=lambda x: x[0],
        reverse=True,
    )
    top = [doc for score, doc in scored[:top_k] if score > 0]
    return top if top else db[:top_k]


# ===================================================
# 5. 去匿名化：[TYPE_N] → 原始答案
# ===================================================
PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+\d+\]")


def deanonymize(voted_answer: str, retrieved_docs: list[dict]) -> str:
    """
    若 voted_answer 包含 [TYPE_N] 佔位符，
    在 retrieved_docs 中尋找 sanitized_answer == voted_answer 的文件，
    回傳對應的 original_answer；否則原樣回傳。
    """
    if not PLACEHOLDER_RE.search(voted_answer):
        return voted_answer

    for doc in retrieved_docs:
        san_ans = doc.get("sanitized_answer", "")
        if san_ans and san_ans.strip() == voted_answer.strip():
            original = doc.get("original_answer", voted_answer)
            logging.info(f"  [De-anon] '{voted_answer}' → '{original}'")
            return original

    # 嘗試部分匹配（佔位符是答案的子字串）
    for doc in retrieved_docs:
        san_ans = doc.get("sanitized_answer", "")
        if san_ans and voted_answer in san_ans:
            original = doc.get("original_answer", voted_answer)
            logging.info(f"  [De-anon partial] '{voted_answer}' → '{original}'")
            return original

    logging.warning(f"  [De-anon] 無法映射 '{voted_answer}'，保留原樣")
    return voted_answer


# ===================================================
# 6. DP 投票機制
# ===================================================
def dp_vote_gaussian(answers: list[str], sigma: float = GAUSSIAN_SIGMA) -> str:
    """對答案列表加 Gaussian 噪音後投票，回傳最高分答案"""
    counter = Counter(a.strip().lower() for a in answers if a.strip())
    if not counter:
        return ""
    noisy_scores = {ans: cnt + np.random.normal(0, sigma) for ans, cnt in counter.items()}
    return max(noisy_scores, key=noisy_scores.get)


def dp_vote_laplace(answers: list[str], scale: float = LAPLACE_SCALE) -> str:
    """對答案列表加 Laplace 噪音後投票，回傳最高分答案"""
    counter = Counter(a.strip().lower() for a in answers if a.strip())
    if not counter:
        return ""
    noisy_scores = {ans: cnt + np.random.laplace(0, scale) for ans, cnt in counter.items()}
    return max(noisy_scores, key=noisy_scores.get)


# ===================================================
# 7. Prompt 建構
# ===================================================
def build_kshot_prompt(question: str, context: str, exemplars: list[dict]) -> str:
    """建構 k-shot 抽取型 QA prompt"""
    prompt = (
        "You are a precise answer extraction system. Extract ONLY the direct answer.\n\n"
        "RULES:\n"
        "- Output the shortest accurate answer (phrase or entity)\n"
        "- NO full sentences, NO explanations\n"
        "- Just the answer, nothing else\n\n"
    )
    for i, ex in enumerate(exemplars, 1):
        ex_ctx = str(ex.get("sanitized_context", ex.get("context", "")))
        ex_q = str(ex.get("sanitized_question", ex.get("question", "")))
        ex_a = str(ex.get("sanitized_answer", ex.get("answer", "")))
        if ex_ctx and ex_q and ex_a:
            prompt += f"Example {i}:\nContext: {ex_ctx}\nQuestion: {ex_q}\nAnswer: {ex_a}\n\n"
    prompt += f"Target Task:\nContext: {context}\nQuestion: {question}\nAnswer:"
    return prompt


def build_rag_prompt(question: str, retrieved_docs: list[dict]) -> str:
    """建構 RAG prompt（方法 1 使用）"""
    blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_ctx = doc.get("context", doc.get("sanitized_context", ""))
        doc_q = doc.get("question", doc.get("sanitized_question", ""))
        doc_a = doc.get("answer", doc.get("sanitized_answer", ""))
        blocks.append(
            f"[Document {i}]\n"
            f"Context: {doc_ctx}\n"
            f"Related Q: {doc_q}\n"
            f"Reference A: {doc_a}"
        )
    joined = "\n\n".join(blocks)
    return (
        "You are a retrieval-augmented QA assistant. "
        "Answer ONLY based on the retrieved context below.\n\n"
        f"Retrieved Context:\n{joined}\n\n"
        f"User Question:\n{question}\n\nAnswer:"
    )


def build_recompose_prompt(question: str, raw_answer: str) -> str:
    """建構 LLM_2 重組句子的 prompt（方法 3 使用）"""
    return (
        "You are a helpful assistant. "
        "Given the question and a short extracted answer, "
        "write a natural, complete sentence that answers the question.\n\n"
        f"Question: {question}\n"
        f"Extracted Answer: {raw_answer}\n\n"
        "Natural Answer:"
    )


# ===================================================
# 8. API 呼叫函數
# ===================================================
async def call_api(prompt: str, model_type: str, max_tokens: int = MAX_TOKENS_ANSWER) -> str:
    """一般非串流 API 呼叫"""
    model_name = get_model_name(model_type)
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful QA assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logging.error(f"API 呼叫失敗: {exc}")
        return ""


async def call_api_stream_ttft(
    prompt: str,
    model_type: str,
    max_tokens: int = MAX_TOKENS_ANSWER,
) -> tuple[float, str]:
    """
    串流 API 呼叫，回傳 (ttft_seconds, full_answer)
    ttft_seconds = 送出請求 → 第一個 token 抵達的時間
    """
    model_name = get_model_name(model_type)
    start = time.perf_counter()
    ttft = None
    full_text = []

    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful QA assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                full_text.append(delta_content)

    except Exception as exc:
        logging.error(f"串流 API 呼叫失敗: {exc}")
        return (time.perf_counter() - start, "")

    return (ttft if ttft is not None else time.perf_counter() - start, "".join(full_text).strip())


async def single_inference(
    semaphore: asyncio.Semaphore,
    n: int,
    prompt: str,
    model_type: str,
) -> tuple[str, float]:
    """帶 Semaphore 的單次推論（用於並行批次）"""
    async with semaphore:
        t0 = time.perf_counter()
        answer = await call_api(prompt, model_type)
        return answer, round(time.perf_counter() - t0, 4)


# ===================================================
# 9. 方法 1：Baseline RAG — 計量 TTFT
# ===================================================
async def method1_baseline_rag(
    question: str,
    context: str,
    synthetic_db: list[dict],
    model_type: str,
) -> dict:
    """
    Standard RAG 流程：
    1. 從 synthetic KB 取回相關文件
    2. 組成 RAG prompt
    3. 串流呼叫 LLM，記錄 TTFT
    """
    t_start = time.perf_counter()

    # Step 1: 檢索
    t_retrieve_start = time.perf_counter()
    user_input = f"{question} {context[:200]}"
    retrieved = retrieve(user_input, synthetic_db)
    t_retrieve = round(time.perf_counter() - t_retrieve_start, 4)

    # Step 2: 建立 RAG prompt
    prompt = build_rag_prompt(question, retrieved)

    # Step 3: 串流呼叫，取得 TTFT
    t_llm_start = time.perf_counter()
    ttft, answer = await call_api_stream_ttft(prompt, model_type, MAX_TOKENS_ANSWER)
    t_llm_total = round(time.perf_counter() - t_llm_start, 4)

    t_total = round(time.perf_counter() - t_start, 4)

    logging.info(f"  [方法1] TTFT={ttft:.4f}s  檢索={t_retrieve}s  LLM={t_llm_total}s  總計={t_total}s")
    logging.info(f"  [方法1] 答案: {answer[:100]}")

    return {
        "method": "method1_baseline_rag",
        "ttft_seconds": round(ttft, 4),
        "retrieve_seconds": t_retrieve,
        "llm_total_seconds": t_llm_total,
        "total_seconds": t_total,
        "answer": answer,
    }


# ===================================================
# 10. 方法 2：DP-ICL — N=100 + Gaussian 噪音，計量整體時間
# ===================================================
async def method2_dp_icl(
    question: str,
    context: str,
    synthetic_db: list[dict],
    model_type: str,
) -> dict:
    """
    DP-ICL 流程：
    1. N=100 次平行推論，每次隨機抽 k-shot 範例（Standard KB）
    2. 對答案計數加 Gaussian 噪音
    3. 取最高分答案
    """
    t_start = time.perf_counter()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # 建立 N=100 個任務，每次隨機 k-shot
    tasks = []
    for n in range(N_DP_ICL):
        exemplars = random.sample(synthetic_db, min(K_SHOTS, len(synthetic_db)))
        prompt = build_kshot_prompt(question, context, exemplars)
        tasks.append(single_inference(semaphore, n, prompt, model_type))

    t_inference_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    t_inference = round(time.perf_counter() - t_inference_start, 4)

    # 整理答案
    answers = []
    for r in results:
        if isinstance(r, tuple):
            answers.append(r[0])
        else:
            answers.append("")

    # Gaussian 噪音投票
    t_vote_start = time.perf_counter()
    final_answer = dp_vote_gaussian(answers, GAUSSIAN_SIGMA)
    t_vote = round(time.perf_counter() - t_vote_start, 6)

    t_total = round(time.perf_counter() - t_start, 4)
    success_count = sum(1 for a in answers if a.strip())

    logging.info(f"  [方法2] N={N_DP_ICL}  成功={success_count}/{N_DP_ICL}  "
                 f"推論={t_inference}s  投票={t_vote}s  總計={t_total}s")
    logging.info(f"  [方法2] 最終答案: {final_answer}")

    return {
        "method": "method2_dp_icl",
        "n_inferences": N_DP_ICL,
        "success_count": success_count,
        "inference_seconds": t_inference,
        "vote_seconds": t_vote,
        "total_seconds": t_total,
        "final_answer": final_answer,
        "gaussian_sigma": GAUSSIAN_SIGMA,
    }


# ===================================================
# 11. 方法 3：SDP-ICL — N=5 + Laplace + 去匿名 + LLM_2 重組
# ===================================================
async def method3_sdp_icl(
    question: str,
    context: str,
    sanitized_db: list[dict],
    model_type: str,
) -> dict:
    """
    SDP-ICL 流程：
    1. 從 sanitized KB 取回相關文件（去敏化版 k-shot）
    2. LLM_1 做 N=5 次並行推論
    3. Laplace 噪音投票，取最終答案
    4. 若答案含 [XXX] 佔位符，映射回原始值
    5. LLM_2 將原始問題 + 答案重組為自然語言
    """
    t_start = time.perf_counter()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Step 1: 從 sanitized KB 檢索
    t_retrieve_start = time.perf_counter()
    user_input = f"{question} {context[:200]}"
    retrieved_docs = retrieve(user_input, sanitized_db)
    t_retrieve = round(time.perf_counter() - t_retrieve_start, 4)

    # Step 2: LLM_1 做 N=5 次推論（使用 sanitized exemplars）
    tasks = []
    for n in range(N_SDP_ICL):
        exemplars = random.sample(sanitized_db, min(K_SHOTS, len(sanitized_db)))
        prompt = build_kshot_prompt(question, context, exemplars)
        tasks.append(single_inference(semaphore, n, prompt, model_type))

    t_inference_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    t_inference = round(time.perf_counter() - t_inference_start, 4)

    answers = [r[0] if isinstance(r, tuple) else "" for r in results]

    # Step 3: Laplace 噪音投票
    t_vote_start = time.perf_counter()
    voted_answer = dp_vote_laplace(answers, LAPLACE_SCALE)
    t_vote = round(time.perf_counter() - t_vote_start, 6)

    # Step 4: 去匿名化 [XXX] → 原始值
    t_deanon_start = time.perf_counter()
    deanon_answer = deanonymize(voted_answer, retrieved_docs)
    t_deanon = round(time.perf_counter() - t_deanon_start, 6)

    # Step 5: LLM_2 重組自然語言句子
    recompose_prompt = build_recompose_prompt(question, deanon_answer)
    t_recompose_start = time.perf_counter()
    final_answer = await call_api(recompose_prompt, model_type, MAX_TOKENS_RECOMPOSE)
    t_recompose = round(time.perf_counter() - t_recompose_start, 4)

    t_total = round(time.perf_counter() - t_start, 4)
    success_count = sum(1 for a in answers if a.strip())

    logging.info(
        f"  [方法3] N={N_SDP_ICL}  成功={success_count}/{N_SDP_ICL}  "
        f"檢索={t_retrieve}s  推論={t_inference}s  "
        f"投票={t_vote}s  去匿名={t_deanon}s  重組={t_recompose}s  總計={t_total}s"
    )
    logging.info(f"  [方法3] voted='{voted_answer}'  deanon='{deanon_answer}'")
    logging.info(f"  [方法3] 最終重組: {final_answer[:120]}")

    return {
        "method": "method3_sdp_icl",
        "n_inferences": N_SDP_ICL,
        "success_count": success_count,
        "retrieve_seconds": t_retrieve,
        "inference_seconds": t_inference,
        "vote_seconds": t_vote,
        "deanon_seconds": t_deanon,
        "recompose_seconds": t_recompose,
        "total_seconds": t_total,
        "voted_answer": voted_answer,
        "deanon_answer": deanon_answer,
        "final_answer": final_answer,
        "laplace_scale": LAPLACE_SCALE,
    }


# ===================================================
# 12. 主程式
# ===================================================
async def main():
    logging.info("\n" + "=" * 60)
    logging.info("🚀 開始三種 QA 方法時間比較測試")
    logging.info("=" * 60)

    # 載入資料集
    with open(SANITIZED_DB_PATH, encoding="utf-8") as f:
        sanitized_db = json.load(f)
    with open(SYNTHETIC_DB_PATH, encoding="utf-8") as f:
        synthetic_db = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    # 隨機抽取測試題
    test_questions = random.sample(test_set, min(N_TEST_QUESTIONS, len(test_set)))
    logging.info(f"測試題目數: {len(test_questions)}")
    logging.info(f"Sanitized DB: {len(sanitized_db)} 筆  Synthetic DB: {len(synthetic_db)} 筆")

    all_results = []
    instance_id = None

    try:
        # 每個方法分開 load/unload 模型，確保公平計時
        for method_idx, method_name in enumerate(
            ["method1_baseline_rag", "method2_dp_icl", "method3_sdp_icl"], start=1
        ):
            logging.info(f"\n{'='*60}")
            logging.info(f"▶  方法 {method_idx}: {method_name}")
            logging.info(f"{'='*60}")

            # 載入模型
            instance_id = load_model(MODEL_TYPE)
            if not instance_id:
                logging.error(f"❌ 模型載入失敗，跳過 {method_name}")
                continue

            logging.info("⏳ 等待 3 秒讓模型完全就緒...")
            await asyncio.sleep(3)

            method_results = []

            try:
                for q_idx, test_item in enumerate(test_questions, start=1):
                    question = test_item.get("question", "")
                    context = test_item.get("context", "")
                    ground_truth = test_item.get("ground_truth_answers", [])
                    item_id = test_item.get("id", str(q_idx))

                    logging.info(f"\n  題目 {q_idx}/{len(test_questions)}: {question[:80]}")

                    if method_idx == 1:
                        result = await method1_baseline_rag(
                            question, context, synthetic_db, MODEL_TYPE
                        )
                    elif method_idx == 2:
                        result = await method2_dp_icl(
                            question, context, synthetic_db, MODEL_TYPE
                        )
                    else:
                        result = await method3_sdp_icl(
                            question, context, sanitized_db, MODEL_TYPE
                        )

                    result["question_id"] = item_id
                    result["question"] = question
                    result["ground_truth"] = ground_truth
                    method_results.append(result)

            finally:
                # 每個方法結束後立即卸載模型
                unload_model(instance_id)
                instance_id = None
                logging.info("⏳ 等待 2 秒讓 GPU 完全釋放...")
                await asyncio.sleep(2)

            all_results.extend(method_results)
            _print_method_summary(method_name, method_results)

        # 儲存結果
        _save_results(all_results)
        _print_final_summary(all_results)

    except KeyboardInterrupt:
        logging.warning("\n⚠️  使用者中斷執行")
    except Exception as exc:
        logging.error(f"\n❌ 執行例外: {exc}", exc_info=True)
    finally:
        if instance_id:
            logging.info("🧹 清理殘留模型...")
            unload_model(instance_id)
        if all_results:
            _save_results(all_results)
        logging.info("✅ 程式執行完畢")


# ===================================================
# 13. 結果輸出
# ===================================================
def _print_method_summary(method_name: str, results: list[dict]):
    logging.info(f"\n{'─'*60}")
    logging.info(f"📊 {method_name} 摘要（{len(results)} 題）")
    logging.info(f"{'─'*60}")

    if not results:
        logging.info("  無結果")
        return

    if method_name == "method1_baseline_rag":
        ttfts = [r["ttft_seconds"] for r in results]
        totals = [r["total_seconds"] for r in results]
        logging.info(f"  TTFT  avg={np.mean(ttfts):.4f}s  min={np.min(ttfts):.4f}s  max={np.max(ttfts):.4f}s")
        logging.info(f"  Total avg={np.mean(totals):.4f}s  min={np.min(totals):.4f}s  max={np.max(totals):.4f}s")

    elif method_name == "method2_dp_icl":
        totals = [r["total_seconds"] for r in results]
        inferences = [r["inference_seconds"] for r in results]
        logging.info(f"  推論  avg={np.mean(inferences):.4f}s  min={np.min(inferences):.4f}s  max={np.max(inferences):.4f}s")
        logging.info(f"  Total avg={np.mean(totals):.4f}s  min={np.min(totals):.4f}s  max={np.max(totals):.4f}s")

    elif method_name == "method3_sdp_icl":
        totals = [r["total_seconds"] for r in results]
        inferences = [r["inference_seconds"] for r in results]
        recomposes = [r["recompose_seconds"] for r in results]
        logging.info(f"  推論  avg={np.mean(inferences):.4f}s  min={np.min(inferences):.4f}s  max={np.max(inferences):.4f}s")
        logging.info(f"  重組  avg={np.mean(recomposes):.4f}s  min={np.min(recomposes):.4f}s  max={np.max(recomposes):.4f}s")
        logging.info(f"  Total avg={np.mean(totals):.4f}s  min={np.min(totals):.4f}s  max={np.max(totals):.4f}s")


def _print_final_summary(all_results: list[dict]):
    logging.info(f"\n{'='*60}")
    logging.info("📊 最終時間比較摘要")
    logging.info(f"{'='*60}")

    for method_key in ["method1_baseline_rag", "method2_dp_icl", "method3_sdp_icl"]:
        res = [r for r in all_results if r["method"] == method_key]
        if not res:
            continue
        totals = [r["total_seconds"] for r in res]
        label = {
            "method1_baseline_rag": "方法1 Baseline RAG (TTFT)",
            "method2_dp_icl":       "方法2 DP-ICL (N=100)",
            "method3_sdp_icl":      "方法3 SDP-ICL (N=5)",
        }[method_key]

        if method_key == "method1_baseline_rag":
            ttfts = [r["ttft_seconds"] for r in res]
            logging.info(f"\n  {label}")
            logging.info(f"    TTFT avg: {np.mean(ttfts):.4f}s")
            logging.info(f"    Total avg: {np.mean(totals):.4f}s")
        else:
            logging.info(f"\n  {label}")
            logging.info(f"    Total avg: {np.mean(totals):.4f}s")


def _save_results(all_results: list[dict]):
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"timing_comparison_{ts}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "generated_at": ts,
        "config": {
            "model_type": MODEL_TYPE,
            "n_test_questions": N_TEST_QUESTIONS,
            "k_shots": K_SHOTS,
            "n_dp_icl": N_DP_ICL,
            "n_sdp_icl": N_SDP_ICL,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "laplace_scale": LAPLACE_SCALE,
            "max_concurrent": MAX_CONCURRENT,
        },
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"\n💾 結果已儲存至: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
