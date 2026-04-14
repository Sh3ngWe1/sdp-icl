"""
run_sdp_inference_dgx_v1.py - Week 3: SDP-ICL 推論資料收集 (DGX API 狂暴版)

主要功能：
1. 透過本地 llmster API (OpenAI 相容格式) 呼叫 DGX 上的大模型。
2. 針對 QA 任務，進行 N=100 的隨機子採樣推論 (Subsample-and-Aggregate 的前半段)。
3. 精準記錄每一次推論的耗時，用於後續的成本分析。
4. 完全免除本地 GPU 記憶體管理負擔。
5. 使用 JSONL 格式即時寫入，大幅降低 RAM 佔用。
6. 自動載入 8K context 模型（避免 token limit 錯誤）。
7. 4 個併行推論處理（約 2x 加速）。
8. 執行完成後自動卸載模型，釋放 GPU 資源。
"""

import json
import os
import random
import time
import logging
import asyncio
import requests
from openai import AsyncOpenAI

# ================= 1. 實驗設定區 =================
# 🚀 SDP-ICL 核心配置 Llama QA (Sanitized & Standard)
EXPERIMENT_CONFIGS = [
    {
        "task_name": "sdp_qa_llama_sanitized_n100_5000",
        "model_type": "llama",
        "exemplar_db_path": "data/sanitized/squad_sanitized_train.json",
        "test_set_path": "data/qa_validation_set_5000.json",
        "description": "🧠 SDP Inference: Llama-3.1-8B (N=100, Sanitized)"
    },
    {
        "task_name": "sdp_qa_llama_standard_n100_5000",
        "model_type": "llama",
        "exemplar_db_path": "data/synthetic/squad_synthetic_train.json",
        "test_set_path": "data/qa_validation_set_5000.json",
        "description": "⚖️ SDP 對照組: Llama-3.1-8B (N=100, Standard)"
    },
    {
        "task_name": "sdp_qa_qwen_sanitized_n100_5000",
        "model_type": "qwen",
        "exemplar_db_path": "data/sanitized/squad_sanitized_train.json",    
        "test_set_path": "data/qa_validation_set_5000.json",
        "description": "🧠 SDP Inference: Qwen-2.5-7B (N=100, Sanitized)"
    },
    {
        "task_name": "sdp_qa_qwen_standard_n100_5000",
        "model_type": "qwen",
        "exemplar_db_path": "data/synthetic/squad_synthetic_train.json",
        "test_set_path": "data/qa_validation_set_5000.json",
        "description": "⚖️ SDP 對照組: Qwen-2.5-7B (N=100, Standard)"
    }
]


# 通用設定
N_ENSEMBLES = 100   # 🔥 直接跑滿 100 次
K_SHOTS = 3
MAX_NEW_TOKENS = 50 # 短抽取任務，限制 50 個 Token 大幅加速
MAX_CONCURRENT = 4  # 🚀 最多同時 4 個並行請求

# API 伺服器設定
API_BASE_URL = "http://127.0.0.1:1234"
API_V1_URL = f"{API_BASE_URL}/v1"
API_KEY = "lm-studio"
LLMSTER_MODEL_NAME_QWEN = "qwen2.5-7b-instruct" # ⚠️ 請確保這與你在 llmster 載入的模型名稱完全一致
LLMSTER_MODEL_NAME_LLAMA = "meta-llama-3.1-8b-instruct"
TARGET_CONTEXT_LENGTH = 8192  # 🎯 設定 8K context window（避免 token limit 錯誤）

random.seed(42)

# 設定 logging 目錄
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 建立 logger（只設定 console handler，file handler 會在每個實驗開始時動態添加）
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 設定 formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console Handler（所有實驗共用）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 初始化 API 客戶端（改用 AsyncOpenAI）
client = AsyncOpenAI(base_url=API_V1_URL, api_key=API_KEY)

# 全域變數：追蹤當前的 file handler
current_file_handler = None

# ================= 2. 模型管理函數 =================
def load_model_8k(model_type):
    """載入 8K context 模型"""
    logging.info(f"\n{'='*60}")
    logging.info("📥 載入模型 (8K Context)")
    logging.info(f"{'='*60}")
    
    try:
        if model_type == "qwen":
            model_name = LLMSTER_MODEL_NAME_QWEN
        elif model_type == "llama":
            model_name = LLMSTER_MODEL_NAME_LLAMA
        else:
            raise ValueError(f"不支援的模型: {model_type}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/models/load",
            json={
                "model": model_name,
                "context_length": TARGET_CONTEXT_LENGTH,
                "flash_attention": True,
                "echo_load_config": True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"✅ 載入成功！")
            logging.info(f"   Instance ID: {result.get('instance_id')}")
            logging.info(f"   載入耗時: {result.get('load_time_seconds', 0):.2f}s")
            
            if 'load_config' in result:
                config = result['load_config']
                logging.info(f"   Context Length: {config.get('context_length')}")
                logging.info(f"   Flash Attention: {config.get('flash_attention')}")
                logging.info(f"   Eval Batch Size: {config.get('eval_batch_size')}")
            
            return result.get('instance_id')
        else:
            logging.error(f"❌ 載入失敗: {response.status_code}")
            logging.error(f"   {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"❌ 載入失敗: {e}")
        return None

def unload_model(instance_id):
    """卸載模型，釋放資源"""
    logging.info(f"\n{'='*60}")
    logging.info("📤 卸載模型，釋放 GPU 資源")
    logging.info(f"{'='*60}")
    
    if not instance_id:
        logging.warning("⚠️  無 instance_id，跳過卸載")
        return False
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/models/unload",
            json={"instance_id": instance_id},
            timeout=30
        )
        
        if response.status_code == 200:
            logging.info(f"✅ 卸載成功！GPU 資源已釋放")
            return True
        else:
            logging.warning(f"⚠️  卸載回應: {response.status_code}")
            logging.warning(f"   {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ 卸載失敗: {e}")
        return False

# ================= 3. Prompt 建構 =================
def build_prompt(question, context, exemplars, model_type):
    """建構 prompt，帶錯誤處理"""
    try:
        # Qwen 最佳策略：v2 優化版（帶正反例示範）
        if model_type in ["qwen", "llama"]:
            prompt = "You are a precise answer extraction system. Extract ONLY the direct answer.\n\n"
            prompt += "RULES:\n"
            prompt += "- Output the shortest accurate answer (phrase or entity)\n"
            prompt += "- NO full sentences, NO explanations, NO meta-commentary\n"
            prompt += "- Just the answer, nothing else\n\n"
            prompt += "CORRECT Examples:\n"
            prompt += "Q: Where is Paris located? → France\n"
            prompt += "Q: What color is the sky? → blue\n"
            prompt += "Q: Who invented the telephone? → Alexander Graham Bell\n\n"
            prompt += "WRONG Examples (DO NOT DO THIS):\n"
            prompt += "❌ 'Paris is located in France.'\n"
            prompt += "❌ 'The answer is France.'\n"
            prompt += "❌ 'Based on the context, France.'\n\n"
        else:
            prompt = "You are a helpful assistant. Please answer the question based on the provided context.\n\n"
        
        # 加入 k-shot 範例（帶防禦性檢查）
        for i, ex in enumerate(exemplars):
            # 防禦性取得欄位，確保是字串
            ex_context = str(ex.get('sanitized_context', ex.get('context', '')))
            ex_question = str(ex.get('sanitized_question', ex.get('question', '')))
            ex_answer = str(ex.get('sanitized_answer', ex.get('answer', '')))
            
            # 如果欄位為空，記錄警告但繼續
            if not ex_context or not ex_question or not ex_answer:
                logging.warning(f"⚠️  Exemplar {i+1} 有空欄位，跳過")
                continue
            
            prompt += f"Example {i+1}:\nContext: {ex_context}\nQuestion: {ex_question}\nAnswer: {ex_answer}\n\n"
        
        # 目標任務（防禦性轉換）
        question = str(question) if question else ""
        context = str(context) if context else ""
        prompt += f"Target Task:\nContext: {context}\nQuestion: {question}\nAnswer:"
        
        return prompt
        
    except Exception as e:
        logging.error(f"❌ build_prompt 失敗: {e}")
        # 返回一個最簡單的 prompt
        return f"Question: {question}\nAnswer:"

# ================= 4. 推論引擎 (API 呼叫版 - Async) =================
async def generate_with_api(prompt, model_type, max_retries=3):
    """透過 llmster API 取得模型回應（異步版本）"""
    for attempt in range(max_retries):
        try:
            if model_type == "qwen":
                model_name = LLMSTER_MODEL_NAME_QWEN
            elif model_type == "llama":
                model_name = LLMSTER_MODEL_NAME_LLAMA
            else:
                raise ValueError(f"不支援的模型: {model_type}")
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"You are a precise answer extraction system for {model_type}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # 設為 0.0 確保抽取結果穩定 (相當於之前的 do_sample=False)
                max_tokens=MAX_NEW_TOKENS
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"API 呼叫失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return ""
            await asyncio.sleep(2)
    return ""

# ================= 5. 單次推論任務 (用於平行化) =================
async def single_inference_task(semaphore, n, question, context, exemplar_db, model_type):
    """執行單次推論任務，帶並發控制和完整錯誤處理"""
    async with semaphore:  # 限制並發數量
        start_time = time.perf_counter()
        
        try:
            # 1. 隨機抽樣 3 篇淨化文章（防禦性檢查）
            if len(exemplar_db) < K_SHOTS:
                logging.error(f"❌ exemplar_db 只有 {len(exemplar_db)} 筆，少於 K_SHOTS={K_SHOTS}")
                return "", round(time.perf_counter() - start_time, 4)
            
            sampled_exemplars = random.sample(exemplar_db, K_SHOTS)
            
            # 2. 組合 Prompt（已有內部錯誤處理）
            prompt = build_prompt(question, context, sampled_exemplars, model_type)
            
            # 3. 呼叫 API
            answer = await generate_with_api(prompt, model_type)
            end_time = time.perf_counter()
            
            # 4. 返回結果與時間
            time_taken = round(end_time - start_time, 4)
            return answer, time_taken
            
        except ValueError as e:
            # random.sample 失敗
            logging.error(f"❌ 抽樣失敗 (推論 {n+1}): {e}")
            return "", round(time.perf_counter() - start_time, 4)
            
        except Exception as e:
            # 任何其他錯誤
            logging.error(f"❌ 推論任務失敗 (推論 {n+1}): {e}")
            return "", round(time.perf_counter() - start_time, 4)

# ================= 6. 主迴圈 (改用 JSONL + Asyncio) =================
async def run_single_experiment(config):
    global current_file_handler
    
    task_name = config['task_name']
    model_type = config['model_type']
    exemplar_db_path = config['exemplar_db_path']
    test_set_path = config['test_set_path']
    output_path = f"results/sdp/val_5000/{task_name}_results.jsonl"  # 改用 .jsonl 格式
    
    # 🆕 為這個實驗設定獨立的 log 檔案
    log_file = os.path.join(log_dir, f"{task_name}.log")
    
    # 移除舊的 file handler（如果存在）
    if current_file_handler:
        logger.removeHandler(current_file_handler)
        current_file_handler.close()
    
    # 建立新的 file handler
    current_file_handler = logging.FileHandler(log_file, encoding='utf-8')
    current_file_handler.setLevel(logging.INFO)
    current_file_handler.setFormatter(formatter)
    logger.addHandler(current_file_handler)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"📝 Log 檔案: {log_file}")
    logging.info(f"{config['description']}")
    logging.info(f"{'='*60}")
    
    if not os.path.exists(os.path.dirname(output_path)): 
        os.makedirs(os.path.dirname(output_path))
    
    with open(exemplar_db_path, "r", encoding="utf-8") as f: 
        exemplar_db = json.load(f)
    with open(test_set_path, "r", encoding="utf-8") as f: 
        test_set = json.load(f)
    
    # 🆕 防禦性檢查：確保 exemplar_db 足夠
    if len(exemplar_db) < K_SHOTS:
        logging.error(f"❌ exemplar_db 只有 {len(exemplar_db)} 筆，少於 K_SHOTS={K_SHOTS}，無法執行")
        return {
            "task_name": task_name,
            "error_records": [],
            "summary": {
                "total_questions_processed": 0,
                "questions_with_errors": 0,
                "total_inferences": 0,
                "total_failures": 0,
                "overall_failure_rate": 0
            }
        }

    # 讀取已處理的題目 ID（從 JSONL 檔案，支援斷電後斷點續傳）
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    processed_ids.add(item.get('id'))
                except json.JSONDecodeError as e:
                    # 斷電時最後一行可能寫入不完整，跳過損壞行避免啟動失敗
                    logging.warning(f"⚠️ JSONL 第 {line_num} 行損壞（可能斷電中斷），跳過: {e}")
    
    logging.info(f"開始執行 API 推論 ({task_name})...")
    logging.info(f"已處理: {len(processed_ids)}/{len(test_set)}")
    
    # 建立 Semaphore 來限制並發數
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 🆕 追蹤錯誤統計
    error_records = []
    total_questions_processed = 0
    total_inferences = 0
    total_failures = 0
    
    for idx, test_item in enumerate(test_set):
        item_id = test_item.get('id', str(idx))
        if item_id in processed_ids: continue
        
        # 🆕 整題包在 try/except 中，避免單題爆炸影響後續題目
        try:
            context = test_item.get('context', "")
            question = test_item.get('question', "")
            
            logging.info(f"處理問題 {idx + 1}/{len(test_set)} ... (執行 {N_ENSEMBLES} 次抽樣推論，{MAX_CONCURRENT} 個並行)")
            
            # 建立 100 個異步任務
            tasks = [
                single_inference_task(semaphore, n, question, context, exemplar_db, model_type)
                for n in range(N_ENSEMBLES)
            ]
            
            # 🆕 平行執行所有任務（return_exceptions=True 避免單個任務失敗炸整題）
            question_start_time = time.perf_counter()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            question_end_time = time.perf_counter()
            
            # 🆕 處理結果（包括可能的 Exception）
            ensemble_raw_answers = []
            ensemble_times = []
            
            for result in batch_results:
                if isinstance(result, Exception):
                    # 如果是 Exception，記錄錯誤並視為失敗
                    logging.error(f"❌ 單次推論返回 Exception: {result}")
                    ensemble_raw_answers.append("")
                    ensemble_times.append(0.0)
                elif isinstance(result, tuple) and len(result) == 2:
                    # 正常結果
                    answer, time_taken = result
                    ensemble_raw_answers.append(answer)
                    ensemble_times.append(time_taken)
                else:
                    # 非預期的結果格式
                    logging.error(f"❌ 非預期的結果格式: {type(result)}")
                    ensemble_raw_answers.append("")
                    ensemble_times.append(0.0)
            
            # 🆕 統計失敗次數
            fail_count = sum(1 for answer in ensemble_raw_answers if answer == "")
            success_count = N_ENSEMBLES - fail_count
            
            total_time = round(question_end_time - question_start_time, 2)
            avg_time = round(sum(ensemble_times) / len(ensemble_times), 4) if ensemble_times else 0.0
            
            # 🆕 顯示成功率
            if fail_count > 0:
                logging.warning(f"  -> 完成 {N_ENSEMBLES} 次推論，成功: {success_count}/{N_ENSEMBLES} ({success_count/N_ENSEMBLES*100:.1f}%)，失敗: {fail_count}")
                logging.warning(f"     總耗時: {total_time}s，平均單次: {avg_time}s")
                
                # 🆕 記錄錯誤資訊
                error_records.append({
                    "question_id": item_id,
                    "question_index": idx + 1,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "context_length": len(context),
                    "total_inferences": N_ENSEMBLES,
                    "successful_inferences": success_count,
                    "failed_inferences": fail_count,
                    "failure_rate": round(fail_count / N_ENSEMBLES * 100, 2),
                    "avg_inference_time": avg_time,
                    "total_time": total_time
                })
            else:
                logging.info(f"  -> 完成 {N_ENSEMBLES} 次推論，全部成功！總耗時: {total_time}s，平均單次: {avg_time}s")
            
            # 🆕 累積統計
            total_questions_processed += 1
            total_inferences += N_ENSEMBLES
            total_failures += fail_count

            # 將該題的 100 個答案與時間存入結果中
            result_entry = test_item.copy()
            result_entry['ensemble_raw_answers'] = ensemble_raw_answers
            result_entry['ensemble_times'] = ensemble_times
            result_entry['success_count'] = success_count  # 🆕 記錄成功次數
            result_entry['fail_count'] = fail_count  # 🆕 記錄失敗次數
            
            # 立即寫入 JSONL（每題寫一行，節省記憶體）
            # flush + fsync 確保斷電時已完成的題目能寫入磁碟，支援斷點續傳
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            logging.info(f"✅ 已完成並儲存: {idx + 1}/{len(test_set)}")
            
        except Exception as e:
            # 🆕 整題處理失敗的容錯機制
            logging.error(f"❌ 問題 {idx + 1} 整題處理失敗: {e}")
            logging.error(f"   問題 ID: {item_id}")
            logging.error(f"   將記錄為全失敗並繼續下一題")
            
            # 記錄這題為完全失敗
            error_records.append({
                "question_id": item_id,
                "question_index": idx + 1,
                "question": str(test_item.get('question', ''))[:100],
                "context_length": len(str(test_item.get('context', ''))),
                "total_inferences": N_ENSEMBLES,
                "successful_inferences": 0,
                "failed_inferences": N_ENSEMBLES,
                "failure_rate": 100.0,
                "error_message": str(e),
                "avg_inference_time": 0.0,
                "total_time": 0.0
            })
            
            total_questions_processed += 1
            total_inferences += N_ENSEMBLES
            total_failures += N_ENSEMBLES
            
            # 嘗試寫入一個失敗記錄
            try:
                result_entry = test_item.copy()
                result_entry['ensemble_raw_answers'] = [""] * N_ENSEMBLES
                result_entry['ensemble_times'] = [0.0] * N_ENSEMBLES
                result_entry['success_count'] = 0
                result_entry['fail_count'] = N_ENSEMBLES
                result_entry['error'] = str(e)
                
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    
                logging.info(f"⚠️  已記錄失敗題目: {idx + 1}/{len(test_set)}")
            except Exception as write_error:
                logging.error(f"❌ 無法寫入失敗記錄: {write_error}")
                
            # 繼續處理下一題
            continue

    logging.info(f"🎉 {task_name} 完美收官！結果已儲存至: {output_path}\n")
    
    # 🆕 統計資訊
    error_stats = {
        "task_name": task_name,
        "error_records": error_records,
        "summary": {
            "total_questions_processed": total_questions_processed,
            "questions_with_errors": len(error_records),
            "total_inferences": total_inferences,
            "total_failures": total_failures,
            "overall_failure_rate": round(total_failures / total_inferences * 100, 2) if total_inferences > 0 else 0
        }
    }
    
    # 🆕 為這個實驗生成獨立的 error report
    generate_single_error_report(error_stats, task_name)
    
    return error_stats

async def main():
    """主程式進入點"""
    logging.info("\n🚀 開始 SDP-ICL 推論實驗")
    
    # 收集所有實驗的錯誤統計
    all_error_stats = []
    instance_id = None
    
    try:
        for config in EXPERIMENT_CONFIGS:
            model_type = config['model_type']
            
            # 1. 該任務開始：載入對應模型
            instance_id = load_model_8k(model_type)
            if not instance_id:
                logging.error(f"❌ 模型載入失敗 ({config['task_name']})，跳過此任務")
                continue
            
            logging.info("\n⏳ 等待 3 秒讓模型完全就緒...")
            await asyncio.sleep(3)
            
            try:
                # 2. 執行該任務的實驗
                error_stats = await run_single_experiment(config)
                all_error_stats.append(error_stats)
            finally:
                # 3. 該任務結束：卸載模型，釋放 GPU 資源
                unload_model(instance_id)
                instance_id = None
        
        logging.info("\n🎉 所有實驗完成！")
        
        # 4. 生成總體錯誤報告
        generate_summary_error_report(all_error_stats)
        
    except KeyboardInterrupt:
        logging.warning("\n⚠️  使用者中斷執行")
        if all_error_stats:
            generate_summary_error_report(all_error_stats)
        
    except Exception as e:
        logging.error(f"\n❌ 執行過程發生錯誤: {e}")
        if all_error_stats:
            generate_summary_error_report(all_error_stats)
        
    finally:
        # 清理 file handler
        global current_file_handler
        if current_file_handler:
            logger.removeHandler(current_file_handler)
            current_file_handler.close()
            current_file_handler = None
        
        # 若中斷時模型仍在載入，則卸載
        if instance_id:
            logging.info("\n🧹 清理資源...")
            unload_model(instance_id)
        
        logging.info("✅ 程式執行完畢，資源已釋放")

def generate_single_error_report(error_stats, task_name):
    """為單個實驗生成錯誤報告"""
    logging.info("\n" + "="*60)
    logging.info(f"📊 {task_name} 錯誤統計")
    logging.info("="*60)
    
    summary = error_stats['summary']
    logging.info(f"處理題數: {summary['total_questions_processed']}")
    logging.info(f"有錯誤的題數: {summary['questions_with_errors']}")
    logging.info(f"總推論次數: {summary['total_inferences']:,}")
    logging.info(f"總失敗次數: {summary['total_failures']:,}")
    logging.info(f"失敗率: {summary['overall_failure_rate']:.2f}%")
    
    if summary['questions_with_errors'] > 0:
        logging.info(f"\n錯誤題目:")
        for error in error_stats['error_records'][:5]:
            logging.info(f"   - 問題 {error['question_index']}: {error['failed_inferences']}/{error['total_inferences']} 失敗 ({error['failure_rate']:.1f}%)")
        if len(error_stats['error_records']) > 5:
            logging.info(f"   ... 還有 {len(error_stats['error_records']) - 5} 個錯誤題目")
    
    # 儲存單個實驗的 error report
    error_report = {
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "task_name": task_name,
        "summary": summary,
        "error_details": error_stats['error_records']
    }
    
    error_report_path = f"results/error_report_{task_name}.json"
    
    if not os.path.exists(os.path.dirname(error_report_path)):
        os.makedirs(os.path.dirname(error_report_path))
    
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n💾 錯誤報告已儲存至: {error_report_path}")
    logging.info("="*60)

def generate_summary_error_report(all_error_stats):
    """生成所有實驗的總體錯誤報告"""
    if not all_error_stats:
        logging.info("\n📊 無錯誤統計資料")
        return
    
    logging.info("\n" + "="*60)
    logging.info("📊 總體錯誤統計報告")
    logging.info("="*60)
    
    # 計算總體統計
    total_questions = sum(stat['summary']['total_questions_processed'] for stat in all_error_stats)
    total_questions_with_errors = sum(stat['summary']['questions_with_errors'] for stat in all_error_stats)
    total_inferences = sum(stat['summary']['total_inferences'] for stat in all_error_stats)
    total_failures = sum(stat['summary']['total_failures'] for stat in all_error_stats)
    
    overall_failure_rate = (total_failures / total_inferences * 100) if total_inferences > 0 else 0
    
    # 顯示摘要
    logging.info(f"總實驗數: {len(all_error_stats)}")
    logging.info(f"總處理題數: {total_questions}")
    logging.info(f"有錯誤的題數: {total_questions_with_errors} ({total_questions_with_errors/total_questions*100:.1f}%)" if total_questions > 0 else "有錯誤的題數: 0")
    logging.info(f"總推論次數: {total_inferences:,}")
    logging.info(f"總失敗次數: {total_failures:,}")
    logging.info(f"整體失敗率: {overall_failure_rate:.2f}%")
    
    # 準備報告資料
    error_report = {
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "overall_summary": {
            "total_questions_processed": total_questions,
            "questions_with_errors": total_questions_with_errors,
            "total_inferences": total_inferences,
            "total_failures": total_failures,
            "overall_failure_rate": round(overall_failure_rate, 2)
        },
        "experiments": []
    }
    
    # 為每個實驗添加詳細資訊
    for stat in all_error_stats:
        experiment_data = {
            "task_name": stat['task_name'],
            "summary": stat['summary'],
            "error_details": stat['error_records']
        }
        error_report['experiments'].append(experiment_data)
        
        # 顯示每個實驗的統計
        logging.info(f"\n📋 {stat['task_name']}:")
        logging.info(f"   處理題數: {stat['summary']['total_questions_processed']}")
        logging.info(f"   有錯誤的題數: {stat['summary']['questions_with_errors']}")
        if stat['summary']['questions_with_errors'] > 0:
            logging.info(f"   失敗率: {stat['summary']['overall_failure_rate']:.2f}%")
            logging.info(f"   錯誤題目:")
            for error in stat['error_records'][:5]:  # 只顯示前 5 個
                logging.info(f"      - 問題 {error['question_index']}: {error['failed_inferences']}/{error['total_inferences']} 失敗 ({error['failure_rate']:.1f}%)")
            if len(stat['error_records']) > 5:
                logging.info(f"      ... 還有 {len(stat['error_records']) - 5} 個錯誤題目")
    
    # 儲存到 JSON 檔案（總體報告）
    error_report_path = f"results/error_report_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    if not os.path.exists(os.path.dirname(error_report_path)):
        os.makedirs(os.path.dirname(error_report_path))
    
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n💾 錯誤報告已儲存至: {error_report_path}")
    logging.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())