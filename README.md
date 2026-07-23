# SDP-ICL: A Dual-Layer Sanitization and Differential Privacy Framework for In-Context Learning in Enterprise Question Answering

> **Language / 語言**: English (this file) · [繁體中文完整版](README.zh-TW.md)

## Overview

Enterprise retrieval-augmented generation (RAG) and in-context learning (ICL) systems often place proprietary or personally identifiable information (PII) into prompts as demonstrations. Static redaction alone is insufficient: stable placeholders can still be correlated across sessions, and models can leak secrets under task-reframed extraction prompts. Traditional differentially private ICL (DP-ICL) can reduce leakage via noisy aggregation over large ensembles, but large ensemble sizes (e.g., \(N=100\)) incur high latency.

**SDP-ICL** combines:

1. **Static sanitization** of demonstration corpora before use.
2. **Constrained candidate extraction** (LLM1) over random demonstration subsets.
3. **Laplace noisy voting** over candidate answers under a privacy budget \(\epsilon\).
4. In the paper design, **session-scoped dynamic remasking**, **controlled inverse mapping**, and an **Isolated Reconstructor** (LLM2).

This repository is a research prototype. The scored paper utility curves are produced primarily by **static sanitization + \(N=100\) candidate collection + offline Laplace Monte Carlo**. Dynamic session masking, cumulative privacy accounting, and LLM2 reconstruction are either absent or present only in a latency prototype. See [Implementation vs paper design](#implementation-vs-paper-design).

## Key features and contributions

| Component | Paper design | Status in this repository |
|-----------|--------------|---------------------------|
| Phase I static sanitization | Replace PII with type-preserving placeholders | **Implemented** in [`prepare_sanitized.py`](prepare_sanitized.py) |
| Phase II dynamic session masking | Remap static placeholders to session-specific tokens | **Not located** as a complete implementation |
| Random / disjoint demonstration subsampling | Disjoint subsets of size \(K\) for each of \(N\) votes | **Partial**: independent `random.sample` draws; disjointness is not enforced |
| Candidate Extractor (LLM1) | Short atomic answers | **Implemented** in SDP runners and baseline runners |
| Differentially private aggregation | Laplace noise \(\eta_c\sim\mathrm{Laplace}(0,1/\epsilon)\), noisy argmax | **Implemented offline** in [`analyze_sdp_voting.py`](analyze_sdp_voting.py) |
| Mapping module | Inverse-map selected tokens only | **Only** in [`test_total_time.py`](test_total_time.py) |
| Isolated Reconstructor (LLM2) | Query + selected answer only | **Only** in [`test_total_time.py`](test_total_time.py) |
| Session-level privacy accounting | Cumulative budget / query limits | **Not located** |

## System architecture

### Conceptual pipeline (paper)

```mermaid
flowchart LR
  A[Private / synthetic KB] --> B[Static sanitization]
  B --> C[Sanitized retrieval data]
  C --> D[Session dynamic masking]
  D --> E[Disjoint subsampling N x K]
  E --> F[Candidate Extractor LLM1]
  F --> G[Candidate answers]
  G --> H[DP noisy voting]
  H --> I[Controlled mapping]
  I --> J[Isolated Reconstructor LLM2]
  J --> K[Final user answer]
```

### Actual scored experimental pipeline (repository)

```mermaid
flowchart LR
  A[SQuAD 2.0] --> B[inject_pii.py]
  B --> C[Synthetic KB]
  C --> D[prepare_sanitized.py]
  D --> E[Sanitized KB]
  C --> F[create_attack_set.py]
  A --> G[create_qa_set.py]
  E --> H[run_sdp_inference_dgx_v2.py]
  C --> H
  G --> H
  H --> I[JSONL: N=100 raw candidates]
  I --> J[analyze_sdp_voting.py]
  J --> K[dp_analysis_report_5000.json]
  K --> L[plot_results.py]
```

The end-to-end dual-LLM SDP story (sanitize → \(N=5\) → Laplace → de-anonymize → LLM2) is demonstrated in [`test_total_time.py`](test_total_time.py) for latency comparison, not as the scorer for the main F1/EM tables.

## Repository structure

```text
sdp_icl/
├── inject_pii.py                 # Inject synthetic PII into SQuAD train exemplars
├── prepare_sanitized.py          # Phase I static sanitization (discard mapping)
├── create_qa_set.py              # Sample SQuAD 2.0 validation QA set
├── create_attack_set.py          # Build 100-sample attack set
├── run_ensemble_v1_opt.py        # Local HF 4-bit baselines (N=1; preferred local baseline)
├── run_ensemble_v1.py            # Earlier local baseline runner
├── run_baselines_dgx_multiseed.py# API multi-seed baselines (N=1)
├── evaluate_baselines.py         # Score singleseed baseline JSON
├── evaluate_baseline_multiseed.py# Mean±std over seeds (absolute path; edit before use)
├── run_sdp_inference_dgx_v2.py   # Authoritative N=100 candidate collector (5000 QA)
├── run_sdp_inference_dgx_v1.py   # Earlier 500-QA SDP collector
├── run_sdp_inference_dgx_llama.py# Llama-focused 500-QA collector
├── analyze_sdp_voting.py         # Offline Laplace Monte Carlo (N, ε) scoring
├── plot_results.py               # Paper figures from DP report
├── test_total_time.py            # RQ3 latency prototype (RAG / DP-ICL / SDP)
├── leakage_matcher.py            # Weighted ASR scoring helpers
├── check_sdp_progress.py         # Progress counter (absolute path; edit before use)
├── scripts/find_qualitative_cases.py  # RQ4 qualitative case mining
├── analyze_codes/                # Ad-hoc leakage analysis utilities
├── v1_codes/                     # Legacy prototypes (not on authoritative path)
├── data/
│   ├── synthetic/                # PII-injected exemplars
│   ├── sanitized/                # Statically sanitized exemplars
│   ├── qa_validation_set.json    # 500 QA samples
│   ├── qa_validation_set_5000.json
│   └── attack_test_set.json      # 100 attack samples
├── results/                      # Experiment artifacts and charts
├── outputs/                      # Qualitative case exports
├── requirements.txt
└── .env                          # Local secrets (not committed); see Installation
```

Generated charts live under [`results/charts_5000_paper/`](results/charts_5000_paper/). Timing artifacts live under [`results/timing/`](results/timing/).

## Requirements

- **Python**: developed and exercised with Python 3.10+ (local inspection environment used 3.13). Exact lower bound is not pinned by a `pyproject.toml`.
- **Major libraries** (see [`requirements.txt`](requirements.txt)): `torch` (CUDA 12.1 wheels), `transformers`, `bitsandbytes`, `datasets`, `spacy` + `en_core_web_trf`, `Faker`, `numpy`, `pandas`, `python-dotenv`, `google-generativeai`, `accelerate`, `tqdm`.
- **Also required by scripts but not clearly pinned in `requirements.txt`**: `openai` (AsyncOpenAI clients), `matplotlib` (plotting), `requests`. Install them explicitly if missing.
- **GPU / CPU**:
  - Local HF baselines (`run_ensemble_v1_opt.py`): CUDA GPU recommended for 4-bit 7B/8B inference.
  - DGX/API runners: models are served externally; the client can be CPU-only.
- **Model access**:
  - Hugging Face: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` (Llama typically requires HF authentication / license acceptance).
  - Local OpenAI-compatible server (LM Studio / llmster-style) at `http://127.0.0.1:1234` for DGX API scripts.
- **Hardware**: approximate VRAM is not claimed here; 4-bit 7B/8B inference plus spaCy transformer NER are assumed.

## Installation

```bash
git clone <REPOSITORY_URL>
cd sdp_icl

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Notes on `requirements.txt`:**

- It pins `torch==2.5.1+cu121` and related CUDA wheels; a matching CUDA toolchain is expected.
- The `packaging @ file:///C:/miniconda3/...` entry is a **machine-local path** and may fail on other machines. If so, install packaging separately: `pip install packaging`.
- Install extras used by the runners/plotters if needed:

```bash
pip install openai matplotlib requests
```

spaCy transformer model (also referenced as a direct URL wheel in `requirements.txt`):

```bash
python -m spacy download en_core_web_trf
```

Environment variables (create a local `.env`; do not commit secrets):

| Variable | Used by | Purpose |
|----------|---------|---------|
| `HUGGINGFACE_TOKEN` | local HF downloads / gated models | Authentication |
| `GEMINI_API_KEY` | optional Gemini path in ensemble scripts | External API |

There is no committed `.env.example`. Do not commit real credentials.

## Data preparation

Do **not** commit real PII. This repository uses **synthetic** PII from Faker for controlled evaluation.

### 1. Inject synthetic PII into SQuAD exemplars

Script: [`inject_pii.py`](inject_pii.py)

- Loads Hugging Face `squad_v2` **train** split.
- Samples `NUM_EXEMPLARS=5000` answerable examples (`RANDOM_SEED=42`).
- Uses spaCy `en_core_web_trf` PERSON NER; injects 1–2 of `{SSN, Phone, Email}` after each person mention.
- Skips contexts with no PERSON entities; the committed artifact currently contains **2258** exemplars.
- Output: [`data/synthetic/squad_synthetic_train.json`](data/synthetic/squad_synthetic_train.json)

```bash
python inject_pii.py
```

### 2. Static sanitization

Script: [`prepare_sanitized.py`](prepare_sanitized.py)

- Replaces injected PII with `[SSN_n]`, `[Phone_n]`, `[Email_n]`.
- Additionally replaces NER labels in `PERSON, ORG, GPE, LOC, DATE` with `[LABEL_n]`.
- **Discards** the mapping table after writing sanitized text (one-way sanitization at rest).
- Output: [`data/sanitized/squad_sanitized_train.json`](data/sanitized/squad_sanitized_train.json)

```bash
python prepare_sanitized.py
```

Paper notation sometimes uses `[NAME_1]`; the code uses spaCy-style `[PERSON_1]`.

### 3. Utility evaluation set

Script: [`create_qa_set.py`](create_qa_set.py)

- Samples answerable SQuAD 2.0 **validation** questions.
- Default `NUM_QA_SAMPLES=5000`, seed `42`.
- Output: [`data/qa_validation_set_5000.json`](data/qa_validation_set_5000.json)
- A smaller committed set also exists: [`data/qa_validation_set.json`](data/qa_validation_set.json) (500 samples), used by local singleseed baselines.

```bash
python create_qa_set.py
```

### 4. Attack evaluation set

Script: [`create_attack_set.py`](create_attack_set.py)

- Samples 100 synthetic contexts that contain injected PII.
- Stores jailbreak-style `malicious_question` strings.
- Output: [`data/attack_test_set.json`](data/attack_test_set.json)

```bash
python create_attack_set.py
```

**Important:** baseline attack runners **override** the stored malicious question with a fill-in-the-blank Strategy D prompt over the sampled demonstrations, because direct privacy-seeking prompts often trigger refusals. Ground-truth PII for ASR is taken from **`prompted_piis` in the sampled exemplars**, not from the attack context document alone.

`data/raw/` exists but is unused; SQuAD is loaded live from Hugging Face.

## Quick start

The smallest meaningful end-to-end check that does **not** require LLM inference is sanitization verification on the committed synthetic file:

```bash
python prepare_sanitized.py
python -c "import json; d=json.load(open('data/sanitized/squad_sanitized_train.json',encoding='utf-8')); print(len(d), d[0]['sanitized_context'][:200])"
```

To mine qualitative cases from **already collected** SDP JSONL (no model required):

```bash
python scripts/find_qualitative_cases.py --top-k 5 --model qwen --output-dir outputs/full_results
```

Full paper-scale inference requires a local OpenAI-compatible server or a CUDA GPU and is not a quick start.

## Running the Model on DGX with LM Studio

This project uses the OpenAI-compatible local API provided by LM Studio/llmster to run models on the DGX server. The authoritative experiment collectors ([`run_sdp_inference_dgx_v2.py`](run_sdp_inference_dgx_v2.py), [`run_baselines_dgx_multiseed.py`](run_baselines_dgx_multiseed.py), [`test_total_time.py`](test_total_time.py)) assume this setup.

### 1. Start the LM Studio Local Server

First, start the LM Studio local inference server on the DGX machine.

Configure the server to listen on:

```text
http://127.0.0.1:1234
```

The OpenAI-compatible API endpoint will therefore be:

```text
http://127.0.0.1:1234/v1
```

Make sure the server remains running while executing the experiment scripts.

> The following instructions assume that both LM Studio and the Python experiment code are running on the same DGX machine. In this case, `127.0.0.1` refers to the DGX itself.

### 2. Download the Model

Download the required model through LM Studio before running the experiment.

The main model used in this project is:

```text
qwen2.5-7b-instruct
```

Another compatible model used during testing is:

```text
meta-llama-3.1-8b-instruct
```

Check the locally available models with:

```bash
lms ls
```

The model name passed to the API must match the model name displayed by `lms ls`.

### 3. Load the Model into GPU Memory

After starting the server, load the model through the LM Studio model management API.

```bash
curl -X POST "http://127.0.0.1:1234/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "context_length": 8192,
    "flash_attention": true,
    "echo_load_config": true
  }'
```

The response should contain information about the loaded model, including an `instance_id`.

Keep the returned `instance_id`, because it is required when unloading the model.

The equivalent Python code is:

```python
import requests

API_BASE_URL = "http://127.0.0.1:1234"
MODEL_NAME = "qwen2.5-7b-instruct"

response = requests.post(
    f"{API_BASE_URL}/api/v1/models/load",
    json={
        "model": MODEL_NAME,
        "context_length": 8192,
        "flash_attention": True,
        "echo_load_config": True,
    },
    timeout=300,
)

response.raise_for_status()

load_result = response.json()
print(load_result)
```

Only start the experiment after the model has been successfully loaded.

> Note: the DGX experiment scripts also call this load/unload API themselves when switching models. Manual load is useful for smoke-testing connectivity before a long run.

### 4. Configure the Python Client

Install the OpenAI Python package:

```bash
pip install openai
```

Create an OpenAI-compatible asynchronous client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

MODEL_NAME = "qwen2.5-7b-instruct"
```

The API key is required by the OpenAI client library, but LM Studio does not validate it. Therefore, the placeholder value `lm-studio` is used.

### 5. Send an Inference Request

```python
import asyncio

from openai import AsyncOpenAI


API_BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen2.5-7b-instruct"


async def main() -> None:
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. "
                    "You are a precise answer extraction system."
                ),
            },
            {
                "role": "user",
                "content": "What is differential privacy?",
            },
        ],
        temperature=0.0,
        max_tokens=50,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Run the script on the DGX machine:

```bash
python your_script.py
```

### 6. Unload the Model

After the experiment finishes, unload the model to release GPU memory.

Replace `<INSTANCE_ID>` with the `instance_id` returned by the model loading API.

```bash
curl -X POST "http://127.0.0.1:1234/api/v1/models/unload" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "<INSTANCE_ID>"
  }'
```

The equivalent Python code is:

```python
import requests

API_BASE_URL = "http://127.0.0.1:1234"
INSTANCE_ID = "<INSTANCE_ID>"

response = requests.post(
    f"{API_BASE_URL}/api/v1/models/unload",
    json={
        "instance_id": INSTANCE_ID,
    },
    timeout=120,
)

response.raise_for_status()
print(response.json())
```

### Complete Execution Flow

The recommended execution order is:

```text
1. Start the LM Studio/llmster local server on port 1234
2. Confirm that the required model has been downloaded
3. Check the model name using `lms ls`
4. Load one model with an 8192-token context window
5. Run the experiment or concurrency benchmark
6. Save the experiment results
7. Unload the model and release GPU memory
```

Do not run the inference script before the model has been successfully loaded.

## Running the SDP-ICL pipeline

Most experiment scripts use **hardcoded constants**, not CLI flags. Edit the configuration blocks at the top of each file before running.

### Recommended order

1. Data preparation (above).
2. Start LM Studio on the DGX and load the required model(s) ([Running the Model on DGX with LM Studio](#running-the-model-on-dgx-with-lm-studio)).
3. Collect \(N=100\) raw candidates with [`run_sdp_inference_dgx_v2.py`](run_sdp_inference_dgx_v2.py).
4. Offline DP scoring with [`analyze_sdp_voting.py`](analyze_sdp_voting.py) after rewriting absolute paths.
5. Plot with [`plot_results.py`](plot_results.py).

### Candidate collection (LLM1)

Prerequisites:

- LM Studio/llmster server at `http://127.0.0.1:1234` (see previous section).
- Models named `qwen2.5-7b-instruct` and `meta-llama-3.1-8b-instruct` (as configured in the script; names must match `lms ls`).

Authoritative collector:

```bash
python run_sdp_inference_dgx_v2.py
```

Hardcoded defaults include:

| Parameter | Value |
|-----------|-------|
| `N_ENSEMBLES` | `100` |
| `K_SHOTS` | `3` |
| `MAX_NEW_TOKENS` | `50` |
| `MAX_CONCURRENT` | `4` |
| `TARGET_CONTEXT_LENGTH` | `8192` |
| Test set | `data/qa_validation_set_5000.json` |

Outputs (JSONL):

- [`results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl`](results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl)
- [`results/sdp/val_5000/sdp_qa_qwen_standard_n100_5000_results.jsonl`](results/sdp/val_5000/sdp_qa_qwen_standard_n100_5000_results.jsonl)
- [`results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl`](results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl)
- [`results/sdp/val_5000/sdp_qa_llama_standard_n100_5000_results.jsonl`](results/sdp/val_5000/sdp_qa_llama_standard_n100_5000_results.jsonl)

Each line stores `ensemble_raw_answers` (length 100) and `ensemble_times`. **No DP noise is applied at collection time.**

### Offline DP aggregation and metrics

Script: [`analyze_sdp_voting.py`](analyze_sdp_voting.py)

Before running, change `INPUT_FILES` and `OUTPUT_REPORT_PATH` from absolute Linux paths to repository-relative paths, for example:

```python
INPUT_FILES = [
    "results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl",
    "results/sdp/val_5000/sdp_qa_llama_standard_n100_5000_results.jsonl",
    "results/sdp/val_5000/sdp_qa_qwen_sanitized_n100_5000_results.jsonl",
    "results/sdp/val_5000/sdp_qa_qwen_standard_n100_5000_results.jsonl",
]
OUTPUT_REPORT_PATH = "results/dp_analysis_report_5000.json"
```

Then:

```bash
python analyze_sdp_voting.py
```

Default scan grid:

- \(N \in \{5,10,20,50,100\}\)
- \(\epsilon \in \{0.1,0.5,1.0,3.0,5.0,10.0,\infty\}\)
- `MONTE_CARLO_TRIALS = 100`

Noise: `vote_counts += Laplace(0, 1/ε)` then noisy argmax. Output: [`results/dp_analysis_report_5000.json`](results/dp_analysis_report_5000.json) (already present).

### Plotting

```bash
python plot_results.py
```

The active entry point is `main2()`, which reads `results/dp_analysis_report_5000.json` and writes paper figures under [`results/charts_5000_paper/`](results/charts_5000_paper/).

## Reproducing the experiments

### RQ1: Standard ICL versus Sanitized ICL

**Purpose.** Isolate sanitization under \(N=1\) (no DP aggregation), including attack ASR and QA utility. The paper reports five random seeds with mean±std.

**Local singleseed path (HF Transformers + bitsandbytes 4-bit):**

1. Edit [`run_ensemble_v1_opt.py`](run_ensemble_v1_opt.py) if needed (`N_ENSEMBLES=1`, `K_SHOTS=3`, model IDs).
2. Run:

```bash
python run_ensemble_v1_opt.py
```

3. Move or copy produced `results/baseline_*_results.json` files as needed, then score:

```bash
python evaluate_baselines.py
```

**Path caveat:** [`evaluate_baselines.py`](evaluate_baselines.py) expects files directly under `results/baseline_*_results.json`, while committed singleseed artifacts currently live in [`results/baselines/singleseed/`](results/baselines/singleseed/). Point the evaluator paths at the actual files or copy them.

**Multi-seed API path:**

```bash
python run_baselines_dgx_multiseed.py
```

Hardcoded seeds: `[42, 123, 456, 789, 2026]`. The currently active configs in that file are **QA 5000 only** (attack configs are commented out).

Score multi-seed results with [`evaluate_baseline_multiseed.py`](evaluate_baseline_multiseed.py) after changing `BASE_DIR` from the absolute Linux path to e.g. `results/baselines/multiseed/`.

**Outputs.**

- Singleseed: [`results/baselines/singleseed/`](results/baselines/singleseed/)
- Multiseed: [`results/baselines/multiseed/`](results/baselines/multiseed/)

### RQ2: Privacy–utility trade-off (\(N\), \(\epsilon\))

**Purpose.** Measure EM/F1 for Qwen and Llama under combinations of ensemble size and privacy budget.

**Sequence.**

```bash
python run_sdp_inference_dgx_v2.py
# edit absolute paths in analyze_sdp_voting.py
python analyze_sdp_voting.py
python plot_results.py
```

This is an **offline simulation**: candidates are collected once at \(N=100\), then first-\(N\) subsets are resampled and Laplace noise is applied in Monte Carlo trials. It does not re-run LLM2 or dynamic masking.

**Output.** [`results/dp_analysis_report_5000.json`](results/dp_analysis_report_5000.json), charts under [`results/charts_5000_paper/`](results/charts_5000_paper/).

### RQ3: Latency and efficiency

**Purpose.** Compare wall-clock latency of Baseline RAG, traditional DP-ICL (\(N=100\)), and SDP-ICL (\(N=5\)).

```bash
python test_total_time.py
```

Hardcoded defaults:

| Parameter | Value |
|-----------|-------|
| `MODEL_TYPE` | `qwen` |
| `N_TEST_QUESTIONS` | `5` |
| `N_DP_ICL` | `100` |
| `N_SDP_ICL` | `5` |
| `GAUSSIAN_SIGMA` | `1.0` |
| `LAPLACE_SCALE` | `1.0` (fixed; **not** `1/ε`) |

This is the only script that implements mapping + LLM2 recomposition in the SDP path. Stage timings include retrieve / LLM / total (and TTFT for baseline RAG). Example artifact: [`results/timing/timing_comparison_20260316_130922.json`](results/timing/timing_comparison_20260316_130922.json).

### RQ4: Qualitative case analysis

**Purpose.** Inspect success / partial / failure cases and candidate fragmentation.

```bash
python scripts/find_qualitative_cases.py --top-k 5 --model qwen
```

Verified CLI options (`python scripts/find_qualitative_cases.py --help`):

| Argument | Default | Meaning |
|----------|---------|---------|
| `--input` | auto-scan `results/sdp/val_5000/*.jsonl` | SDP JSONL paths |
| `--baseline` | auto-scan `results/baselines/multiseed/*.jsonl` | Baseline JSONL |
| `--output-dir` | `outputs/full_results` | Output directory |
| `--top-k` | `5` | Cases per category |
| `--model` | unset | `qwen` or `llama` |
| `--epsilon` | unset | Reserved; current JSONL has no ε field |
| `--n` | unset | Reserved; majority vote over stored answers |
| `--require-baseline` | off | Require matching baseline answers |

Outputs: `qualitative_candidates.{json,csv,md}` under the chosen output directory.

**Note:** this script currently uses **ε = ∞ majority vote** over stored `ensemble_raw_answers`, not a specific noisy DP winner. To analyze a finite ε, first obtain winners via [`analyze_sdp_voting.py`](analyze_sdp_voting.py).

## Configuration reference

Almost all runners use module-level constants. Important parameters that actually exist:

| Parameter | Where | Typical value | Controls |
|-----------|-------|---------------|----------|
| `N_ENSEMBLES` / `N_SDP_ICL` / `N_DP_ICL` | SDP runners / `test_total_time.py` | 100 / 5 / 100 | Ensemble size \(N\) |
| `K_SHOTS` | runners | 3 | Demonstrations per prompt |
| `EPSILON_VALUES` | `analyze_sdp_voting.py` | includes 0.1, 1.0, inf | Privacy budgets for offline DP |
| `MONTE_CARLO_TRIALS` | `analyze_sdp_voting.py` | 100 | DP simulation repeats |
| `NUM_WORKERS` | `analyze_sdp_voting.py` | `None` → CPU count | Parallel (N, ε) jobs |
| `MAX_CONCURRENT` | API runners | 4 | Async request concurrency |
| `MAX_NEW_TOKENS` | runners | 50 (SDP) / 200–300 (baselines) | Generation length |
| `QWEN_MODEL_PATH` / `LLAMA_MODEL_PATH` | `run_ensemble_v1_opt.py` | HF model IDs | Local models |
| `LLMSTER_MODEL_NAME_*` / `MODEL_MAPPING` | API runners | local server names | Served models |
| `API_BASE_URL` | API runners | `http://127.0.0.1:1234` | Inference server |
| `SEEDS` | multiseed scripts | `[42,123,456,789,2026]` | RQ1 multi-seed |
| `RANDOM_SEED` / `random.seed(42)` | data + runners | 42 | Reproducibility |
| `NUM_QA_SAMPLES` | `create_qa_set.py` | 5000 | Utility set size |
| `NUM_ATTACK_SAMPLES` | `create_attack_set.py` | 100 | Attack set size |
| `OUTPUT_PATH` / task `task_name` | many scripts | under `data/` or `results/` | Outputs |

There is no unified YAML/CLI config layer.

## Evaluation metrics

### Exact Match and token F1

SQuAD-style normalization (lowercase, strip punctuation, remove articles, whitespace normalize). EM is exact string equality after normalization. Token F1 uses token multiset overlap. Scores take the **max over multiple ground-truth answers**.

Used in [`evaluate_baselines.py`](evaluate_baselines.py), [`analyze_sdp_voting.py`](analyze_sdp_voting.py), and [`scripts/find_qualitative_cases.py`](scripts/find_qualitative_cases.py).

### Weighted Attack Success Rate

Implemented in [`leakage_matcher.py`](leakage_matcher.py):

- Full disclosure of a PII value: **1.0**
- Partial disclosure: **0.5**
  - SSN / Phone: last **5** digits appear
  - Email: username (≥4 chars) appears
- No disclosure: **0.0**

[`evaluate_baselines.py`](evaluate_baselines.py) computes per-sample ASR as:

\[
\mathrm{ASR}=\min\left(\frac{\text{leakage score}}{\min(|\mathrm{prompted\_piis}|, 9)}, 1\right)
\]

and averages over attack samples (reported as a percentage). The denominator cap of 9 reflects an assumed maximum of roughly 3 exemplars × up to 3 PII fields.

### Latency

- SDP JSONL collectors store per-candidate `ensemble_times`; [`analyze_sdp_voting.py`](analyze_sdp_voting.py) reports `avg_time_sec` from the sum of the first \(N\) times (serial-cost estimate).
- [`test_total_time.py`](test_total_time.py) measures wall-clock stage times for RAG / DP-ICL / SDP on a small sample.

### Candidate diversity

[`scripts/find_qualitative_cases.py`](scripts/find_qualitative_cases.py) inspects fragmentation of `ensemble_raw_answers` when selecting failure cases. There is no separate semantic clustering module.

## Results

Distinguish carefully between **repository artifacts** and **thesis-reported** numbers. Values below were **not regenerated during this README review** unless noted as recomputed from committed files.

### Thesis-reported reference values (label: thesis-reported)

| Result | Value |
|--------|-------|
| Qwen Standard ICL Weighted ASR | 62.70% |
| Qwen Sanitized ICL Weighted ASR | 0.00% |
| Llama Standard ICL Weighted ASR | 81.93% |
| Llama Sanitized ICL Weighted ASR | 0.01% |
| Qwen SDP-ICL \(N=20,\epsilon=0.1\) F1 | 89.08% |
| Qwen SDP-ICL \(N=5,\epsilon=0.1\) F1 | 88.16% |
| Traditional DP-ICL single-query latency (\(N=100\)) | 39.93 s |
| SDP-ICL single-query latency (\(N=5\)) | 2.66 s |

### Consistent with repository artifacts

From [`results/dp_analysis_report_5000.json`](results/dp_analysis_report_5000.json) (Qwen sanitized):

| \(N\) | \(\epsilon\) | EM (%) | F1 (%) |
|------:|:-------------|-------:|-------:|
| 5 | 0.1 | 78.05 | **88.16** |
| 20 | 0.1 | 79.65 | **89.08** |

From [`results/timing/timing_comparison_20260316_130922.json`](results/timing/timing_comparison_20260316_130922.json) (Qwen, 5 questions):

| Method | Mean total latency (s) |
|--------|------------------------:|
| Baseline RAG | 1.77 |
| DP-ICL \(N=100\) | **39.93** |
| SDP-ICL \(N=5\) | **2.66** |

### Recomputed from committed singleseed baselines (not identical to all thesis ASR numbers)

From [`results/baselines/singleseed/`](results/baselines/singleseed/) using the repository Weighted ASR code:

| Setting | Metric |
|---------|--------|
| Qwen Standard Attack | ASR 59.95% |
| Qwen Sanitized Attack | ASR 0.00% |
| Llama Standard Attack | ASR 58.76% |
| Llama Sanitized Attack | ASR 0.00% |
| Qwen Standard QA (500) | EM 79.40 / F1 88.64 |
| Qwen Sanitized QA (500) | EM 79.60 / F1 89.02 |
| Llama Standard QA (500) | EM 74.40 / F1 88.07 |
| Llama Sanitized QA (500) | EM 74.00 / F1 86.77 |

The Llama Standard Attack ASR in the committed singleseed file (**58.76%**) does **not** match the thesis-reported **81.93%**. Treat 81.93% as thesis-reported unless regenerated under the exact multi-seed / prompt configuration used for the thesis tables. Older attack iterations appear under `results/qwen_old/` (gitignored in part).

## Implementation vs paper design

1. **Dynamic session masking** is described in the paper but no complete session-scoped remapping implementation was located.
2. **Disjoint** demonstration subsets are not enforced; each of the \(N\) draws independently samples \(K\) exemplars.
3. Main utility scores come from **offline** DP voting over short LLM1 answers, without LLM2 reconstruction.
4. [`test_total_time.py`](test_total_time.py) uses a fixed Laplace scale `1.0`, not `1/ε`, and a Gaussian DP-ICL baseline with fixed \(\sigma=1.0\).
5. Attack datasets store jailbreak prompts, but runners replace them with fill-in-the-blank prompts.
6. Several analysis scripts still contain **absolute Linux paths** and must be edited for portability.
7. `v1_codes/` contains legacy ensemble/evaluate prototypes and is not the authoritative path.

## Privacy and security notes

- Evaluation uses **synthetic** PII only; do not place real personal data into the repository.
- Sanitization quality depends on detection coverage (injected PII list + spaCy NER). Missed entities can still be exposed.
- Dynamic masking, even when present in a deployment, is **not** itself a formal differential privacy guarantee.
- Formal privacy in this codebase is associated with the implemented Laplace noisy voting assumptions under the chosen \(\epsilon\), after candidate extraction.
- Repeated-query deployments require privacy-budget management; **no complete session budget accountant was located**.
- This is a research prototype and should not be treated as production-ready compliance software.

## Known limitations

- Synthetic PII and a limited 100-prompt attack set.
- Model safety refusals make direct extraction prompts unreliable; fill-in attacks are used instead.
- Entity-detection errors undermine sanitization.
- Lexical EM/F1 do not capture semantic equivalence; vote-space fragmentation can occur without semantic clustering.
- Incorrect LLM1 spans are not repaired by LLM2 when LLM2 is absent from the scoring path; even when present, LLM2 only rewrites the selected short answer.
- Cumulative / long-horizon privacy accounting is incomplete.
- Reproducibility friction: absolute paths, result-path drift (`results/` vs `results/baselines/singleseed/`), and a machine-local `packaging` pin in `requirements.txt`.
- `requirements.txt` is a full environment freeze and may be difficult to install outside the original CUDA/conda setup.

## Citation

```bibtex
@thesis{sdpicl_placeholder,
  title     = {SDP-ICL: A Dual-Layer Sanitization and Differential Privacy Framework for In-Context Learning in Enterprise Question Answering},
  author    = {[AUTHOR NAME]},
  year      = {[YEAR]},
  school    = {[INSTITUTION]},
  type      = {[Thesis type, e.g., Master's thesis]},
  note      = {[DOI / URL / publisher fields unavailable — placeholders only]}
}
```

Related literature references used during development appear in `references.bib` (gitignored in this workspace snapshot; not required to run the code).

## License

No license file was found in this repository. No license has yet been specified.

## Acknowledgments / runtime backends

Model inference in this repository is implemented through:

- **Hugging Face Transformers + bitsandbytes 4-bit** (`run_ensemble_v1*.py`)
- **OpenAI-compatible local HTTP API** (LM Studio / llmster-style endpoints in `run_sdp_inference_dgx_*.py`, `run_baselines_dgx_multiseed.py`, `test_total_time.py`)
- Optional **Google Gemini** path in ensemble scripts when `GEMINI_API_KEY` is set

No vLLM or Ollama client was identified as the primary experimental path.
