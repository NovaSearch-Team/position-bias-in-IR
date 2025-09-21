# 📍 Position Bias in Information Retrieval

Code and data for our paper:
**[An Empirical Study of Position Bias in Modern Information Retrieval](https://arxiv.org/abs/2505.13950)** (EMNLP 2025 Findings).


## 📘 Overview

This repository accompanies our paper, which investigates **position bias**—a persistent issue where modern retrieval models disproportionately prioritize early-passage content and systematically underweight information appearing further down.

**Research Question: How prevalent is position bias among today’s state-of-the-art IR models, and how does this bias manifest across different IR architectures?**

We propose a unified evaluation framework and introduce **two position-aware retrieval datasets** that reveal how popular retrieval models—including sparse (BM25), dense embedding-based, ColBERT-style, and reranker models—perform across different key information positions. Our analysis includes a diagnostic metric, **Position Sensitivity Index (PSI)**, to quantify this bias from a worst-case perspective.

## 📈 Key Findings

* **BM25** is robust to position shifts due to its position-agnostic nature.
* **Dense embedding models** and **ColBERT-style models** show **significant performance drops** as query-related content move to later sections.
* **Reranker models** are **largely immune to position bias**, making them a strong solution for mitigating early-passage overemphasis.

📄 See detailed results in our paper (Table 1).

## 📊 Datasets

We provide two publicly available datasets for position-aware retrieval:

### 🔹 [SQuAD-PosQ](https://huggingface.co/datasets/rajpurkar/squad_v2)

* Based on **SQuAD v2.0**, with questions grouped by the character-level start position of the answer.
* Enables fine-grained evaluation of models on shorter passages.
* Six positional buckets: `[0–100]`, `[100–200]`, `[200–300]`, `[300–400]`, `[400–500]`, `[500–3120]`.

### 🔹 [FineWeb-PosQ](https://huggingface.co/datasets/NovaSearch/FineWeb-PosQ)

* Constructed from the **FineWeb-edu** corpus, covering long passages (500–1024 words).
* Includes synthetic, position-targeted questions using GPT-based generation and a two-stage filtering pipeline.
* Three position buckets: `beginning`, `middle`, and `end`.

## 🧱 Code Structure

```text
position-bias-in-IR
├── exp_SQuAD-PosQ.py             # creation & evaluation for SQuAD-PosQ
├── exp_FineWeb-PosQ.py           # Evaluation for FineWeb-PosQ
├── utils.py                      # Top-K retrieval utils: BM25, dense, ColBERT-style, rerankers
├── commercial_embedding_api.py   # Wrappers for commercial embedding APIs
├── appendix_exp_cosine_sim.py    # Code for "A.4 Representation Behavior" in paper
```

## 🧪 Reproducing Our Experiments

We provide scripts to reproduce all benchmark evaluations:

```bash
# Sparse Retrievers
python exp_SQuAD-PosQ.py \
    --data_name_or_path "rajpurkar/squad_v2" \
    --score_type "bm25"

python exp_FineWeb-PosQ.py \
    --data_name_or_path "NovaSearch/FineWeb-PosQ" \
    --score_type "bm25"

# Dense Embedding-based Retrievers
python exp_SQuAD-PosQ.py \
    --data_name_or_path "rajpurkar/squad_v2" \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --model_type "local" \
    --score_type "single_vec"

python exp_FineWeb-PosQ.py \
    --data_name_or_path "NovaSearch/FineWeb-PosQ" \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --model_type "local" \
    --score_type "single_vec"

python exp_FineWeb-PosQ.py \
    --data_name_or_path "NovaSearch/FineWeb-PosQ" \
    --model_name_or_path "voyage" \
    --model_type "api" \
    --score_type "single_vec"

# ColBERT-style Late-interaction Models
python exp_SQuAD-PosQ.py \
    --data_name_or_path "rajpurkar/squad_v2" \
    --model_name_or_path "BAAI/bge-m3" \
    --model_type "local" \
    --score_type "multi_vec" \
    --query_sampling

python exp_FineWeb-PosQ.py \
    --data_name_or_path "NovaSearch/FineWeb-PosQ" \
    --model_name_or_path "BAAI/bge-m3" \
    --model_type "local" \
    --score_type "multi_vec" \
    --query_sampling

# Full-interaction Reranker Models
python exp_SQuAD-PosQ.py \
    --data_name_or_path "rajpurkar/squad_v2" \
    --model_name_or_path "Qwen/Qwen3-Reranker-0.6B" \
    --model_type "local" \
    --first_stage_model_name_or_path "bm25" \
    --first_stage_model_type "local" \
    --score_type "reranker" \
    --query_sampling 

python exp_FineWeb-PosQ.py \
    --data_name_or_path "NovaSearch/FineWeb-PosQ" \
    --model_name_or_path "Qwen/Qwen3-Reranker-0.6B" \
    --model_type "local" \
    --first_stage_model_name_or_path "nvidia/NV-embed-v2" \
    --first_stage_model_type "local" \
    --score_type "reranker" \
    --query_sampling
```

📌 **Primary Metrics**:
All experiments are evaluated using both **Recall** and **NDCG** to measure retrieval quality and ranking effectiveness.

We also propose the **Position Sensitivity Index (PSI)** to quantify position bias:

$$
\text{PSI} = 1 - \frac{\min(\text{group scores})}{\max(\text{group scores})}
$$

A lower PSI indicates greater robustness to answer position shifts.

📎 **Usage Tips**:

* Use `--query_sampling` for fast evaluation on the Tiny subsets (specially for ColBERT and reranker models).
* Use `--first_stage_model_name_or_path` to specify the retrieval backbone for reranker models (e.g., BM25, dense embedding models) to get Top-100 results.
  🔍 *Note*: The choice of the first-stage retriever **may slightly affect final scores** (compared to the paper), but **overall trends and conclusions remain consistent**.

For example, the metrics of `Qwen/Qwen3-Reranker-0.6B` on `FineWeb-PosQ` are as follows:

> model, #queries, span_class, Recall@5, Recall@10, Recall@20, Recall@30, Recall@50, Recall@100, NDCG@5, NDCG@10, NDCG@20, NDCG@30, NDCG@50, NDCG@100

> Qwen/Qwen3-Reranker-0.6B, 1000, beginning, 0.9580, 0.9610, 0.9610, 0.9610, 0.9610, 0.9610, 0.9493, **0.9503**, 0.9503, 0.9503, 0.9503, 0.9503

> Qwen/Qwen3-Reranker-0.6B, 1000, middle, 0.9740, 0.9800, 0.9810, 0.9820, 0.9830, 0.9830, 0.9479, **0.9497**, 0.9500, 0.9503, 0.9505, 0.9505

> Qwen/Qwen3-Reranker-0.6B, 1000, end, 0.9660, 0.9780, 0.9830, 0.9830, 0.9860, 0.9860, 0.9210, **0.9246**, 0.9260, 0.9260, 0.9266, 0.9266

> PSI value = 1 - 0.9246 / 0.9503 = 1 - 0.973 = 0.027 


## Citation

If you use our datasets or codebase, please cite:

```bibtex
@misc{zeng2025empiricalstudypositionbias,
      title={An Empirical Study of Position Bias in Modern Information Retrieval}, 
      author={Ziyang Zeng and Dun Zhang and Jiacheng Li and Panxiang Zou and Yudong Zhou and Yuqing Yang},
      year={2025},
      eprint={2505.13950},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.13950}, 
}
```